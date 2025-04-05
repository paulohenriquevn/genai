import os
import re
import logging
import sqlglot
from sqlglot import select, parse_one, exp
from sqlglot.errors import ParseError
from sqlglot.expressions import Subquery, Table, Column, Expression
from sqlglot.optimizer.normalize_identifiers import normalize_identifiers
from sqlglot.optimizer.qualify_columns import quote_identifiers
from typing import Dict, List, Optional, Any, Set, Tuple

# Importação relativa somente uma vez da classe base
from .query_builder_base import BaseQueryBuilder
from connector.semantic_layer_schema import SemanticSchema


class LocalQueryBuilder(BaseQueryBuilder):
    """
    Construtor de queries para arquivos locais como CSV e Parquet.
    
    Este construtor gera queries SQL que podem ser executadas em
    mecanismos de processamento local como DuckDB ou SQLite.
    """
    
    def __init__(self, schema: SemanticSchema, dataset_path: str):
        """
        Inicializa o construtor de queries locais.
        
        Args:
            schema: Esquema semântico para a construção da query.
            dataset_path: Caminho base para os datasets.
        """
        super().__init__(schema)
        self.dataset_path = dataset_path
    
    def _get_columns(self) -> List[str]:
        """
        Obtém a lista de colunas para a query SELECT.
        
        Returns:
            List[str]: Lista de expressões de colunas.
        """
        if not hasattr(self.schema, 'columns') or not self.schema.columns:
            return ["*"]
        
        columns = []
        for col in self.schema.columns:
            # Define a expressão da coluna
            if hasattr(col, 'expression') and col.expression:
                column_expr = col.expression
            else:
                column_expr = normalize_identifiers(col.name).sql()
            
            # Aplica transformações específicas para esta coluna
            if hasattr(self.schema, 'transformations') and self.schema.transformations:
                column_expr = self.transformation_manager.apply_column_transformations(
                    column_expr, col.name, self.schema.transformations
                )
                if hasattr(col, 'alias') and not col.alias:
                    col.alias = normalize_identifiers(col.name).sql()
            
            # Adiciona alias se especificado
            if hasattr(col, 'alias') and col.alias:
                column_expr = f"{column_expr} AS {col.alias}"
            
            columns.append(column_expr)
        
        return columns
    
    def _get_table_expression(self) -> str:
        """
        Obtém a expressão de tabela para a cláusula FROM.
        
        Returns:
            str: Expressão de tabela.
            
        Raises:
            ValueError: Se o formato do arquivo não for suportado.
        """
        # Verifica se a configuração para o caminho da fonte de dados está disponível
        if not hasattr(self.schema, 'source_path') or not self.schema.source_path:
            raise ValueError("Source path não definido no esquema")
        
        # Obtém o tipo de fonte e caminho
        source_type = self.schema.source_type.lower()
        filepath = os.path.join(self.dataset_path, self.schema.source_path)
        abspath = os.path.abspath(filepath)
        
        # Gera expressão específica para o tipo de fonte
        if source_type == "parquet":
            return f"read_parquet('{abspath}')"
        elif source_type == "csv":
            return f"read_csv('{abspath}')"
        else:
            raise ValueError(f"Formato de arquivo não suportado: {source_type}")


class SqlQueryBuilder(BaseQueryBuilder):
    """
    Construtor de queries para bancos de dados SQL.
    
    Este construtor gera queries SQL para serem executadas em
    sistemas de banco de dados relacionais.
    """
    
    def __init__(self, schema: SemanticSchema):
        """
        Inicializa o construtor de queries SQL.
        
        Args:
            schema: Esquema semântico para a construção da query.
        """
        super().__init__(schema)
    
    def _get_columns(self) -> List[str]:
        """
        Obtém a lista de colunas para a query SELECT.
        
        Returns:
            List[str]: Lista de expressões de colunas.
        """
        if not hasattr(self.schema, 'columns') or not self.schema.columns:
            return ["*"]
        
        columns = []
        for col in self.schema.columns:
            # Define a expressão da coluna
            if hasattr(col, 'expression') and col.expression:
                column_expr = col.expression
            else:
                column_expr = normalize_identifiers(col.name).sql()
            
            # Aplica transformações específicas para esta coluna
            if hasattr(self.schema, 'transformations') and self.schema.transformations:
                column_expr = self.transformation_manager.apply_column_transformations(
                    column_expr, col.name, self.schema.transformations
                )
                if hasattr(col, 'alias') and not col.alias:
                    col.alias = normalize_identifiers(col.name).sql()
            
            # Adiciona alias se especificado
            if hasattr(col, 'alias') and col.alias:
                column_expr = f"{column_expr} AS {col.alias}"
            
            columns.append(column_expr)
        
        return columns
    
    def _get_table_expression(self) -> str:
        """
        Obtém a expressão de tabela para a cláusula FROM.
        
        Returns:
            str: Expressão de tabela.
        """
        # Verifica se temos informações sobre a tabela de origem
        if hasattr(self.schema, 'source_table'):
            table_name = self.schema.source_table
        else:
            # Fallback para o nome do esquema como nome da tabela
            table_name = self.schema.name
        
        return normalize_identifiers(table_name.lower()).sql()


class ViewQueryBuilder(BaseQueryBuilder):
    """
    Construtor de queries para views.
    
    Este construtor gera queries SQL para criar views personalizadas
    combinando dados de múltiplas fontes através de relacionamentos.
    """
    
    def __init__(
        self,
        schema: SemanticSchema,
        schema_dependencies_dict: Dict[str, Any]
    ):
        """
        Inicializa o construtor de queries para views.
        
        Args:
            schema: Esquema semântico para a construção da view.
            schema_dependencies_dict: Dicionário de dependências de esquema.
        """
        super().__init__(schema)
        self.schema_dependencies_dict = schema_dependencies_dict
        self.logger = logging.getLogger(f"{self.__class__.__name__}[{schema.name}]")
        self._table_alias_map = {}  # Mapeamento de aliases de tabela
        self._column_source_map = {}  # Mapeamento de colunas para suas tabelas de origem
        
        # Inicializa mapeamentos de dados
        self._init_mappings()
    
    def _init_mappings(self) -> None:
        """
        Inicializa mapeamentos de tabelas e colunas baseados no esquema.
        """
        # Mapeia aliases de tabelas
        if hasattr(self.schema, 'relations') and self.schema.relations:
            for relation in self.schema.relations:
                # Extrai nomes de tabelas da relação
                source_parts = relation.source_table.split(".")
                target_parts = relation.target_table.split(".")
                
                # Registra aliases de tabela
                source_table = source_parts[0]
                target_table = target_parts[0]
                
                self._table_alias_map[source_table] = source_table
                self._table_alias_map[target_table] = target_table
        
        # Mapeia colunas para suas tabelas de origem
        for col in self.schema.columns:
            col_name = col.name
            if "." in col_name:
                parts = col_name.split(".")
                if len(parts) >= 2:
                    table_name = parts[0]
                    field_name = ".".join(parts[1:])
                    self._column_source_map[col_name] = (table_name, field_name)
            else:
                # Se não há prefixo de tabela, será resolvido durante a construção da query
                self._column_source_map[col_name] = (None, col_name)
    
    @staticmethod
    def sanitize_identifier(name: str) -> str:
        """
        Sanitiza um identificador SQL para evitar problemas de sintaxe.
        
        Args:
            name: Nome do identificador.
            
        Returns:
            str: Identificador sanitizado.
        """
        # Remove caracteres inválidos para identificadores SQL
        sanitized = re.sub(r'[^\w\d_]', '_', name)
        
        # Garante que o identificador começa com letra ou sublinhado
        if sanitized and sanitized[0].isdigit():
            sanitized = f"_{sanitized}"
            
        return sanitized
    
    @staticmethod
    def normalize_view_column_name(name: str) -> str:
        """
        Normaliza o nome de uma coluna da view.
        
        Args:
            name: Nome original da coluna.
            
        Returns:
            str: Nome normalizado.
        """
        # Se o nome já tem um prefixo de tabela, preserva a estrutura
        if "." in name:
            parts = name.split(".")
            if len(parts) >= 2:
                table = ViewQueryBuilder.sanitize_identifier(parts[0])
                column = ViewQueryBuilder.sanitize_identifier(".".join(parts[1:]))
                return normalize_identifiers(f"{table}.{column}").sql()
        
        # Caso contrário, sanitiza o nome inteiro
        sanitized = ViewQueryBuilder.sanitize_identifier(name)
        return normalize_identifiers(sanitized).sql()
    
    @staticmethod
    def normalize_view_column_alias(name: str) -> str:
        """
        Normaliza o alias de uma coluna da view.
        
        Args:
            name: Alias original da coluna.
            
        Returns:
            str: Alias normalizado.
        """
        # Substitui pontos por sublinhados para criar um alias válido
        sanitized = name.replace(".", "_")
        sanitized = ViewQueryBuilder.sanitize_identifier(sanitized)
        return normalize_identifiers(sanitized).sql()
    
    def _get_group_by_columns(self) -> List[str]:
        """
        Obtém as colunas para a cláusula GROUP BY com aliasing adequado.
        
        Returns:
            List[str]: Lista de expressões de colunas para GROUP BY.
        """
        group_by_cols = []
        if hasattr(self.schema, 'group_by') and self.schema.group_by:
            for col in self.schema.group_by:
                group_by_cols.append(self.normalize_view_column_alias(col))
        return group_by_cols
    
    def _get_order_by_columns(self) -> List[Dict[str, str]]:
        """
        Obtém as colunas para a cláusula ORDER BY com aliasing adequado.
        
        Returns:
            List[Dict[str, str]]: Lista de cláusulas de ordenação.
        """
        order_by_cols = []
        if hasattr(self.schema, 'order_by') and self.schema.order_by:
            for order_clause in self.schema.order_by:
                # Suporta tanto strings simples quanto objetos de ordenação
                if isinstance(order_clause, dict):
                    col_name = order_clause.get('column')
                    direction = order_clause.get('direction', 'ASC').upper()
                    order_by_cols.append({
                        'column': self.normalize_view_column_alias(col_name),
                        'direction': direction
                    })
                elif isinstance(order_clause, str):
                    # Verifica se há especificação de direção
                    parts = order_clause.split()
                    col_name = parts[0]
                    direction = parts[1].upper() if len(parts) > 1 else 'ASC'
                    order_by_cols.append({
                        'column': self.normalize_view_column_alias(col_name),
                        'direction': direction
                    })
        return order_by_cols
    
    def _get_aliases(self) -> List[str]:
        """
        Obtém os aliases para todas as colunas.
        
        Returns:
            List[str]: Lista de aliases normalizados.
        """
        aliases = []
        for col in self.schema.columns:
            if hasattr(col, 'alias') and col.alias:
                aliases.append(col.alias)
            else:
                aliases.append(self.normalize_view_column_alias(col.name))
        return aliases
    
    def _get_columns(self) -> List[str]:
        """
        Obtém a lista de colunas para a query SELECT.
        
        Returns:
            List[str]: Lista de expressões de colunas.
        """
        columns = []
        aliases = self._get_aliases()
        
        for i, col in enumerate(self.schema.columns):
            # Define a expressão da coluna
            if hasattr(col, 'expression') and col.expression:
                # Pré-processa a expressão SQL personalizada
                expr = col.expression
                # Substitui hífens e pontos entre identificadores
                expr = re.sub(r"([a-zA-Z0-9_]+)-([a-zA-Z0-9_]+)", r"\1_\2", expr)
                expr = re.sub(r"([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)", r"\1_\2", expr)
                
                try:
                    # Tenta analisar a expressão SQL
                    column_expr = parse_one(expr).sql()
                except Exception as e:
                    self.logger.warning(f"Erro ao analisar expressão '{expr}': {str(e)}. Usando expressão como texto.")
                    column_expr = expr
            else:
                # Normaliza o nome da coluna para uso em SQL
                column_expr = self.normalize_view_column_alias(col.name)
            
            # Aplica transformações definidas para esta coluna
            if hasattr(self.schema, 'transformations') and self.schema.transformations:
                column_expr = self.transformation_manager.apply_column_transformations(
                    column_expr, col.name, self.schema.transformations
                )
            
            # Adiciona alias para a coluna
            alias = aliases[i]
            column_expr = f"{column_expr} AS {alias}"
            
            columns.append(column_expr)
        
        return columns
    
    def _get_referenced_tables(self) -> Set[str]:
        """
        Obtém o conjunto de tabelas referenciadas no esquema.
        
        Returns:
            Set[str]: Conjunto de nomes de tabelas.
        """
        tables = set()
        
        # Adiciona tabelas das relações
        if hasattr(self.schema, 'relations') and self.schema.relations:
            for relation in self.schema.relations:
                source_table = relation.source_table.split(".")[0] if "." in relation.source_table else relation.source_table
                target_table = relation.target_table.split(".")[0] if "." in relation.target_table else relation.target_table
                tables.add(source_table)
                tables.add(target_table)
        
        # Adiciona tabelas das colunas
        for col in self.schema.columns:
            col_name = col.name
            if "." in col_name:
                table_name = col_name.split(".")[0]
                tables.add(table_name)
        
        return tables
    
    def build_query(self) -> str:
        """
        Constrói a query SQL com aliasing adequado para colunas.
        
        Returns:
            str: Query SQL completa.
        """
        # Constrói a query a partir da expressão de tabela
        table_expr = self._get_table_expression()
        
        # Constrói a query usando os aliases de colunas
        query = select(*self._get_aliases()).from_(table_expr)
        
        # Adiciona DISTINCT se necessário
        if self._check_distinct():
            query = query.distinct()
        
        # Adiciona GROUP BY se houver
        if hasattr(self.schema, 'group_by') and self.schema.group_by:
            group_by_cols = self._get_group_by_columns()
            if group_by_cols:
                query = query.group_by(*group_by_cols)
        
        # Adiciona ORDER BY se houver
        order_by_clauses = self._get_order_by_columns()
        if order_by_clauses:
            for order_clause in order_by_clauses:
                column = order_clause['column']
                direction = order_clause['direction']
                if direction == 'DESC':
                    query = query.order_by(exp.Desc(exp.Column(column)))
                else:
                    query = query.order_by(column)
        
        # Adiciona LIMIT se houver
        if hasattr(self.schema, 'limit') and self.schema.limit:
            query = query.limit(self.schema.limit)
        
        # Formata a query final
        return query.transform(quote_identifiers).sql(pretty=True)
    
    def get_head_query(self, n: int = 5) -> str:
        """
        Obtém uma query que retorna apenas as primeiras linhas.
        
        Args:
            n: Número de linhas a retornar.
            
        Returns:
            str: Query SQL para as primeiras linhas.
        """
        # Constrói a query a partir da expressão de tabela
        table_expr = self._get_table_expression()
        
        # Constrói a query usando os aliases de colunas
        query = select(*self._get_aliases()).from_(table_expr)
        
        # Adiciona DISTINCT se necessário
        if self._check_distinct():
            query = query.distinct()
        
        # Adiciona GROUP BY se houver
        if hasattr(self.schema, 'group_by') and self.schema.group_by:
            group_by_cols = self._get_group_by_columns()
            if group_by_cols:
                query = query.group_by(*group_by_cols)
        
        # Adiciona ORDER BY se houver
        order_by_clauses = self._get_order_by_columns()
        if order_by_clauses:
            for order_clause in order_by_clauses:
                column = order_clause['column']
                direction = order_clause['direction']
                if direction == 'DESC':
                    query = query.order_by(exp.Desc(exp.Column(column)))
                else:
                    query = query.order_by(column)
        
        # Adiciona LIMIT
        query = query.limit(n)
        
        # Formata a query final
        return query.transform(quote_identifiers).sql(pretty=True)
    
    def get_row_count(self) -> str:
        """
        Obtém uma query que retorna o número total de linhas.
        
        Returns:
            str: Query SQL para contagem de linhas.
        """
        # Constrói a query a partir da expressão de tabela
        table_expr = self._get_table_expression()
        
        # Cria uma query de contagem que respeita configurações como DISTINCT
        query = select("COUNT(*) AS total_rows").from_(table_expr)
        
        # Adiciona DISTINCT se necessário
        if self._check_distinct():
            query = select("COUNT(DISTINCT *) AS total_rows").from_(table_expr)
        
        # Formata a query final
        return query.transform(quote_identifiers).sql(pretty=True)
    
    def _get_sub_query_from_loader(self, loader: Any) -> Subquery:
        """
        Obtém uma subquery a partir de um carregador de dados.
        
        Args:
            loader: Carregador de dados.
            
        Returns:
            Subquery: Objeto de subquery.
        """
        schema_name = None
        query_str = None
        
        # Extrai o nome do esquema
        if hasattr(loader, 'schema') and hasattr(loader.schema, 'name'):
            schema_name = loader.schema.name
        elif hasattr(loader, 'name'):
            schema_name = loader.name
        elif hasattr(loader, 'source_id'):
            schema_name = loader.source_id
        else:
            schema_name = "subquery"
        
        # Tenta obter a query do loader
        try:
            # Estratégia 1: Usar query_builder
            if hasattr(loader, 'query_builder') and loader.query_builder:
                if hasattr(loader.query_builder, 'build_query'):
                    query_str = loader.query_builder.build_query()
                elif hasattr(loader.query_builder, 'get_query'):
                    query_str = loader.query_builder.get_query()
                elif isinstance(loader.query_builder, str):
                    query_str = loader.query_builder
            
            # Estratégia 2: Usar métodos do loader
            if not query_str and hasattr(loader, 'build_query'):
                query_str = loader.build_query()
            elif not query_str and hasattr(loader, 'get_query'):
                query_str = loader.get_query()
            elif not query_str and hasattr(loader, 'query'):
                query_str = loader.query if callable(loader.query) else loader.query
            
            # Estratégia 3: Se loader for uma string, considera como SQL
            if not query_str and isinstance(loader, str):
                query_str = loader
            
            # Estratégia 4: Fallback para uma query simples
            if not query_str:
                if hasattr(loader, 'schema') and hasattr(loader.schema, 'name'):
                    table_name = loader.schema.name
                elif isinstance(loader, dict) and 'name' in loader:
                    table_name = loader['name']
                elif hasattr(loader, 'table_name'):
                    table_name = loader.table_name
                else:
                    table_name = schema_name
                
                query_str = f"SELECT * FROM {normalize_identifiers(table_name).sql()}"
            
            # Analisa a query obtida
            sub_query = parse_one(query_str)
            return exp.Subquery(this=sub_query, alias=normalize_identifiers(schema_name).sql())
            
        except Exception as e:
            self.logger.error(f"Erro ao obter query do loader para '{schema_name}': {str(e)}")
            # Cria uma subquery vazia como fallback
            sub_query = select("1").from_(exp.Table("dual"))
            return exp.Subquery(this=sub_query, alias=normalize_identifiers(schema_name).sql())
    
    def _resolve_column_source(self, column_name: str) -> Tuple[str, str]:
        """
        Resolve a tabela de origem de uma coluna.
        
        Args:
            column_name: Nome da coluna.
            
        Returns:
            Tuple[str, str]: Tupla (tabela, coluna).
        """
        # Verifica se a coluna já tem informação de origem
        if column_name in self._column_source_map:
            return self._column_source_map[column_name]
        
        # Tenta extrair a tabela de origem do nome da coluna
        if "." in column_name:
            parts = column_name.split(".")
            if len(parts) >= 2:
                table_name = parts[0]
                field_name = ".".join(parts[1:])
                return (table_name, field_name)
        
        # Se não conseguir determinar, retorna None para a tabela
        return (None, column_name)
    
    def _build_join_graph(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Constrói um grafo de junções entre tabelas.
        
        Returns:
            Dict[str, List[Dict[str, Any]]]: Grafo de junções.
        """
        join_graph = {}
        
        # Se não há relações, não há junções
        if not hasattr(self.schema, 'relations') or not self.schema.relations:
            return join_graph
        
        # Constrói o grafo de junções
        for relation in self.schema.relations:
            # Extrai tabelas da relação
            source_table = relation.source_table.split(".")[0] if "." in relation.source_table else relation.source_table
            target_table = relation.target_table.split(".")[0] if "." in relation.target_table else relation.target_table
            
            # Extrai colunas da relação
            source_column = relation.source_column
            target_column = relation.target_column
            
            # Adiciona entrada para a tabela de origem
            if source_table not in join_graph:
                join_graph[source_table] = []
            
            # Adiciona entrada para a tabela de destino
            if target_table not in join_graph:
                join_graph[target_table] = []
            
            # Adiciona a junção ao grafo
            join_graph[source_table].append({
                'target_table': target_table,
                'source_column': source_column,
                'target_column': target_column,
                'relationship_type': relation.relationship_type if hasattr(relation, 'relationship_type') else 'inner'
            })
            
            # Adiciona a junção inversa para facilitar navegação no grafo
            join_graph[target_table].append({
                'target_table': source_table,
                'source_column': target_column,
                'target_column': source_column,
                'relationship_type': relation.relationship_type if hasattr(relation, 'relationship_type') else 'inner'
            })
        
        return join_graph
    
    def _determine_join_order(self) -> List[Dict[str, Any]]:
        """
        Determina a ordem ótima de junção entre tabelas.
        
        Returns:
            List[Dict[str, Any]]: Lista de junções ordenadas.
        """
        # Constrói o grafo de junções
        join_graph = self._build_join_graph()
        
        # Se não há junções, retorna lista vazia
        if not join_graph:
            return []
        
        # Encontra as tabelas referenciadas
        referenced_tables = self._get_referenced_tables()
        
        # Determina a tabela inicial (preferência para tabelas com mais junções)
        start_table = None
        max_joins = -1
        for table in referenced_tables:
            if table in join_graph and len(join_graph[table]) > max_joins:
                start_table = table
                max_joins = len(join_graph[table])
        
        if not start_table and join_graph:
            # Se não encontrou nas tabelas referenciadas, usa a primeira do grafo
            start_table = next(iter(join_graph.keys()))
        
        if not start_table:
            # Se não há tabela inicial, não há junções
            return []
        
        # Algoritmo para determinar a ordem de junção
        visited = {start_table}
        join_order = []
        queue = [(start_table, None, None)]  # (tabela_atual, tabela_anterior, junção)
        
        while queue:
            current_table, prev_table, join_info = queue.pop(0)
            
            # Se há informação de junção, adiciona à ordem
            if prev_table is not None and join_info is not None:
                join_order.append({
                    'source_table': prev_table,
                    'target_table': current_table,
                    'source_column': join_info['source_column'],
                    'target_column': join_info['target_column'],
                    'relationship_type': join_info['relationship_type']
                })
            
            # Explora as junções da tabela atual
            if current_table in join_graph:
                for join in join_graph[current_table]:
                    target = join['target_table']
                    if target not in visited:
                        visited.add(target)
                        queue.append((target, current_table, join))
        
        return join_order
    
    def _get_table_expression(self) -> str:
        """
        Obtém a expressão de tabela para a cláusula FROM.
        
        Returns:
            str: Expressão de tabela.
            
        Raises:
            ValueError: Se não for possível construir a expressão.
        """
        # Determina as tabelas referenciadas
        referenced_tables = self._get_referenced_tables()
        
        # Verifica se há tabelas referenciadas
        if not referenced_tables:
            raise ValueError("Não há tabelas referenciadas no esquema")
        
        # Determina a tabela inicial
        join_order = self._determine_join_order()
        start_table = join_order[0]['source_table'] if join_order else next(iter(referenced_tables))
        
        # Verifica se a tabela inicial existe nas dependências
        if start_table not in self.schema_dependencies_dict:
            available = ", ".join(self.schema_dependencies_dict.keys())
            raise ValueError(f"Tabela inicial '{start_table}' não encontrada nas dependências. Disponíveis: {available}")
        
        # Obtém o loader da tabela inicial
        primary_loader = self.schema_dependencies_dict[start_table]
        primary_query = self._get_sub_query_from_loader(primary_loader)
        
        # Constrói a consulta base
        query = select("*").from_(primary_query)
        
        # Adiciona as junções
        for join_info in join_order:
            source_table = join_info['source_table']
            target_table = join_info['target_table']
            
            # Pula a primeira junção, pois já usamos a tabela inicial
            if source_table == start_table and target_table not in query.find_all(exp.Table):
                # Verifica se a tabela alvo existe nas dependências
                if target_table not in self.schema_dependencies_dict:
                    self.logger.warning(f"Tabela '{target_table}' não encontrada nas dependências")
                    continue
                
                # Obtém o loader da tabela alvo
                target_loader = self.schema_dependencies_dict[target_table]
                target_query = self._get_sub_query_from_loader(target_loader)
                
                # Determina o tipo de junção
                join_type = join_info['relationship_type']
                if isinstance(join_type, str):
                    join_type = join_type.lower()
                    if join_type == 'one_to_many' or join_type == 'many_to_one' or join_type == 'inner':
                        join_type = "INNER"
                    elif join_type == 'one_to_one':
                        join_type = "INNER"
                    elif join_type == 'many_to_many':
                        join_type = "INNER"
                    elif join_type in ('left', 'right', 'full'):
                        join_type = join_type.upper()
                    else:
                        join_type = "INNER"
                else:
                    join_type = "INNER"
                
                # Constrói a condição de junção
                source_column = self.normalize_view_column_name(join_info['source_column'])
                target_column = self.normalize_view_column_name(join_info['target_column'])
                
                # Se as colunas já têm prefixo de tabela, usa diretamente
                if "." not in source_column:
                    source_column = f"{normalize_identifiers(source_table).sql()}.{source_column}"
                if "." not in target_column:
                    target_column = f"{normalize_identifiers(target_table).sql()}.{target_column}"
                
                join_condition = f"{source_column} = {target_column}"
                
                # Adiciona a junção
                query = query.join(
                    target_query,
                    on=join_condition,
                    join_type=join_type,
                    append=True
                )
        
        # Seleciona as colunas específicas
        column_exprs = []
        for col in self.schema.columns:
            col_name = col.name
            table_name, field_name = self._resolve_column_source(col_name)
            
            # Se não determinou a tabela, usa a primeira
            if not table_name:
                table_name = start_table
            
            # Normaliza os nomes
            table_expr = normalize_identifiers(table_name).sql()
            field_expr = normalize_identifiers(field_name).sql()
            column_expr = f"{table_expr}.{field_expr}"
            
            # Define o alias para a coluna
            if hasattr(col, 'alias') and col.alias:
                alias = normalize_identifiers(col.alias).sql()
            else:
                alias = normalize_identifiers(self.normalize_view_column_alias(col_name)).sql()
            
            column_exprs.append(f"{column_expr} AS {alias}")
        
        # Constrói a subquery com as colunas selecionadas
        subquery = select(*column_exprs).from_(exp.Subquery(this=query, alias="source_query"))
        
        # Aplica transformações
        final_query = select(*self._get_columns()).from_(subquery)
        
        # Define o alias para a query completa
        view_name = normalize_identifiers(self.schema.name).sql()
        return exp.Subquery(this=final_query, alias=view_name).sql(pretty=True)
    
    def validate_query_builder(self) -> bool:
        """
        Valida se o construtor de queries pode gerar uma query SQL válida.
        
        Returns:
            bool: True se a validação passar, False caso contrário.
        """
        try:
            # Tenta construir a query
            query = self.build_query()
            
            # Verifica se a query é válida
            sqlglot.parse_one(query)
            
            # Verifica referências a tabelas
            referenced_tables = self._get_referenced_tables()
            for table in referenced_tables:
                if table not in self.schema_dependencies_dict:
                    self.logger.warning(f"Tabela '{table}' referenciada no esquema, mas não encontrada nas dependências")
                    return False
            
            return True
        except Exception as e:
            self.logger.error(f"Erro na validação do construtor de queries: {str(e)}")
            return False


class SQLSanitizer:
    """
    Utilitário para sanitização de queries SQL.
    
    Esta classe fornece métodos para prevenir injeção SQL e
    sanitizar nomes de tabelas e colunas.
    """
    
    @staticmethod
    def is_sql_query(text: str) -> bool:
        """
        Verifica se um texto contém uma query SQL maliciosa.
        
        Args:
            text: Texto a ser verificado.
            
        Returns:
            bool: True se contiver SQL malicioso, False caso contrário.
        """
        if not text:
            return False
            
        # Lista de padrões SQL comuns para injeção
        sql_patterns = [
            r";\s*SELECT", r";\s*INSERT", r";\s*UPDATE", r";\s*DELETE", 
            r";\s*DROP", r";\s*TRUNCATE", r";\s*ALTER", r";\s*CREATE",
            r"--", r"/\*.*\*/", r"UNION\s+(?:ALL\s+)?SELECT",
            r"SELECT\s+@@", r"EXEC\s+(?:xp|sp)_"
        ]
        
        text_lower = text.lower()
        
        # Verifica cada padrão
        for pattern in sql_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True
        
        return False
    
    @staticmethod
    def sanitize_identifier(name: str) -> str:
        """
        Sanitiza um identificador SQL.
        
        Args:
            name: Nome do identificador.
            
        Returns:
            str: Identificador sanitizado.
        """
        # Remove caracteres inválidos para identificadores SQL
        sanitized = re.sub(r'[^\w\d_]', '_', name)
        
        # Garante que o identificador começa com letra ou sublinhado
        if sanitized and sanitized[0].isdigit():
            sanitized = f"_{sanitized}"
            
        return sanitized
    
    @staticmethod
    def sanitize_view_column_name(name: str) -> str:
        """
        Sanitiza o nome de uma coluna para uso em views.
        
        Args:
            name: Nome original da coluna.
            
        Returns:
            str: Nome sanitizado.
        """
        # Se o nome já tem um prefixo de tabela, preserva a estrutura
        if "." in name:
            parts = name.split(".")
            if len(parts) >= 2:
                table = SQLSanitizer.sanitize_identifier(parts[0])
                column = SQLSanitizer.sanitize_identifier(".".join(parts[1:]))
                return f"{table}.{column}"
        
        # Caso contrário, sanitiza o nome inteiro
        return SQLSanitizer.sanitize_identifier(name)
    
    @staticmethod
    def sanitize_sql_params(params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitiza parâmetros para uso em queries SQL.
        
        Args:
            params: Dicionário de parâmetros.
            
        Returns:
            Dict[str, Any]: Parâmetros sanitizados.
        """
        sanitized = {}
        
        for key, value in params.items():
            # Sanitiza as chaves
            clean_key = SQLSanitizer.sanitize_identifier(key)
            
            # Sanitiza os valores string
            if isinstance(value, str):
                # Escapa aspas simples
                clean_value = value.replace("'", "''")
                sanitized[clean_key] = clean_value
            else:
                sanitized[clean_key] = value
        
        return sanitized
    
    @staticmethod
    def remove_comments(sql: str) -> str:
        """
        Remove comentários de uma query SQL.
        
        Args:
            sql: Query SQL.
            
        Returns:
            str: Query sem comentários.
        """
        # Remove comentários de linha (--) 
        sql = re.sub(r'--.*?(\n|$)', ' ', sql)
        
        # Remove comentários de bloco (/* ... */)
        sql = re.sub(r'/\*.*?\*/', ' ', sql, flags=re.DOTALL)
        
        return sql.strip()


class SQLDialectTranspiler:
    """
    Classe para transpilação de queries entre diferentes dialetos SQL.
    """
    DIALECT_MAPPINGS: Dict[str, Dict[str, Dict[str, str]]] = {
        'postgres': {
            'data_types': {
                'DATETIME': 'TIMESTAMP',
                'INT': 'INTEGER',
                'FLOAT': 'DOUBLE PRECISION',
                'BOOL': 'BOOLEAN',
                'CATEGORICAL': 'TEXT'
            },
            'functions': {
                'ROUND(': 'ROUND(',
                'COALESCE(': 'COALESCE(',
                'UPPER(': 'UPPER(',
                'LOWER(': 'LOWER(',
                'SUBSTRING(': 'SUBSTRING(',
                'TO_DATE(': 'TO_DATE(',
                'CURRENT_DATE': 'CURRENT_DATE',
                'CONCAT(': 'CONCAT('
            },
            'window_functions': {
                'OVER ()': 'OVER (ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)',
                'ROWS BETWEEN CURRENT ROW AND CURRENT ROW': 'ROWS BETWEEN CURRENT ROW AND CURRENT ROW'
            },
            'date_functions': {
                'EXTRACT(YEAR FROM ': 'EXTRACT(YEAR FROM ',
                'EXTRACT(MONTH FROM ': 'EXTRACT(MONTH FROM ',
                'EXTRACT(DAY FROM ': 'EXTRACT(DAY FROM '
            },
            'misc': {
                'NULLIF(': 'NULLIF(',
                'DIV': '/',
                'POW': 'POWER'
            }
        },
        'sqlite': {
            'data_types': {
                'TIMESTAMP': 'DATETIME',
                'INTEGER': 'INT',
                'DOUBLE PRECISION': 'REAL',
                'BOOLEAN': 'INTEGER',
                'TEXT': 'TEXT'
            },
            'functions': {
                'ROUND(': 'ROUND(',
                'COALESCE(': 'COALESCE(',
                'UPPER(': 'UPPER(',
                'LOWER(': 'LOWER(',
                'SUBSTRING(': 'SUBSTR(',
                'CURRENT_DATE': 'DATE(\'now\')',
                'CONCAT(': '||'
            },
            'window_functions': {
                'OVER ()': '(ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)',
                'OVER (ROWS BETWEEN CURRENT ROW AND CURRENT ROW)': '(ROWS BETWEEN CURRENT ROW AND CURRENT ROW)'
            },
            'date_functions': {
                'EXTRACT(YEAR FROM ': 'strftime(\'%Y\', ',
                'EXTRACT(MONTH FROM ': 'strftime(\'%m\', ',
                'EXTRACT(DAY FROM ': 'strftime(\'%d\', '
            },
            'misc': {
                'NULLIF(': 'CASE WHEN ',
                'DIV': '/',
                'POW': 'POWER'
            }
        },
        'duckdb': {
            'data_types': {
                'DATETIME': 'TIMESTAMP',
                'INT': 'INTEGER',
                'FLOAT': 'DOUBLE',
                'BOOL': 'BOOLEAN',
                'CATEGORICAL': 'VARCHAR'
            },
            'functions': {
                'ROUND(': 'ROUND(',
                'COALESCE(': 'COALESCE(',
                'UPPER(': 'UPPER(',
                'LOWER(': 'LOWER(',
                'SUBSTRING(': 'SUBSTRING(',
                'TO_DATE(': 'DATE(',
                'CURRENT_DATE': 'CURRENT_DATE',
                'CONCAT(': 'CONCAT('
            },
            'window_functions': {
                'OVER ()': 'OVER (ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)',
                'ROWS BETWEEN CURRENT ROW AND CURRENT ROW': 'ROWS BETWEEN CURRENT ROW AND CURRENT ROW'
            },
            'date_functions': {
                'EXTRACT(YEAR FROM ': 'EXTRACT(YEAR FROM ',
                'EXTRACT(MONTH FROM ': 'EXTRACT(MONTH FROM ',
                'EXTRACT(DAY FROM ': 'EXTRACT(DAY FROM '
            },
            'misc': {
                'NULLIF(': 'NULLIF(',
                'DIV': '/',
                'POW': 'POWER'
            }
        }
    }

    @classmethod
    def transpile(cls, query: str, source_dialect: str, target_dialect: str) -> str:
        """
        Transpila uma query de um dialeto SQL para outro.
        
        Args:
            query (str): Query original
            source_dialect (str): Dialeto de origem
            target_dialect (str): Dialeto de destino
        
        Returns:
            str: Query transpilada
        """
        if source_dialect == target_dialect:
            return query

        # Verifica se os dialetos existem no mapeamento
        if source_dialect not in cls.DIALECT_MAPPINGS or target_dialect not in cls.DIALECT_MAPPINGS:
            raise ValueError(f"Dialetos não suportados: {source_dialect} -> {target_dialect}")

        transpiled_query = query

        # Transpilação de tipos de dados
        for source_type, target_type in cls.DIALECT_MAPPINGS[source_dialect]['data_types'].items():
            transpiled_query = transpiled_query.replace(source_type, 
                                    cls.DIALECT_MAPPINGS[target_dialect]['data_types'].get(source_type, target_type))

        # Transpilação de funções
        for source_func, target_func in cls.DIALECT_MAPPINGS[source_dialect]['functions'].items():
            transpiled_query = transpiled_query.replace(source_func, 
                                    cls.DIALECT_MAPPINGS[target_dialect]['functions'].get(source_func, target_func))

        # Transpilação de funções de janela
        for source_window, target_window in cls.DIALECT_MAPPINGS[source_dialect]['window_functions'].items():
            transpiled_query = transpiled_query.replace(source_window, 
                                    cls.DIALECT_MAPPINGS[target_dialect]['window_functions'].get(source_window, target_window))

        # Transpilação de funções de data
        for source_date_func, target_date_func in cls.DIALECT_MAPPINGS[source_dialect]['date_functions'].items():
            transpiled_query = transpiled_query.replace(source_date_func, 
                                    cls.DIALECT_MAPPINGS[target_dialect]['date_functions'].get(source_date_func, target_date_func))

        # Transpilação de diversos
        for source_misc, target_misc in cls.DIALECT_MAPPINGS[source_dialect]['misc'].items():
            transpiled_query = transpiled_query.replace(source_misc, 
                                    cls.DIALECT_MAPPINGS[target_dialect]['misc'].get(source_misc, target_misc))

        # Tratamentos especiais
        transpiled_query = cls._handle_special_cases(transpiled_query, source_dialect, target_dialect)

        return transpiled_query

    @classmethod
    def _handle_special_cases(cls, query: str, source_dialect: str, target_dialect: str) -> str:
        """
        Lida com casos especiais de transpilação que requerem lógica mais complexa.
        
        Args:
            query (str): Query original
            source_dialect (str): Dialeto de origem
            target_dialect (str): Dialeto de destino
        
        Returns:
            str: Query transpilada
        """
        # Tratamento especial para NULLIF
        if source_dialect == 'local' and target_dialect in ['sqlite', 'postgres', 'duckdb']:
            # Converte NULLIF para expressões específicas de cada dialeto
            nullif_pattern = r'NULLIF\((.*?), (.*?)\)'
            
            def replace_nullif(match):
                expr, zero = match.groups()
                if target_dialect == 'sqlite':
                    return f'CASE WHEN {zero} = 0 THEN NULL ELSE {expr} / {zero} END'
                elif target_dialect in ['postgres', 'duckdb']:
                    return f'NULLIF({expr}, {zero})'
                return match.group(0)
            
            query = re.sub(nullif_pattern, replace_nullif, query)

        # Tratamento para normalização
        if 'NORMALIZE(' in query and target_dialect in ['sqlite', 'postgres', 'duckdb']:
            # Substitui a função de normalização por cálculo explícito
            normalize_pattern = r'NORMALIZE\((.*?)\)'
            
            def replace_normalize(match):
                expr = match.group(1)
                if target_dialect == 'sqlite':
                    return f'((({expr}) - (SELECT MIN({expr}) FROM (SELECT * FROM READ_CSV(\'/home/paulo/Projetos/genai/modulo/dados/vendas.csv\')))) / NULLIF((SELECT MAX({expr}) FROM (SELECT * FROM READ_CSV(\'/home/paulo/Projetos/genai/modulo/dados/vendas.csv\'))) - (SELECT MIN({expr}) FROM (SELECT * FROM READ_CSV(\'/home/paulo/Projetos/genai/modulo/dados/vendas.csv\'))), 0))'
                elif target_dialect in ['postgres', 'duckdb']:
                    return f'((({expr}) - MIN({expr}) OVER ()) / NULLIF(MAX({expr}) OVER () - MIN({expr}) OVER (), 0))'
                return match.group(0)
            
            query = re.sub(normalize_pattern, replace_normalize, query)

        return query


# Usa SQLParser definido em query_builder_base.py


class QueryBuilderFacade:
    def transpile_query(self, query: str, source_dialect: str, target_dialect: str) -> str:
        try:
            return SQLParser.transpile_sql_dialect(query, source_dialect, target_dialect)
        except Exception as e:
            print(f"Erro ao transpilar query: {e}")
            return query  # Retorna a query original em caso de erro
