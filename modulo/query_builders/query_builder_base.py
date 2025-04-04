"""
Módulo para construção avançada de queries SQL.

Este módulo fornece classes base para a construção dinâmica de
queries SQL baseadas em esquemas semânticos, suportando várias
fontes de dados e transformações.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod

import sqlglot
from sqlglot import parse_one, select
from sqlglot.optimizer.normalize_identifiers import normalize_identifiers
from sqlglot.optimizer.qualify_columns import quote_identifiers

from connector.semantic_layer_schema import (
    SemanticSchema, 
    ColumnSchema, 
    TransformationType, 
    TransformationRule
)

logger = logging.getLogger(__name__)

class QuerySQLTransformationManager:
    """
    Gerenciador de transformações SQL para expressões de query.
    
    Esta classe fornece métodos para aplicar várias transformações
    a expressões SQL, como preenchimento de valores nulos,
    mapeamento de valores, normalização, etc.
    """
    
    @staticmethod
    def _quote_str(value: str) -> str:
        """
        Coloca aspas e faz escape de um valor string para SQL.
        
        Args:
            value: Valor a ser processado.
            
        Returns:
            str: Valor com aspas e escape.
        """
        if value is None:
            return "NULL"
        # Substitui aspas simples por aspas simples duplicadas para escape em SQL
        escaped = str(value).replace("'", "''")
        return f"'{escaped}'"
    
    @staticmethod
    def _validate_numeric(
        value: Union[int, float], param_name: str
    ) -> Union[int, float]:
        """
        Valida que um valor é numérico.
        
        Args:
            value: Valor a ser validado.
            param_name: Nome do parâmetro para mensagem de erro.
            
        Returns:
            Union[int, float]: Valor validado.
            
        Raises:
            ValueError: Se o valor não for numérico.
        """
        if not isinstance(value, (int, float)):
            try:
                value = float(value)
            except (ValueError, TypeError):
                raise ValueError(
                    f"Parâmetro {param_name} deve ser numérico, recebeu {type(value)}"
                )
        return value
    
    @staticmethod
    def apply_transformations(expr: str, transformations: List[TransformationRule]) -> str:
        """
        Aplica uma lista de transformações a uma expressão.
        
        Args:
            expr: Expressão a ser transformada.
            transformations: Lista de transformações a aplicar.
            
        Returns:
            str: Expressão transformada.
        """
        if not transformations:
            return expr
        
        transformed_expr = expr
        for transformation in transformations:
            method_name = f"_{transformation.type.value}"
            if hasattr(QuerySQLTransformationManager, method_name):
                method = getattr(QuerySQLTransformationManager, method_name)
                transformed_expr = method(transformed_expr, transformation.params)
            else:
                raise ValueError(f"Tipo de transformação não suportado: {method_name}")
        
        return transformed_expr
    
    @staticmethod
    def _fill_na(expr: str, params: Dict) -> str:
        """
        Substitui valores nulos com um valor padrão.
        
        Args:
            expr: Expressão SQL.
            params: Parâmetros com 'value' contendo o valor para substituição.
            
        Returns:
            str: Expressão transformada.
        """
        value = params.get('value')
        if isinstance(value, str):
            value = QuerySQLTransformationManager._quote_str(value)
        else:
            value = QuerySQLTransformationManager._validate_numeric(
                value, "value"
            )
        return f"COALESCE({expr}, {value})"
    
    @staticmethod
    def _map_values(expr: str, params: Dict) -> str:
        """
        Mapeia valores com base em um dicionário.
        
        Args:
            expr: Expressão SQL.
            params: Parâmetros com 'mapping' contendo o dicionário de mapeamento.
            
        Returns:
            str: Expressão transformada.
        """
        mapping = params.get('mapping', {})
        if not mapping:
            return expr
        
        case_stmt = (
            "CASE "
            + " ".join(
                f"WHEN {expr} = {QuerySQLTransformationManager._quote_str(key)} THEN {QuerySQLTransformationManager._quote_str(value)}"
                for key, value in mapping.items()
            )
            + f" ELSE {expr} END"
        )
        
        return case_stmt
    
    @staticmethod
    def _normalize(expr: str, params: Dict) -> str:
        """
        Normaliza valores (min-max scaling).
        
        Args:
            expr: Expressão SQL.
            params: Parâmetros da normalização.
            
        Returns:
            str: Expressão transformada.
        """
        return f"(({expr} - MIN({expr}) OVER ()) / NULLIF((MAX({expr}) OVER () - MIN({expr}) OVER ()), 0))"
    
    @staticmethod
    def _standardize(expr: str, params: Dict) -> str:
        """
        Padroniza valores (z-score normalization).
        
        Args:
            expr: Expressão SQL.
            params: Parâmetros da padronização.
            
        Returns:
            str: Expressão transformada.
        """
        return f"(({expr} - AVG({expr}) OVER ()) / NULLIF(STDDEV({expr}) OVER (), 0))"
    
    @staticmethod
    def _round(expr: str, params: Dict) -> str:
        """
        Arredonda valores numéricos.
        
        Args:
            expr: Expressão SQL.
            params: Parâmetros com 'decimals' indicando o número de casas.
            
        Returns:
            str: Expressão transformada.
        """
        decimals = QuerySQLTransformationManager._validate_numeric(
            params.get('decimals', 0), "decimals"
        )
        return f"ROUND({expr}, {int(decimals)})"
    
    @staticmethod
    def _strip(expr: str, params: Dict) -> str:
        """
        Remove espaços no início e fim de strings.
        
        Args:
            expr: Expressão SQL.
            params: Parâmetros (não utilizados).
            
        Returns:
            str: Expressão transformada.
        """
        return f"TRIM({expr})"
    
    @staticmethod
    def _to_lowercase(expr: str, params: Dict) -> str:
        """
        Converte strings para minúsculas.
        
        Args:
            expr: Expressão SQL.
            params: Parâmetros (não utilizados).
            
        Returns:
            str: Expressão transformada.
        """
        return f"LOWER({expr})"
    
    @staticmethod
    def _to_uppercase(expr: str, params: Dict) -> str:
        """
        Converte strings para maiúsculas.
        
        Args:
            expr: Expressão SQL.
            params: Parâmetros (não utilizados).
            
        Returns:
            str: Expressão transformada.
        """
        return f"UPPER({expr})"
    
    @staticmethod
    def _replace(expr: str, params: Dict) -> str:
        """
        Substitui ocorrências de um valor por outro.
        
        Args:
            expr: Expressão SQL.
            params: Parâmetros com 'old_value' e 'new_value'.
            
        Returns:
            str: Expressão transformada.
        """
        old_value = params.get('old_value', '')
        new_value = params.get('new_value', '')
        return f"REPLACE({expr}, {QuerySQLTransformationManager._quote_str(old_value)}, {QuerySQLTransformationManager._quote_str(new_value)})"
    
    @staticmethod
    def _clip(expr: str, params: Dict) -> str:
        """
        Limita valores dentro de um intervalo [min, max].
        
        Args:
            expr: Expressão SQL.
            params: Parâmetros com 'lower' e 'upper'.
            
        Returns:
            str: Expressão transformada.
        """
        lower = QuerySQLTransformationManager._validate_numeric(params.get('lower', 0), "lower")
        upper = QuerySQLTransformationManager._validate_numeric(params.get('upper', 0), "upper")
        return f"LEAST(GREATEST({expr}, {lower}), {upper})"
    
    @staticmethod
    def _format_date(expr: str, params: Dict) -> str:
        """
        Formata uma data.
        
        Args:
            expr: Expressão SQL com data.
            params: Parâmetros com 'format' para formato da data.
            
        Returns:
            str: Expressão transformada.
        """
        date_format = params.get('format', '%Y-%m-%d')
        return f"DATE_FORMAT({expr}, {QuerySQLTransformationManager._quote_str(date_format)})"
    
    @staticmethod
    def _extract_date_component(expr: str, params: Dict) -> str:
        """
        Extrai componente de data (ano, mês, dia).
        
        Args:
            expr: Expressão SQL com data.
            params: Parâmetros com 'component'.
            
        Returns:
            str: Expressão transformada.
        """
        component = params.get('component', 'year').upper()
        return f"EXTRACT({component} FROM {expr})"
    
    @staticmethod
    def _remove_duplicates(expr: str, params: Dict) -> str:
        """
        Remove duplicatas.
        
        Args:
            expr: Expressão SQL.
            params: Parâmetros (não utilizados).
            
        Returns:
            str: Expressão transformada.
        """
        return f"DISTINCT {expr}"
    
    @staticmethod
    def _convert_type(expr, params):
        """
        Método para converter o tipo de uma expressão de coluna.
        
        Args:
            expr: Expressão da coluna a ser convertida
            transformation: Regra de transformação de tipo
        
        Returns:
            Expressão com tipo convertido
        """
        # Obtém o tipo de destino dos parâmetros da transformação
        target_type = params.get('type')
        
        # Mapeamento de tipos suportados
        type_mapping = {
            'int': 'INTEGER',
            'integer': 'INTEGER',
            'float': 'FLOAT',
            'double': 'DOUBLE',
            'string': 'VARCHAR',
            'bool': 'BOOLEAN',
            'boolean': 'BOOLEAN',
            'date': 'DATE',
            'datetime': 'TIMESTAMP'
        }
        
        # Valida o tipo de destino
        if target_type not in type_mapping:
            raise ValueError(f"Tipo de conversão não suportado: {target_type}")
        
        # Converte para o tipo SQL correspondente
        sql_type = type_mapping[target_type]
        
        # Cria uma nova expressão com o tipo convertido
        try:
            converted_expr = expr.cast(sql_type)
            return converted_expr
        except Exception as e:
            # Registra ou trata erros de conversão
            logger.warning(f"Erro ao converter tipo para {sql_type}: {str(e)}")
            return expr  # Retorna a expressão original em caso de falha


    @staticmethod
    def get_column_transformations(
        column_name: str, schema_transformations: List[TransformationRule]
    ) -> List[TransformationRule]:
        """
        Obtém todas as transformações que se aplicam a uma coluna específica.
        
        Args:
            column_name: Nome da coluna.
            schema_transformations: Lista de todas as transformações do esquema.
            
        Returns:
            List[TransformationRule]: Lista de transformações para a coluna.
        """
        if not schema_transformations:
            return []
            
        column_transforms = []
        for transform in schema_transformations:
            # Verifica se a transformação se aplica a esta coluna
            if transform.column.lower() == column_name.lower():
                column_transforms.append(transform)
                
        return column_transforms
    
    @staticmethod
    def apply_column_transformations(
        expr: str, column_name: str, schema_transformations: List[TransformationRule]
    ) -> str:
        """
        Aplica todas as transformações para uma coluna específica a uma expressão.
        
        Args:
            expr: Expressão SQL a ser transformada.
            column_name: Nome da coluna.
            schema_transformations: Lista de todas as transformações no esquema.
            
        Returns:
            str: Expressão SQL transformada.
        """
        transformations = QuerySQLTransformationManager.get_column_transformations(
            column_name, schema_transformations
        )
        return QuerySQLTransformationManager.apply_transformations(expr, transformations)


class BaseQueryBuilder(ABC):
    """
    Classe base para construtores de queries SQL.
    
    Esta classe define a interface e a funcionalidade básica para construir
    queries SQL dinâmicas com base em esquemas semânticos.
    """
    
    def __init__(self, schema: SemanticSchema):
        """
        Inicializa o construtor de queries.
        
        Args:
            schema: Esquema semântico para a construção da query.
        """
        self.schema = schema
        self.transformation_manager = QuerySQLTransformationManager()
    
    def validate_query_builder(self) -> None:
        """
        Valida se o construtor de queries pode gerar uma query SQL válida.
        
        Raises:
            ValueError: Se a query não puder ser construída ou for inválida.
        """
        try:
            sqlglot.parse_one(self.build_query())
        except Exception as error:
            raise ValueError(
                f"Falha ao gerar uma query SQL válida do esquema fornecido: {error}"
            )
    
    def build_query(self) -> str:
        """
        Constrói uma query SQL com base no esquema semântico.
        
        Returns:
            str: Query SQL completa.
        """
        # Constrói a query SELECT base
        query = select(*self._get_columns()).from_(self._get_table_expression())
        
        # Adiciona GROUP BY se houver agregações
        if hasattr(self.schema, 'group_by') and self.schema.group_by:
            query = query.group_by(
                *[normalize_identifiers(col) for col in self.schema.group_by]
            )
        
        # Adiciona DISTINCT se necessário
        if self._check_distinct():
            query = query.distinct()
        
        # Adiciona ORDER BY
        if hasattr(self.schema, 'order_by') and self.schema.order_by:
            query = query.order_by(*self.schema.order_by)
        
        # Adiciona LIMIT
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
        # Constrói a query SELECT base
        query = select(*self._get_columns()).from_(self._get_table_expression())
        
        # Adiciona DISTINCT se necessário
        if self._check_distinct():
            query = query.distinct()
        
        # Adiciona GROUP BY se houver agregações
        if hasattr(self.schema, 'group_by') and self.schema.group_by:
            query = query.group_by(
                *[normalize_identifiers(col) for col in self.schema.group_by]
            )
        
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
        return select("COUNT(*)").from_(self._get_table_expression()).sql(pretty=True)
    
    @abstractmethod
    def _get_columns(self) -> List[str]:
        """
        Obtém a lista de colunas para a query SELECT.
        
        Returns:
            List[str]: Lista de expressões de colunas.
        """
        pass
    
    @abstractmethod
    def _get_table_expression(self) -> str:
        """
        Obtém a expressão de tabela para a cláusula FROM.
        
        Returns:
            str: Expressão de tabela.
        """
        pass
    
    def _check_distinct(self) -> bool:
        """
        Verifica se a query deve ser DISTINCT.
        
        Returns:
            bool: True se deve ser DISTINCT, False caso contrário.
        """
        if not hasattr(self.schema, 'transformations') or not self.schema.transformations:
            return False
        
        # Verifica se existe uma transformação para remover duplicatas
        for transformation in self.schema.transformations:
            if transformation.type == TransformationType.DROP_NA:
                return True
        
        return False


class SQLParser:
    """
    Parser para manipulação e transformação de queries SQL.
    
    Esta classe fornece métodos para analisar, transformar e
    manipular queries SQL de várias maneiras.
    """
    
    @staticmethod
    def replace_table_and_column_names(query: str, table_mapping: Dict[str, str]) -> str:
        """
        Transforma uma query SQL substituindo nomes de tabelas.
        
        Args:
            query: Query SQL original.
            table_mapping: Dicionário mapeando nomes originais para novos nomes.
            
        Returns:
            str: Query SQL transformada.
        """
        # Pré-analisa todos os mapeamentos
        parsed_mapping = {}
        for key, value in table_mapping.items():
            try:
                parsed_mapping[key] = parse_one(value)
            except Exception:
                raise ValueError(f"{value} não é uma expressão SQL válida")
        
        def transform_node(node):
            # Manipula nós do tipo Table
            if isinstance(node, sqlglot.expressions.Table):
                original_name = node.name
                
                if original_name in table_mapping:
                    alias = node.alias or original_name
                    mapped_value = parsed_mapping[original_name]
                    
                    if isinstance(mapped_value, sqlglot.expressions.Alias):
                        return sqlglot.expressions.Subquery(
                            this=mapped_value.this.this,
                            alias=alias,
                        )
                    elif isinstance(mapped_value, sqlglot.expressions.Column):
                        return sqlglot.expressions.Table(this=mapped_value.this, alias=alias)
                    
                    return sqlglot.expressions.Subquery(this=mapped_value, alias=alias)
            
            return node
        
        # Analisa a query SQL
        parsed = parse_one(query)
        
        # Transforma a query
        transformed = parsed.transform(transform_node)
        transformed = transformed.transform(quote_identifiers)
        
        # Converte de volta para string SQL
        return transformed.sql(pretty=True)
    
    @staticmethod
    def transpile_sql_dialect(
        query: str, to_dialect: str, from_dialect: Optional[str] = None
    ) -> str:
        """
        Converte uma query SQL de um dialeto para outro.
        
        Args:
            query: Query SQL original.
            to_dialect: Dialeto de destino.
            from_dialect: Dialeto de origem (opcional).
            
        Returns:
            str: Query convertida.
        """
        placeholder = "___PLACEHOLDER___"
        query = query.replace("%s", placeholder)
        
        query = (
            parse_one(query, read=from_dialect) if from_dialect else parse_one(query)
        )
        result = query.sql(dialect=to_dialect, pretty=True)
        
        if to_dialect == "duckdb":
            return result.replace(placeholder, "?")
        
        return result.replace(placeholder, "%s")
    
    @staticmethod
    def extract_table_names(sql_query: str, dialect: str = "postgres") -> List[str]:
        """
        Extrai nomes de tabelas de uma query SQL.
        
        Args:
            sql_query: Query SQL.
            dialect: Dialeto SQL.
            
        Returns:
            List[str]: Lista de nomes de tabelas.
        """
        # Analisa a query SQL
        parsed = sqlglot.parse(sql_query, dialect=dialect)
        table_names = []
        cte_names = set()
        
        for stmt in parsed:
            # Identifica e armazena nomes de CTEs
            for cte in stmt.find_all(sqlglot.expressions.With):
                for cte_expr in cte.expressions:
                    cte_names.add(cte_expr.alias_or_name)
            
            # Extrai nomes de tabelas, excluindo CTEs
            for node in stmt.find_all(sqlglot.expressions.Table):
                if node.name not in cte_names:  # Ignora nomes de CTE
                    table_names.append(node.name)
        
        return table_names