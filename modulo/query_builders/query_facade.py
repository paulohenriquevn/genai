import os
import logging
from typing import Dict, List, Any, Tuple

from connector.semantic_layer_schema import (
    SemanticSchema, 
    ColumnSchema, 
    TransformationRule,
    TransformationType
)
from query_builders.query_builder_base import BaseQueryBuilder
from query_builders.query_builders_implementation import (
    LocalQueryBuilder,
    SqlQueryBuilder,
    ViewQueryBuilder
)


logger = logging.getLogger(__name__)

class QueryBuilderFacade:
    """
    Fachada para os construtores de queries.
    
    Esta classe fornece uma interface simples para criar, executar
    e gerenciar queries SQL a partir de esquemas semânticos.
    """
    
    def __init__(self, base_path: str = None):
        """
        Inicializa a fachada do construtor de queries.
        
        Args:
            base_path: Caminho base para datasets locais.
        """
        self.base_path = base_path or os.getcwd()
        self.builders = {}
        self.schemas = {}
        self.loaders = {}
    
    def register_schema(self, schema: SemanticSchema) -> None:
        """
        Registra um esquema semântico.
        
        Args:
            schema: Esquema semântico a ser registrado.
        """
        self.schemas[schema.name] = schema
        logger.info(f"Esquema '{schema.name}' registrado com sucesso")
    
    def register_loader(self, name: str, loader: Any) -> None:
        """
        Registra um carregador de dados.
        
        Args:
            name: Nome do carregador.
            loader: Objeto carregador de dados.
        """
        self.loaders[name] = loader
        logger.info(f"Carregador '{name}' registrado com sucesso")
    
    def create_builder(self, schema_name: str, builder_type: str = None) -> BaseQueryBuilder:
        """
        Cria um construtor de queries para um esquema.
        
        Args:
            schema_name: Nome do esquema.
            builder_type: Tipo de construtor ('local', 'sql', 'view').
            
        Returns:
            BaseQueryBuilder: Construtor de queries.
            
        Raises:
            ValueError: Se o esquema não estiver registrado ou o tipo não for válido.
        """
        if schema_name not in self.schemas:
            raise ValueError(f"Esquema '{schema_name}' não registrado")
        
        schema = self.schemas[schema_name]
        
        # Determina automaticamente o tipo de construtor se não especificado
        if not builder_type:
            if hasattr(schema, 'relations') and schema.relations:
                builder_type = 'view'
            elif hasattr(schema, 'source_type') and schema.source_type in ('csv', 'parquet'):
                builder_type = 'local'
            else:
                builder_type = 'sql'
        
        # Cria o construtor apropriado
        builder = None
        if builder_type == 'local':
            builder = LocalQueryBuilder(schema, self.base_path)
        elif builder_type == 'sql':
            builder = SqlQueryBuilder(schema)
        elif builder_type == 'view':
            builder = ViewQueryBuilder(schema, self.loaders)
        else:
            raise ValueError(f"Tipo de construtor inválido: {builder_type}")
        
        # Armazena e retorna o construtor
        self.builders[schema_name] = builder
        logger.info(f"Construtor '{builder_type}' criado para esquema '{schema_name}'")
        return builder
    
    def get_builder(self, schema_name: str) -> BaseQueryBuilder:
        """
        Obtém um construtor existente ou cria um novo.
        
        Args:
            schema_name: Nome do esquema.
            
        Returns:
            BaseQueryBuilder: Construtor de queries.
        """
        if schema_name in self.builders:
            return self.builders[schema_name]
        return self.create_builder(schema_name)
    
    def build_query(self, schema_name: str, builder_type: str = None) -> str:
        """
        Constrói uma query SQL para um esquema.
        
        Args:
            schema_name: Nome do esquema.
            builder_type: Tipo de construtor.
            
        Returns:
            str: Query SQL completa.
        """
        builder = self.get_builder(schema_name) if schema_name in self.builders else self.create_builder(schema_name, builder_type)
        return builder.build_query()
    
    def build_head_query(self, schema_name: str, n: int = 5, builder_type: str = None) -> str:
        """
        Constrói uma query que retorna apenas as primeiras linhas.
        
        Args:
            schema_name: Nome do esquema.
            n: Número de linhas a retornar.
            builder_type: Tipo de construtor.
            
        Returns:
            str: Query SQL para as primeiras linhas.
        """
        builder = self.get_builder(schema_name) if schema_name in self.builders else self.create_builder(schema_name, builder_type)
        return builder.get_head_query(n)
    
    def build_count_query(self, schema_name: str, builder_type: str = None) -> str:
        """
        Constrói uma query que retorna o número total de linhas.
        
        Args:
            schema_name: Nome do esquema.
            builder_type: Tipo de construtor.
            
        Returns:
            str: Query SQL para contagem de linhas.
        """
        builder = self.get_builder(schema_name) if schema_name in self.builders else self.create_builder(schema_name, builder_type)
        return builder.get_row_count()
    
    
    def _extract_columns_info(self, schema_name: str) -> List[Dict[str, str]]:
        """
        Extrai informações sobre as colunas de um esquema.
        
        Args:
            schema_name: Nome do esquema.
            
        Returns:
            List[Dict[str, str]]: Lista de informações sobre as colunas.
        """
        if schema_name not in self.schemas:
            raise ValueError(f"Esquema '{schema_name}' não registrado")
        
        schema = self.schemas[schema_name]
        columns_info = []
        
        for col in schema.columns:
            # Determina o tipo da coluna
            col_type = "string"  # Tipo padrão
            
            if hasattr(col, 'type') and col.type:
                if col.type.value in ('int', 'integer'):
                    col_type = "integer"
                elif col.type.value in ('float', 'decimal', 'double'):
                    col_type = "float"
                elif col.type.value in ('bool', 'boolean'):
                    col_type = "boolean"
                elif col.type.value in ('date', 'datetime', 'timestamp'):
                    col_type = "datetime"
                elif col.type.value in ('uuid',):
                    col_type = "uuid"
            
            # Adiciona as informações da coluna
            columns_info.append({
                "name": col.name,
                "type": col_type,
                "description": col.description if hasattr(col, 'description') else None,
                "nullable": col.nullable if hasattr(col, 'nullable') else True
            })
        
        return columns_info
    
    def create_view_schema(
        self,
        view_name: str,
        description: str,
        source_schemas: List[str],
        columns: List[Dict[str, Any]],
        relations: List[Dict[str, str]] = None,
        transformations: List[Dict[str, Any]] = None
    ) -> SemanticSchema:
        """
        Cria um esquema semântico para uma view.
        
        Args:
            view_name: Nome da view.
            description: Descrição da view.
            source_schemas: Lista de nomes dos esquemas de origem.
            columns: Lista de definições de colunas.
            relations: Lista de definições de relações.
            transformations: Lista de definições de transformações.
            
        Returns:
            SemanticSchema: Esquema semântico da view.
        """
        # Verifica se todos os esquemas de origem estão registrados
        for schema_name in source_schemas:
            if schema_name not in self.schemas:
                raise ValueError(f"Esquema de origem '{schema_name}' não registrado")
        
        # Prepara colunas
        schema_columns = []
        for col_def in columns:
            # Cria um objeto de coluna para o esquema
            col = ColumnSchema(
                name=col_def['name'],
                type=col_def['type'],
                description=col_def.get('description', None),
                nullable=col_def.get('nullable', True),
                primary_key=col_def.get('primary_key', False),
                unique=col_def.get('unique', False)
            )
            schema_columns.append(col)
        
        # Prepara relações
        schema_relations = []
        if relations:
            for rel_def in relations:
                # Cria um objeto de relação para o esquema
                rel = {
                    'source_table': rel_def['source_table'],
                    'source_column': rel_def['source_column'],
                    'target_table': rel_def['target_table'],
                    'target_column': rel_def['target_column'],
                    'relationship_type': rel_def.get('relationship_type', 'one_to_many')
                }
                schema_relations.append(rel)
        
        # Prepara transformações
        schema_transformations = []
        if transformations:
            for trans_def in transformations:
                # Cria um objeto de transformação para o esquema
                trans = TransformationRule(
                    type=TransformationType(trans_def['type']),
                    column=trans_def['column'],
                    params=trans_def.get('params', {})
                )
                schema_transformations.append(trans)
        
        # Cria o esquema da view
        view_schema = SemanticSchema(
            name=view_name,
            description=description,
            columns=schema_columns,
            relations=schema_relations,
            transformations=schema_transformations
        )
        
        # Registra o esquema
        self.register_schema(view_schema)
        
        return view_schema
    
    def create_sql_view(
        self,
        view_name: str,
        base_query: str,
        dialect: str = "postgres"
    ) -> str:
        """
        Gera uma definição SQL para criar uma view.
        
        Args:
            view_name: Nome da view.
            base_query: Query SQL base para a view.
            dialect: Dialeto SQL a ser usado.
            
        Returns:
            str: Comando SQL para criar a view.
        """
        # Sanitiza o nome da view
        sanitized_name = view_name.replace(" ", "_").replace("-", "_")
        
        # Cria o comando para definir a view
        view_sql = f"CREATE OR REPLACE VIEW {sanitized_name} AS\n{base_query};"
        
        return view_sql
    
    def create_materialized_view(
        self,
        view_name: str,
        base_query: str,
        dialect: str = "postgres"
    ) -> str:
        """
        Gera uma definição SQL para criar uma view materializada.
        
        Args:
            view_name: Nome da view.
            base_query: Query SQL base para a view.
            dialect: Dialeto SQL a ser usado.
            
        Returns:
            str: Comando SQL para criar a view materializada.
        """
        # Sanitiza o nome da view
        sanitized_name = view_name.replace(" ", "_").replace("-", "_")
        
        # Cria o comando para definir a view materializada
        view_sql = f"CREATE MATERIALIZED VIEW {sanitized_name} AS\n{base_query};"
        
        return view_sql
    
    def generate_refresh_view_sql(
        self,
        view_name: str,
        dialect: str = "postgres"
    ) -> str:
        """
        Gera o SQL para atualizar uma view materializada.
        
        Args:
            view_name: Nome da view materializada.
            dialect: Dialeto SQL a ser usado.
            
        Returns:
            str: Comando SQL para atualizar a view materializada.
        """
        # Sanitiza o nome da view
        sanitized_name = view_name.replace(" ", "_").replace("-", "_")
        
        # Cria o comando para atualizar a view
        if dialect.lower() == "postgres":
            return f"REFRESH MATERIALIZED VIEW {sanitized_name};"
        elif dialect.lower() == "duckdb":
            return f"REFRESH VIEW {sanitized_name};"
        else:
            # Fallback genérico
            return f"-- Para o dialeto {dialect}, verifique a sintaxe específica:\n-- REFRESH [MATERIALIZED] VIEW {sanitized_name};"
    
    def add_transformation_to_schema(
        self,
        schema_name: str,
        transformation_type: str,
        column: str,
        params: Dict[str, Any] = None
    ) -> None:
        """
        Adiciona uma transformação a um esquema existente.
        
        Args:
            schema_name: Nome do esquema.
            transformation_type: Tipo de transformação.
            column: Nome da coluna a ser transformada.
            params: Parâmetros para a transformação.
        """
        if schema_name not in self.schemas:
            raise ValueError(f"Esquema '{schema_name}' não registrado")
        
        schema = self.schemas[schema_name]
        
        # Verifica se o tipo de transformação é válido
        try:
            trans_type = TransformationType(transformation_type)
        except ValueError:
            valid_types = [t.value for t in TransformationType]
            raise ValueError(f"Tipo de transformação inválido. Valores válidos: {', '.join(valid_types)}")
        
        # Cria e adiciona a nova transformação
        transformation = TransformationRule(
            type=trans_type,
            column=column,
            params=params or {}
        )
        
        # Inicializa a lista de transformações se necessário
        if not hasattr(schema, 'transformations') or not schema.transformations:
            schema.transformations = []
            
        schema.transformations.append(transformation)
        
        # Limpa o construtor para este esquema para forçar a recriação
        if schema_name in self.builders:
            del self.builders[schema_name]
        
        logger.info(f"Transformação '{transformation_type}' adicionada ao esquema '{schema_name}'")
    
    def transpile_query(
        self,
        query: str,
        target_dialect: str,
        from_dialect: str = "postgres"
    ) -> str:
        """
        Converte uma query SQL de um dialeto para outro.
        
        Args:
            query: Query SQL original.
            target_dialect: Dialeto de destino.
            from_dialect: Dialeto de origem.
            
        Returns:
            str: Query SQL convertida.
        """
        from query_builders.query_builder_base import SQLParser
        
        return SQLParser.transpile_sql_dialect(
            query, to_dialect=target_dialect, from_dialect=from_dialect
        )