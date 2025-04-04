import os
import logging
from typing import Dict, List, Any, Optional, Union

import pandas as pd
import duckdb

from connector.semantic_layer_schema import (
    ColumnSchema,
    RelationSchema,
    SemanticSchema, 
    ColumnType, 
    TransformationType, 
    TransformationRule
)

class ViewLoader:
    """
    Advanced view loader with semantic layer support and transformations.
    """
    
    def __init__(self, schema: SemanticSchema):
        """
        Initialize the ViewLoader with a semantic schema.
        
        Args:
            schema (SemanticSchema): Semantic schema defining the view.
        """
        self.schema = schema
        self.logger = logging.getLogger(f"ViewLoader[{schema.name}]")
        self.duckdb_conn = duckdb.connect(':memory:')
        self._registered_sources: Dict[str, pd.DataFrame] = {}
        
    def register_source(self, name: str, dataframe: pd.DataFrame) -> None:
        """
        Register a source DataFrame for view construction.
        
        Args:
            name (str): Name of the source.
            dataframe (pd.DataFrame): Source DataFrame.
        """
        self.duckdb_conn.register(name, dataframe)
        self._registered_sources[name] = dataframe
        self.logger.info(f"Registered source: {name}")
    
    def apply_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply transformations defined in the semantic schema.
        
        Args:
            df (pd.DataFrame): Input DataFrame.
        
        Returns:
            pd.DataFrame: Transformed DataFrame.
        """
        for transformation in self.schema.transformations:
            df = self._apply_single_transformation(df, transformation)
        return df
    
    def _apply_single_transformation(self, 
                                    df: pd.DataFrame, 
                                    transformation: TransformationRule) -> pd.DataFrame:
        """
        Apply a single transformation rule.
        
        Args:
            df (pd.DataFrame): Input DataFrame.
            transformation (TransformationRule): Transformation to apply.
        
        Returns:
            pd.DataFrame: Transformed DataFrame.
        """
        try:
            if transformation.type == TransformationType.RENAME:
                df = df.rename(columns={transformation.column: transformation.params.get('new_name')})
            
            elif transformation.type == TransformationType.FILLNA:
                df[transformation.column] = df[transformation.column].fillna(
                    transformation.params.get('value')
                )
            
            elif transformation.type == TransformationType.DROP_NA:
                df = df.dropna(subset=[transformation.column])
            
            elif transformation.type == TransformationType.CONVERT_TYPE:
                target_type = transformation.params.get('type')
                if target_type == 'int':
                    df[transformation.column] = pd.to_numeric(
                        df[transformation.column], errors='coerce'
                    ).astype('Int64')
                elif target_type == 'float':
                    df[transformation.column] = pd.to_numeric(
                        df[transformation.column], errors='coerce'
                    )
                elif target_type == 'datetime':
                    df[transformation.column] = pd.to_datetime(
                        df[transformation.column], 
                        errors='coerce', 
                        format=transformation.params.get('format')
                    )
            
            elif transformation.type == TransformationType.MAP_VALUES:
                df[transformation.column] = df[transformation.column].map(
                    transformation.params.get('mapping', {})
                )
            
            elif transformation.type == TransformationType.CLIP:
                df[transformation.column] = df[transformation.column].clip(
                    lower=transformation.params.get('min'),
                    upper=transformation.params.get('max')
                )
            
            elif transformation.type == TransformationType.REPLACE:
                df[transformation.column] = df[transformation.column].replace(
                    transformation.params.get('old_value'),
                    transformation.params.get('new_value')
                )
            
            else:
                self.logger.warning(f"Unsupported transformation: {transformation.type}")
            
            return df
        
        except Exception as e:
            self.logger.error(f"Error applying transformation {transformation.type}: {e}")
            return df
    
    def construct_view(self) -> pd.DataFrame:
        """
        Construct the view by joining registered sources and applying transformations.
        
        Returns:
            pd.DataFrame: Constructed view DataFrame.
        """
        # Validate sources are registered
        if not self._registered_sources:
            raise ValueError("No sources registered for view construction")
        
        # Construct the view query
        view_query = self._build_view_query()
        
        try:
            # Execute the view query in DuckDB
            result_df = self.duckdb_conn.execute(view_query).df()
            
            # Apply semantic schema transformations
            result_df = self.apply_transformations(result_df)
            
            self.logger.info(f"View {self.schema.name} constructed successfully")
            return result_df
        
        except Exception as e:
            self.logger.error(f"Error constructing view: {e}")
            raise
    
    def _build_view_query(self) -> str:
        """
        Constrói uma consulta SQL para construir a view com base nas relações.
        
        Returns:
            str: Consulta SQL para construção da view.
        """
        if not self.schema.relations:
            # Se não há relações, seleciona de todas as fontes
            source_tables = list(self._registered_sources.keys())
            if len(source_tables) == 1:
                return f"SELECT * FROM {source_tables[0]}"
            else:
                # Se múltiplas fontes, faz um produto cartesiano
                return f"SELECT * FROM {' CROSS JOIN '.join(source_tables)}"
        
        # Constrói consulta de junção baseada em relações
        join_query = None
        first_table = None
        
        for relation in self.schema.relations:
            if first_table is None:
                first_table = relation.source_table
                join_query = first_table
            
            # Adiciona junções
            join_query += f" INNER JOIN {relation.target_table} " \
                        f"ON {relation.source_table}.{relation.source_column} = " \
                        f"{relation.target_table}.{relation.target_column}"
        
        # Se não conseguiu construir a junção, faz junção cruzada
        if join_query is None:
            source_tables = list(self._registered_sources.keys())
            join_query = f"SELECT * FROM {' CROSS JOIN '.join(source_tables)}"
        else:
            join_query = f"SELECT * FROM {join_query}"
        
        return join_query
    
    def validate_view_sources(self) -> bool:
        """
        Validate that registered sources match the semantic schema.
        
        Returns:
            bool: True if sources are valid, False otherwise.
        """
        # Check if registered sources match schema
        registered_tables = set(self._registered_sources.keys())
        required_tables = {
            rel.source_table for rel in self.schema.relations
        } | {
            rel.target_table for rel in self.schema.relations
        }
        
        return registered_tables == required_tables
    
    def close(self) -> None:
        """Close the DuckDB connection."""
        self.duckdb_conn.close()
        self.logger.info("View loader connection closed")

def create_view_from_sources(
    schema: SemanticSchema, 
    sources: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Convenience function to create a view from semantic schema and sources.
    
    Args:
        schema (SemanticSchema): Semantic schema defining the view.
        sources (Dict[str, pd.DataFrame]): Dictionary of source DataFrames.
    
    Returns:
        pd.DataFrame: Constructed view DataFrame.
    
    Raises:
        ValueError: If sources do not match the semantic schema.
    """
    # Create view loader
    view_loader = ViewLoader(schema)
    
    # Register sources
    for name, dataframe in sources.items():
        view_loader.register_source(name, dataframe)
    
    # Validate sources
    if not view_loader.validate_view_sources():
        raise ValueError("Registered sources do not match the semantic schema")
    
    try:
        # Construct and return the view
        return view_loader.construct_view()
    finally:
        # Ensure connection is closed
        view_loader.close()

# Example usage and demonstration
def example_view_loading():
    """
    Demonstra o uso do carregamento de view semântica.
    """
    # Cria um esquema semântico para uma visão de vendas
    sales_view_schema = SemanticSchema(
        name='sales_analysis',
        description='Dados combinados de vendas e clientes',
        source_type='csv',
        columns=[
            ColumnSchema(name='sale_id', type=ColumnType.INTEGER, primary_key=True),
            ColumnSchema(name='customer_id', type=ColumnType.INTEGER),
            ColumnSchema(name='product_name', type=ColumnType.STRING),
            ColumnSchema(name='sale_amount', type=ColumnType.FLOAT),
            ColumnSchema(name='sale_date', type=ColumnType.DATE)
        ],
        relations=[
            RelationSchema(
                source_table='sales', 
                source_column='customer_id',
                target_table='customers', 
                target_column='customer_id'
            )
        ],
        transformations=[
            TransformationRule(
                type=TransformationType.FILLNA, 
                column='sale_amount', 
                params={'value': 0.0}
            ),
            TransformationRule(
                type=TransformationType.CONVERT_TYPE, 
                column='sale_date', 
                params={'type': 'datetime', 'format': '%Y-%m-%d'}
            )
        ]
    )

    # DataFrames de exemplo
    sales_df = pd.DataFrame({
        'sale_id': [1, 2, 3, 4],
        'customer_id': [101, 102, 103, 104],
        'product_name': ['Laptop', 'Phone', 'Tablet', 'Watch'],
        'sale_amount': [1200.0, 800.0, 500.0, None],
        'sale_date': ['2023-01-15', '2023-02-20', '2023-03-10', '2023-04-05']
    })

    customers_df = pd.DataFrame({
        'customer_id': [101, 102, 103, 104],
        'customer_name': ['Alice', 'Bob', 'Charlie', 'David'],
        'customer_city': ['New York', 'San Francisco', 'Chicago', 'Boston']
    })

    # Dicionário de fontes
    sources = {
        'sales': sales_df,
        'customers': customers_df
    }

    try:
        # Cria a view
        view_result = create_view_from_sources(sales_view_schema, sources)
        
        # Imprime o resultado da view
        print("Resultado da View de Vendas:")
        print(view_result)
        
    except Exception as e:
        print(f"Erro ao criar view: {e}")

# Optional: Run the example if this script is executed directly
if __name__ == '__main__':
    example_view_loading()