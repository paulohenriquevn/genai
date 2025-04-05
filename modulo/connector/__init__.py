"""
Módulo de conectores para diferentes fontes de dados.
Este módulo fornece interfaces e implementações para acessar diversos tipos de dados.
"""

# Importações relativas explícitas para evitar ambiguidade
from .csv_connector import CsvConnector
from .data_connector_factory import DataConnectorFactory
from .data_connector import DataConnector
from .datasource_config import DataSourceConfig
from .duckdb_csv_connector import DuckDBCsvConnector
from .exceptions import ConfigurationException
from .metadata import ColumnMetadata, DatasetMetadata
from .postgres_connector import PostgresConnector

# Definição clara das classes públicas exportadas pelo módulo
__all__ = [
    "DataConnector",       # Classe abstrata base para conectores
    "DataSourceConfig",    # Configuração de fonte de dados
    "DataConnectorFactory", # Factory para criar conectores
    "CsvConnector",        # Conector para arquivos CSV
    "PostgresConnector",   # Conector para PostgreSQL
    "DuckDBCsvConnector",  # Conector para DuckDB com CSV
    "ConfigurationException", # Exceção para problemas de configuração
    "ColumnMetadata",      # Metadados de colunas
    "DatasetMetadata"      # Metadados de datasets
]