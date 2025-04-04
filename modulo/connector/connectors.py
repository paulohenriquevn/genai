import os
import glob
import json
import pandas as pd
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
from typing import Optional, Dict, List, Any

from connector.metadata import ColumnMetadata, DatasetMetadata, MetadataRegistry
from connector.semantic_layer_schema import SemanticSchema, TransformationRule, TransformationType
from connector.view_loader_and_transformer import create_view_from_sources

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("connector")


class DataConnectionException(Exception):
    """Exceção para problemas de conexão com fontes de dados."""
    pass


class DataReadException(Exception):
    """Exceção para problemas de leitura de dados."""
    pass


class ConfigurationException(Exception):
    """Exceção para problemas com configurações."""
    pass


class DataSourceConfig:
    """
    Configuração de fonte de dados com suporte a metadados.
    
    Estende a classe base DataSourceConfig para incluir informações
    de metadados sobre as colunas do dataset.
    
    Attributes:
        source_id (str): Identificador único da fonte de dados.
        source_type (str): Tipo da fonte de dados.
        params (Dict): Parâmetros específicos para o conector.
        metadata (Optional[DatasetMetadata]): Metadados do dataset.
    """
    
    def __init__(self, source_id: str, source_type: str, metadata: Optional[Union[Dict, DatasetMetadata]] = None, **params):
        """
        Inicializa a configuração com metadados.
        
        Args:
            source_id: Identificador único da fonte de dados.
            source_type: Tipo da fonte de dados.
            metadata: Metadados do dataset.
            **params: Parâmetros adicionais para o conector.
        """
        self.source_id = source_id
        self.source_type = source_type
        self.params = params
        
        # Processa os metadados
        if metadata is None:
            self.metadata = None
        elif isinstance(metadata, DatasetMetadata):
            self.metadata = metadata
        elif isinstance(metadata, dict):
            try:
                self.metadata = DatasetMetadata.from_dict(metadata)
            except Exception as e:
                logger.warning(f"Erro ao processar metadados: {str(e)}")
                self.metadata = None
        else:
            logger.warning(f"Formato de metadados não suportado: {type(metadata)}")
            self.metadata = None
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MetadataEnabledDataSourceConfig':
        """
        Cria uma instância a partir de um dicionário.
        
        Args:
            config_dict: Dicionário de configuração.
            
        Returns:
            MetadataEnabledDataSourceConfig: Nova instância.
        """
        source_id = config_dict.get('id')
        source_type = config_dict.get('type')
        metadata = config_dict.get('metadata')
        
        if not source_id:
            raise ConfigurationException("ID da fonte de dados não especificado")
        if not source_type:
            raise ConfigurationException("Tipo da fonte de dados não especificado")
            
        # Remove chaves especiais
        params = {k: v for k, v in config_dict.items() if k not in ('id', 'type', 'metadata')}
        
        return cls(source_id, source_type, metadata=metadata, **params)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'DataSourceConfig':
        """
        Cria uma instância de configuração a partir de uma string JSON.
        
        Args:
            json_str: String JSON com configurações.
            
        Returns:
            DataSourceConfig: Nova instância de configuração.
        """
        try:
            config_dict = json.loads(json_str)
            return cls.from_dict(config_dict)
        except json.JSONDecodeError as e:
            raise ConfigurationException(f"Erro ao decodificar JSON: {str(e)}")
        
    def resolve_column_name(self, name_or_alias: str) -> Optional[str]:
        """
        Resolve o nome real de uma coluna a partir de um nome ou alias.
        
        Args:
            name_or_alias: Nome ou alias da coluna.
            
        Returns:
            Optional[str]: Nome real da coluna ou None.
        """
        if self.metadata is None:
            return None
            
        return self.metadata.resolve_column_name(name_or_alias)
    
    def get_column_metadata(self, column_name: str) -> Optional[ColumnMetadata]:
        """
        Obtém metadados para uma coluna específica.
        
        Args:
            column_name: Nome da coluna.
            
        Returns:
            Optional[ColumnMetadata]: Metadados da coluna ou None.
        """
        if self.metadata is None:
            return None
            
        return self.metadata.get_column_metadata(column_name)
    
    def get_recommended_aggregations(self, column_name: str) -> List[str]:
        """
        Obtém as agregações recomendadas para uma coluna.
        
        Args:
            column_name: Nome da coluna.
            
        Returns:
            List[str]: Lista de agregações recomendadas.
        """
        if self.metadata is None:
            return []
            
        metadata = self.metadata.get_column_metadata(column_name)
        return metadata.aggregations if metadata else []
    
    def get_column_type(self, column_name: str) -> Optional[str]:
        """
        Obtém o tipo de dados de uma coluna.
        
        Args:
            column_name: Nome da coluna.
            
        Returns:
            Optional[str]: Tipo de dados da coluna ou None.
        """
        if self.metadata is None:
            return None
            
        metadata = self.metadata.get_column_metadata(column_name)
        return metadata.data_type if metadata else None
    
    def get_column_format(self, column_name: str) -> Optional[str]:
        """
        Obtém o formato de uma coluna.
        
        Args:
            column_name: Nome da coluna.
            
        Returns:
            Optional[str]: Formato da coluna ou None.
        """
        if self.metadata is None:
            return None
            
        metadata = self.metadata.get_column_metadata(column_name)
        return metadata.format if metadata else None


class DataConnector:
    """Base interface for all data connectors."""
    
    def connect(self) -> None:
        """
        Establish connection to the data source.
        
        Raises:
            DataConnectionException: If connection fails.
        """
        pass
    
    def read_data(self, query: Optional[str] = None) -> pd.DataFrame:
        """
        Read data from the source according to the specified query.
        
        Args:
            query: Query to filter/transform data.
            
        Returns:
            pd.DataFrame: DataFrame with the read data.
            
        Raises:
            DataReadException: If reading fails.
        """
        pass
    
    def close(self) -> None:
        """
        Close the connection to the data source.
        
        Raises:
            DataConnectionException: If closing the connection fails.
        """
        pass
    
    def is_connected(self) -> bool:
        """
        Check if the connection is active.
        
        Returns:
            bool: True if connected, False otherwise.
        """
        pass
        
    def apply_semantic_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply transformations defined in the semantic schema.
        
        Args:
            df: Input DataFrame.
            
        Returns:
            pd.DataFrame: Transformed DataFrame.
        """
        if not hasattr(self, 'config') or not self.config or not hasattr(self.config, 'semantic_schema') or not self.config.semantic_schema:
            return df
            
        result_df = df.copy()
        schema = self.config.semantic_schema
        
        for transformation in schema.transformations:
            result_df = self._apply_single_transformation(result_df, transformation)
            
        return result_df
    
    def _apply_single_transformation(self, df: pd.DataFrame, transformation: TransformationRule) -> pd.DataFrame:
        """
        Apply a single transformation rule.
        
        Args:
            df: Input DataFrame.
            transformation: Transformation to apply.
            
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
            
            elif transformation.type == TransformationType.NORMALIZE:
                # Min-max normalization
                min_val = df[transformation.column].min()
                max_val = df[transformation.column].max()
                if max_val > min_val:
                    df[transformation.column] = (df[transformation.column] - min_val) / (max_val - min_val)
            
            elif transformation.type == TransformationType.STANDARDIZE:
                # Z-score standardization
                mean_val = df[transformation.column].mean()
                std_val = df[transformation.column].std()
                if std_val > 0:
                    df[transformation.column] = (df[transformation.column] - mean_val) / std_val
            
            elif transformation.type == TransformationType.ENCODE_CATEGORICAL:
                # Simple one-hot encoding
                encoding_method = transformation.params.get('method', 'one_hot')
                if encoding_method == 'one_hot':
                    dummies = pd.get_dummies(df[transformation.column], prefix=transformation.column)
                    df = pd.concat([df, dummies], axis=1)
                
            elif transformation.type == TransformationType.EXTRACT_DATE:
                # Extract date components
                component = transformation.params.get('component', 'year')
                if component == 'year':
                    df[f"{transformation.column}_year"] = pd.to_datetime(df[transformation.column]).dt.year
                elif component == 'month':
                    df[f"{transformation.column}_month"] = pd.to_datetime(df[transformation.column]).dt.month
                elif component == 'day':
                    df[f"{transformation.column}_day"] = pd.to_datetime(df[transformation.column]).dt.day
                elif component == 'weekday':
                    df[f"{transformation.column}_weekday"] = pd.to_datetime(df[transformation.column]).dt.weekday
            
            elif transformation.type == TransformationType.ROUND:
                decimals = transformation.params.get('decimals', 0)
                df[transformation.column] = df[transformation.column].round(decimals)
            
            else:
                logger.warning(f"Unsupported transformation: {transformation.type}")
            
            return df
        
        except Exception as e:
            logger.error(f"Error applying transformation {transformation.type}: {e}")
            return df
            
    def create_view_from_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create a view based on the semantic schema from a DataFrame.
        
        Args:
            df: Source DataFrame.
            
        Returns:
            pd.DataFrame: View DataFrame.
        """
        if not hasattr(self, 'config') or not self.config or not hasattr(self.config, 'semantic_schema') or not self.config.semantic_schema:
            return df
            
        # Create a temporary view loader
        from view_loader_and_transformer import ViewLoader
        
        view_loader = ViewLoader(self.config.semantic_schema)
        
        # Register the DataFrame as a source
        view_loader.register_source(self.config.source_id, df)
        
        try:
            # Construct and return the view
            view_df = view_loader.construct_view()
            return view_df
        finally:
            # Clean up
            view_loader.close()


class CsvConnector(DataConnector):
    """
    CSV connector with semantic layer support.
    
    Extends the standard CSV connector to utilize metadata information
    for better data interpretation and transformation, and semantic schema
    for view construction.
    Supports reading a directory containing multiple CSV files.
    """
    
    def __init__(self, config: Union[DataSourceConfig]):
        """
        Initialize the connector.
        
        Args:
            config: Data source configuration.
        """
        self.config = config
        self.data = None
        self._connected = False
        self.is_directory = False
        self.csv_files = []
        self.dataframes = {}
        
        # Validate required parameters
        if 'path' not in self.config.params:
            raise ConfigurationException("Parameter 'path' is required for CSV sources")

    def connect(self) -> None:
        """
        Load the CSV file or directory of CSVs into memory.
        """
        try:
            path = self.config.params['path']
            delimiter = self.config.params.get('delimiter', ',')
            encoding = self.config.params.get('encoding', 'utf-8')
            
            # Check if the path is a directory
            if os.path.isdir(path):
                self.is_directory = True
                pattern = self.config.params.get('pattern', '*.csv')
                logger.info(f"Connecting to CSV directory: {path} with pattern {pattern}")
                
                # List all CSV files in the directory
                self.csv_files = glob.glob(os.path.join(path, pattern))
                
                if not self.csv_files:
                    logger.warning(f"No CSV files found in directory: {path}")
                    self._connected = False
                    return
                
                # Load each CSV file into a separate DataFrame
                for csv_file in self.csv_files:
                    try:
                        file_name = os.path.basename(csv_file)
                        logger.info(f"Loading CSV file: {file_name}")
                        
                        df = pd.read_csv(
                            csv_file,
                            delimiter=delimiter,
                            encoding=encoding
                        )
                        
                        # Apply metadata-based transformations for each DataFrame
                        df = self._apply_metadata_transformations(df)
                        
                        # Apply semantic schema transformations
                        df = self.apply_semantic_transformations(df)
                        
                        self.dataframes[file_name] = df
                        
                    except Exception as e:
                        logger.error(f"Error loading CSV file {file_name}: {str(e)}")
                
                # If at least one file was successfully loaded, consider connected
                if self.dataframes:
                    self._connected = True
                    
                    # Concatenate all DataFrames for simple queries (without joins)
                    # This is a simple approach that can be refined later
                    if self.config.params.get('auto_concat', True):
                        try:
                            self.data = pd.concat(self.dataframes.values(), ignore_index=True)
                            logger.info(f"DataFrames successfully concatenated. Total of {len(self.data)} rows.")
                            
                            # Apply view construction using semantic schema if available
                            if hasattr(self.config, 'semantic_schema') and self.config.semantic_schema:
                                self.data = self.create_view_from_dataframe(self.data)
                                
                        except Exception as e:
                            logger.warning(f"Could not concatenate DataFrames: {str(e)}")
                            # Use the first DataFrame as fallback
                            self.data = next(iter(self.dataframes.values()))
                else:
                    self._connected = False
                
            else:
                # Original behavior for a single file
                logger.info(f"Connecting to CSV: {path}")
                self.data = pd.read_csv(
                    path, 
                    delimiter=delimiter, 
                    encoding=encoding
                )
                self._connected = True
                logger.info(f"Successfully connected to CSV: {path}")
                
                # If connected successfully and has metadata, apply transformations
                if self._connected and self.data is not None:
                    self.data = self._apply_metadata_transformations(self.data)
                    
                    # Apply semantic schema transformations
                    self.data = self.apply_semantic_transformations(self.data)
                    
                    # Apply view construction using semantic schema if available
                    if hasattr(self.config, 'semantic_schema') and self.config.semantic_schema:
                        self.data = self.create_view_from_dataframe(self.data)
                
        except Exception as e:
            self._connected = False
            error_msg = f"Error connecting to CSV {self.config.params.get('path')}: {str(e)}"
            logger.error(error_msg)
            raise DataConnectionException(error_msg) from e

    def _apply_metadata_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply metadata-based transformations to the loaded data.
        
        Args:
            df: DataFrame to transform
            
        Returns:
            pd.DataFrame: Transformed DataFrame
        """
        if df is None or not hasattr(self.config, 'metadata') or not self.config.metadata:
            return df
        
        result_df = df.copy()
        metadata = self.config.metadata
        
        # Apply type conversions for each column with metadata
        for column_name, column_metadata in metadata.columns.items():
            if column_name in result_df.columns and column_metadata.data_type:
                try:
                    # Conversion based on the type defined in metadata
                    result_df = self._convert_column_type(result_df, column_name, column_metadata)
                except Exception as e:
                    logger.warning(f"Error converting column {column_name}: {str(e)}")
        
        return result_df

    def _convert_column_type(self, df: pd.DataFrame, column_name: str, metadata: ColumnMetadata) -> pd.DataFrame:
        """
        Convert a column to the type specified in the metadata.
        
        Args:
            df: DataFrame containing the column
            column_name: Name of the column to convert
            metadata: Column metadata
            
        Returns:
            pd.DataFrame: DataFrame with the converted column
        """
        result_df = df.copy()
        data_type = metadata.data_type
        format_str = metadata.format
        
        try:
            # Conversion according to type
            if data_type == 'int':
                result_df[column_name] = pd.to_numeric(result_df[column_name], errors='coerce').astype('Int64')
                logger.info(f"Column {column_name} converted to integer")
                
            elif data_type == 'float':
                result_df[column_name] = pd.to_numeric(result_df[column_name], errors='coerce')
                logger.info(f"Column {column_name} converted to float")
                
            elif data_type == 'date':
                result_df[column_name] = pd.to_datetime(result_df[column_name], format=format_str, errors='coerce')
                logger.info(f"Column {column_name} converted to date")
                
            elif data_type == 'bool':
                # Handle boolean values represented as strings
                true_values = ['true', 'yes', 'y', '1', 'sim', 's']
                false_values = ['false', 'no', 'n', '0', 'não', 'nao']
                
                def to_bool(x):
                    if isinstance(x, str):
                        x = x.lower()
                        if x in true_values:
                            return True
                        if x in false_values:
                            return False
                    return x
                
                result_df[column_name] = result_df[column_name].apply(to_bool)
                logger.info(f"Column {column_name} converted to boolean")
                
        except Exception as e:
            logger.warning(f"Error converting column {column_name} to {data_type}: {str(e)}")
        
        return result_df

    def read_data(self, query: Optional[str] = None) -> pd.DataFrame:
        """
        Read data from the CSV or directory of CSVs, optionally applying an SQL query.
        
        Args:
            query: Optional SQL query to filter or transform the data.
            
        Returns:
            pd.DataFrame: DataFrame with the resulting data.
        """
        if not self._connected:
            raise DataConnectionException("Not connected to data source. Call connect() first.")
            
        try:
            # Simplest case: without query return all data (already concatenated)
            if not query:
                if self.is_directory and self.config.params.get('return_dict', False):
                    # Return a dictionary of DataFrames for advanced processing
                    return self.dataframes
                return self.data.copy() if self.data is not None else pd.DataFrame()
            
            # Adapt the query with metadata if necessary
            if hasattr(self.config, 'metadata') and self.config.metadata:
                query = self._adapt_query_with_metadata(query)
            
            # If it's a directory, we need special logic for queries
            if self.is_directory and self.dataframes:
                return self._execute_query_on_directory(query)
            
            # Default behavior for a single DataFrame
            import sqlite3
            
            # Create an in-memory SQLite connection
            conn = sqlite3.connect(':memory:')
            
            # Register the DataFrame as a temporary table
            table_name = f"csv_data_{self.config.source_id}"
            if self.data is not None:
                self.data.to_sql(table_name, conn, if_exists='replace', index=False)
            
                # Replace table references in the query
                modified_query = query.replace("FROM csv", f"FROM {table_name}")
                
                # Execute the query
                result = pd.read_sql_query(modified_query, conn)
                conn.close()
                
                return result
            else:
                return pd.DataFrame()
            
        except Exception as e:
            error_msg = f"Error reading data from CSV: {str(e)}"
            logger.error(error_msg)
            raise DataReadException(error_msg) from e

    def _execute_query_on_directory(self, query: str) -> pd.DataFrame:
        """
        Execute an SQL query on a directory of CSV files.
        
        Args:
            query: SQL query to execute.
            
        Returns:
            pd.DataFrame: Query result.
        """
        import sqlite3
        
        # Create an in-memory SQLite connection
        conn = sqlite3.connect(':memory:')
        
        # Register each DataFrame as a separate temporary table
        for file_name, df in self.dataframes.items():
            # Remove extension and special characters to create valid table names
            table_name = os.path.splitext(file_name)[0]
            table_name = ''.join(c if c.isalnum() else '_' for c in table_name)
            df.to_sql(table_name, conn, if_exists='replace', index=False)
            logger.info(f"Registered file {file_name} as table {table_name}")
        
        # Also register the concatenated DataFrame for simple queries
        if self.data is not None:
            combined_table = f"csv_data_{self.config.source_id}"
            self.data.to_sql(combined_table, conn, if_exists='replace', index=False)
            
            # Replace generic table references in the query
            modified_query = query.replace("FROM csv", f"FROM {combined_table}")
        else:
            modified_query = query
        
        try:
            # Execute the query
            logger.info(f"Executing query: {modified_query}")
            result = pd.read_sql_query(modified_query, conn)
            return result
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            # Try to infer table names in the query
            error_msg = f"Error executing query. Make sure to use the correct table names: {', '.join(self.dataframes.keys())}"
            raise DataReadException(error_msg) from e
        finally:
            conn.close()

    def _adapt_query_with_metadata(self, query: str) -> str:
        """
        Adapt an SQL query using metadata information.
        
        Args:
            query: Original SQL query.
            
        Returns:
            str: Adapted query.
        """
        if not hasattr(self.config, 'metadata') or not self.config.metadata:
            return query
        
        metadata = self.config.metadata
        adapted_query = query
        
        # Replace aliases with real column names
        for alias, real_name in getattr(metadata, '_alias_lookup', {}).items():
            # Use regex for precise replacement
            import re
            pattern = r'(?<![a-zA-Z0-9_])' + re.escape(alias) + r'(?![a-zA-Z0-9_])'
            adapted_query = re.sub(pattern, real_name, adapted_query)
        
        logger.info(f"Query adapted with metadata: {adapted_query}")
        return adapted_query

    def close(self) -> None:
        """
        Free resources. For CSV, just clear the data reference.
        """
        self.data = None
        self.dataframes = {}
        self.csv_files = []
        self._connected = False
        logger.info(f"CSV connection closed: {self.config.params.get('path')}")

    def is_connected(self) -> bool:
        """
        Check if the connector is active.
        
        Returns:
            bool: True if connected, False otherwise.
        """
        if self.is_directory:
            return self._connected and bool(self.dataframes)
        else:
            return self._connected and self.data is not None

    def get_available_tables(self) -> List[str]:
        """
        Return a list of available tables (file names) when in directory mode.
        
        Returns:
            List[str]: List of available file names/tables
        """
        if not self.is_directory:
            return []
        
        return list(self.dataframes.keys())


class PostgresConnector(DataConnector):
    """
    Conector para bancos de dados PostgreSQL.
    
    Attributes:
        config (DataSourceConfig): Configuração da fonte de dados.
        connection: Conexão com o banco de dados.
    """
    
    def __init__(self, config: DataSourceConfig):
        """
        Inicializa um novo conector PostgreSQL.
        
        Args:
            config: Configuração da fonte de dados.
        """
        self.config = config
        self.connection = None
        
        # Validação de parâmetros obrigatórios
        required_params = ['host', 'database', 'username', 'password']
        missing_params = [param for param in required_params if param not in self.config.params]
        
        if missing_params:
            raise ConfigurationException(
                f"Parâmetros obrigatórios ausentes para PostgreSQL: {', '.join(missing_params)}"
            )
    
    def connect(self) -> None:
        """
        Estabelece conexão com o banco PostgreSQL.
        """
        try:
            import psycopg2
            
            host = self.config.params['host']
            database = self.config.params['database']
            username = self.config.params['username']
            password = self.config.params['password']
            port = self.config.params.get('port', 5432)
            
            logger.info(f"Conectando ao PostgreSQL: {host}/{database}")
            
            self.connection = psycopg2.connect(
                host=host,
                port=port,
                database=database,
                user=username,
                password=password
            )
            
            logger.info(f"Conectado com sucesso ao PostgreSQL: {host}/{database}")
            
        except ImportError:
            error_msg = "Módulo psycopg2 não encontrado. Instale com: pip install psycopg2-binary"
            logger.error(error_msg)
            raise DataConnectionException(error_msg)
        except Exception as e:
            error_msg = f"Erro ao conectar com PostgreSQL: {str(e)}"
            logger.error(error_msg)
            raise DataConnectionException(error_msg) from e
    
    def read_data(self, query: str) -> pd.DataFrame:
        """
        Executa uma consulta SQL no banco PostgreSQL.
        
        Args:
            query: Consulta SQL a ser executada.
            
        Returns:
            pd.DataFrame: DataFrame com os resultados da consulta.
        """
        if not self.is_connected():
            raise DataConnectionException("Não conectado ao banco de dados. Chame connect() primeiro.")
            
        if not query:
            raise DataReadException("Query SQL é obrigatória para conectores PostgreSQL")
            
        try:
            return pd.read_sql_query(query, self.connection)
        except Exception as e:
            error_msg = f"Erro ao executar query no PostgreSQL: {str(e)}"
            logger.error(error_msg)
            raise DataReadException(error_msg) from e
    
    def close(self) -> None:
        """
        Fecha a conexão com o banco de dados.
        """
        if self.connection:
            self.connection.close()
            self.connection = None
            logger.info(f"Conexão PostgreSQL fechada: {self.config.params.get('host')}/{self.config.params.get('database')}")
    
    def is_connected(self) -> bool:
        """
        Verifica se a conexão está ativa.
        
        Returns:
            bool: True se conectado, False caso contrário.
        """
        if not self.connection:
            return False
            
        try:
            # Verifica se a conexão está ativa com uma consulta simples
            cursor = self.connection.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
            return True
        except Exception:
            return False


class DuckDBCsvConnector(DataConnector):
    """
    DuckDB connector with semantic layer support.
    
    This connector uses DuckDB to efficiently process SQL queries on CSV files,
    with additional support for column metadata and semantic layer schema.
    Supports reading a directory containing multiple CSV files.
    
    Attributes:
        config: Connector configuration.
        connection: DuckDB connection.
        table_name: Table name in DuckDB.
        column_mapping: Mapping between aliases and real column names.
        is_directory: Flag indicating if the path is a directory.
        csv_files: List of CSV files in the directory.
        tables: Dictionary of registered table names.
        view_loader: Optional ViewLoader for semantic layer integration.
    """
    
    def __init__(self, config: Union[DataSourceConfig]):
        """
        Initialize the connector.
        
        Args:
            config: Connector configuration.
        """
        self.config = config
        self.connection = None
        self.table_name = f"csv_data_{self.config.source_id}"
        self.column_mapping = {}
        self.is_directory = False
        self.csv_files = []
        self.tables = {}
        self.view_loader = None
        
        # Validate required parameters
        if 'path' not in self.config.params:
            raise ConfigurationException("Parameter 'path' is required for CSV sources")
    
    def connect(self) -> None:
        """
        Establish connection with DuckDB and register the CSV file or directory as tables.
        """
        try:
            import duckdb
            
            # Initialize DuckDB connection
            self.connection = duckdb.connect(database=':memory:')
            
            path = self.config.params['path']
            
            # Check if the path is a directory
            if os.path.isdir(path):
                self.is_directory = True
                pattern = self.config.params.get('pattern', '*.csv')
                logger.info(f"Connecting to CSV directory via DuckDB: {path} with pattern {pattern}")
                
                # List all CSV files in the directory
                self.csv_files = glob.glob(os.path.join(path, pattern))
                
                if not self.csv_files:
                    logger.warning(f"No CSV files found in directory: {path}")
                    return
                
                # Determine parameters for reading CSVs
                delim = self.config.params.get('delim', 
                        self.config.params.get('sep', 
                        self.config.params.get('delimiter', ',')))
                
                has_header = self.config.params.get('header', True)
                auto_detect = self.config.params.get('auto_detect', True)
                
                # Register each CSV file as a view/table in DuckDB
                for csv_file in self.csv_files:
                    try:
                        file_name = os.path.basename(csv_file)
                        # Remove extension and special characters to create valid table names
                        table_name = os.path.splitext(file_name)[0]
                        table_name = ''.join(c if c.isalnum() else '_' for c in table_name)
                        
                        # Build query to create the view
                        query_parts = [f"CREATE VIEW {table_name} AS SELECT * FROM read_csv('{csv_file}'"]
                        params = []
                        
                        params.append(f"delim='{delim}'")
                        params.append(f"header={str(has_header).lower()}")
                        params.append(f"auto_detect={str(auto_detect).lower()}")
                        
                        if params:
                            query_parts.append(", " + ", ".join(params))
                        
                        query_parts.append(")")
                        create_query = "".join(query_parts)
                        
                        logger.info(f"Registering file {file_name} as table {table_name}")
                        logger.debug(f"Query: {create_query}")
                        
                        self.connection.execute(create_query)
                        self.tables[file_name] = table_name
                        
                    except Exception as e:
                        logger.error(f"Error registering CSV file {file_name}: {str(e)}")
                
                # Create a combined view if requested
                if self.config.params.get('create_combined_view', True) and self.tables:
                    try:
                        # Select the first file to get the schema
                        first_table = next(iter(self.tables.values()))
                        schema_query = f"SELECT * FROM {first_table} LIMIT 0"
                        schema_df = self.connection.execute(schema_query).fetchdf()
                        
                        # Create a UNION ALL query for all tables
                        union_parts = []
                        for table_name in self.tables.values():
                            # Check if the table has the same columns
                            try:
                                columns_query = f"SELECT * FROM {table_name} LIMIT 0"
                                table_columns = self.connection.execute(columns_query).fetchdf().columns
                                
                                # Add only tables with compatible structure
                                if set(schema_df.columns) == set(table_columns):
                                    union_parts.append(f"SELECT * FROM {table_name}")
                                else:
                                    logger.warning(f"Table {table_name} ignored in combined view due to schema differences")
                            except:
                                logger.warning(f"Error checking schema for table {table_name}")
                        
                        if union_parts:
                            # Create the combined view
                            combined_query = f"CREATE VIEW {self.table_name} AS {' UNION ALL '.join(union_parts)}"
                            self.connection.execute(combined_query)
                            logger.info(f"Combined view created: {self.table_name}")
                        
                    except Exception as e:
                        logger.warning(f"Could not create combined view: {str(e)}")
                
            else:
                # Original behavior for a single file
                if not os.path.exists(path):
                    # Try to find the file in the current directory
                    current_dir = os.getcwd()
                    base_filename = os.path.basename(path)
                    alternative_path = os.path.join(current_dir, base_filename)
                    
                    if os.path.exists(alternative_path):
                        logger.info(f"File not found at {path}, using alternative: {alternative_path}")
                        path = alternative_path
                    else:
                        logger.warning(f"CSV file not found: {path}")
                        return
                
                logger.info(f"Connecting to CSV via DuckDB: {path}")
                
                # Determine parameters
                delim = self.config.params.get('delim', 
                        self.config.params.get('sep', 
                        self.config.params.get('delimiter', ',')))
                
                has_header = self.config.params.get('header', True)
                auto_detect = self.config.params.get('auto_detect', True)
                
                # Build query to create the view
                query_parts = [f"CREATE VIEW {self.table_name} AS SELECT * FROM read_csv('{path}'"]
                params = []
                
                params.append(f"delim='{delim}'")
                params.append(f"header={str(has_header).lower()}")
                params.append(f"auto_detect={str(auto_detect).lower()}")
                
                if params:
                    query_parts.append(", " + ", ".join(params))
                
                query_parts.append(")")
                create_query = "".join(query_parts)
                
                logger.info(f"Query for DuckDB view creation: {create_query}")
                self.connection.execute(create_query)
                
                # Register the table name
                self.tables[os.path.basename(path)] = self.table_name
            
            # Get columns for mapping
            self._create_column_mapping()
            
            # Check structure of registered tables
            self._log_tables_schema()
            
            # Initialize semantic layer if available
            if hasattr(self.config, 'semantic_schema') and self.config.semantic_schema:
                self._initialize_semantic_layer()
        except Exception as e:
            error_msg = f"Error connecting to DuckDB: {str(e)}"
            logger.error(error_msg)
            raise DataConnectionException(error_msg) from e
                
    def _initialize_semantic_layer(self) -> None:
        """
        Initialize the semantic layer integration with ViewLoader.
        """
        try:
            from view_loader_and_transformer import ViewLoader
            
            # Create view loader with semantic schema
            self.view_loader = ViewLoader(self.config.semantic_schema)
            logger.info(f"Semantic layer initialized for {self.config.source_id}")
            
            # Prepare data for view loader
            # Get sample data for each table to register with the view loader
            for file_name, table_name in self.tables.items():
                try:
                    # Get DataFrame from the table
                    df = self.connection.execute(f"SELECT * FROM {table_name}").fetchdf()
                    # Register as a source for the view loader
                    self.view_loader.register_source(table_name, df)
                    logger.info(f"Registered table {table_name} with ViewLoader")
                except Exception as e:
                    logger.warning(f"Error registering table {table_name} with ViewLoader: {str(e)}")
                    
        except ImportError:
            logger.warning("Could not import ViewLoader. Semantic layer integration disabled.")
            self.view_loader = None
        except Exception as e:
            logger.warning(f"Error initializing semantic layer: {str(e)}")
            self.view_loader = None
            
    def _create_column_mapping(self) -> None:
        """
        Create a mapping between aliases and real column names.
        """
        self.column_mapping = {}
        
        # If no registered tables, nothing to map
        if not self.tables:
            return
        
        # Use the first table to get the columns
        try:
            first_table = next(iter(self.tables.values()))
            query = f"SELECT * FROM {first_table} LIMIT 0"
            columns_df = self.connection.execute(query).fetchdf()
            columns = columns_df.columns
            
            # If we have column metadata, use the defined aliases
            if hasattr(self.config, 'metadata') and self.config.metadata:
                for col_name, metadata in self.config.metadata.columns.items():
                    if col_name in columns:
                        for alias in metadata.alias:
                            self.column_mapping[alias.lower()] = col_name
                
                logger.info(f"Column mapping created from metadata: {self.column_mapping}")
            
            # If we have semantic schema, add column mappings from there as well
            if hasattr(self.config, 'semantic_schema') and self.config.semantic_schema:
                schema = self.config.semantic_schema
                for column in schema.columns:
                    if column.name in columns:
                        # Use description as an alias if available
                        if column.description:
                            self.column_mapping[column.description.lower()] = column.name
                
                logger.info(f"Column mapping enhanced with semantic schema")
            
            else:
                # Otherwise, use heuristic approach
                lower_cols = [col.lower() for col in columns]
                
                # Map generic names to real columns
                generic_mappings = {
                    'date': ['date', 'data', 'dt', 'dia', 'mes', 'ano', 'data_venda', 'data_compra'],
                    'revenue': ['revenue', 'receita', 'valor', 'venda', 'montante', 'faturamento'],
                    'profit': ['profit', 'lucro', 'margem', 'ganho', 'resultado'],
                    'quantity': ['quantity', 'quantidade', 'qtde', 'qtd', 'volume', 'unidades'],
                    'id': ['id', 'codigo', 'code', 'identificador', 'chave'],
                    'product': ['product', 'produto', 'item', 'mercadoria'],
                    'customer': ['customer', 'cliente', 'comprador', 'consumidor']
                }
                
                # Create the mapping
                for generic, options in generic_mappings.items():
                    for option in options:
                        for i, col_lower in enumerate(lower_cols):
                            if option in col_lower:
                                self.column_mapping[generic] = columns[i]
                                break
                        if generic in self.column_mapping:
                            break
                
                logger.info(f"Column mapping created by heuristic: {self.column_mapping}")
        except Exception as e:
            logger.warning(f"Could not create column mapping: {str(e)}")
            
    def _log_tables_schema(self) -> None:
        """
        Log information about table schemas for debugging.
        """
        for file_name, table_name in self.tables.items():
            try:
                schema_info = self.connection.execute(f"DESCRIBE {table_name}").fetchdf()
                logger.info(f"Schema for table {table_name} ({file_name}):")
                for _, row in schema_info.iterrows():
                    logger.info(f"  {row['column_name']} - {row['column_type']}")
            except Exception as e:
                logger.warning(f"Could not get schema for table {table_name}: {str(e)}")
                
    def read_data(self, query: Optional[str] = None) -> pd.DataFrame:
        """
        Read data from the CSV or directory of CSVs, optionally applying an SQL query.
        
        Args:
            query: Optional SQL query.
            
        Returns:
            pd.DataFrame: DataFrame with results.
        """
        if not self.is_connected():
            raise DataConnectionException("Not connected to data source. Call connect() first.")
            
        try:
            # Use semantic layer view if available and no specific query is provided
            if not query and self.view_loader is not None:
                try:
                    # Construct and return view using the semantic schema
                    view_df = self.view_loader.construct_view()
                    logger.info(f"View constructed using semantic schema for {self.config.source_id}")
                    return view_df
                except Exception as view_error:
                    logger.warning(f"Error constructing view: {str(view_error)}. Falling back to regular query.")
            
            # If no specific query, select all data from the main table
            if not query:
                if self.is_directory and self.config.params.get('return_dict', False):
                    # Return a dictionary of DataFrames for each file
                    result = {}
                    for file_name, table_name in self.tables.items():
                        try:
                            df = self.connection.execute(f"SELECT * FROM {table_name}").fetchdf()
                            
                            # Apply semantic transformations if available
                            if hasattr(self, 'apply_semantic_transformations'):
                                df = self.apply_semantic_transformations(df)
                                
                            result[file_name] = df
                        except Exception as e:
                            logger.warning(f"Error reading table {table_name}: {str(e)}")
                    return result
                
                # Use the combined table or the only available table
                table_to_query = self.table_name if self.table_name in self._get_all_tables() else next(iter(self.tables.values()), None)
                
                if table_to_query:
                    query = f"SELECT * FROM {table_to_query}"
                else:
                    return pd.DataFrame()
            else:
                # Adapt the query using metadata and table substitutions
                query = self._adapt_query(query)
            
            logger.info(f"Executing query: {query}")
            
            # Execute the query
            try:
                result_df = self.connection.execute(query).fetchdf()
                
                # Apply semantic transformations if available
                if hasattr(self, 'apply_semantic_transformations'):
                    result_df = self.apply_semantic_transformations(result_df)
                    
                return result_df
            except Exception as query_error:
                logger.warning(f"Error in query: {str(query_error)}. Showing available tables.")
                
                # List available tables to help the user
                available_tables = self._get_all_tables()
                error_msg = (f"Error executing query: {str(query_error)}. "
                            f"Available tables: {', '.join(available_tables)}")
                raise DataReadException(error_msg) from query_error
            
        except Exception as e:
            if isinstance(e, DataReadException):
                raise e
            
            error_msg = f"Error reading data from CSV via DuckDB: {str(e)}"
            logger.error(error_msg)
            
            # Try to provide an empty DataFrame instead of failing
            try:
                return pd.DataFrame()
            except:
                raise DataReadException(error_msg) from e
                
    def _get_all_tables(self) -> List[str]:
        """
        Return all tables and views available in DuckDB.
        
        Returns:
            List[str]: List of table/view names
        """
        try:
            tables_df = self.connection.execute("SHOW TABLES").fetchdf()
            if 'name' in tables_df.columns:
                return tables_df['name'].tolist()
            return []
        except Exception as e:
            logger.warning(f"Error listing tables: {str(e)}")
            return list(self.tables.values())
            
    def _adapt_query(self, query: str) -> str:
        """
        Adapt an SQL query using metadata and semantic schema information.
        
        Args:
            query: Original SQL query.
            
        Returns:
            str: Adapted query.
        """
        adapted_query = query
        
        # Adapt with metadata if available
        if hasattr(self.config, 'metadata') and self.config.metadata:
            adapted_query = self._adapt_query_with_metadata(adapted_query)
            
        # Adapt with semantic schema if available
        if hasattr(self.config, 'semantic_schema') and self.config.semantic_schema:
            adapted_query = self._adapt_query_with_semantic_schema(adapted_query)
            
        # Generic table name substitution
        if "FROM csv" in adapted_query and self.table_name in self._get_all_tables():
            adapted_query = adapted_query.replace("FROM csv", f"FROM {self.table_name}")
            
        return adapted_query
            
    def _adapt_query_with_metadata(self, query: str) -> str:
        """
        Adapt an SQL query using metadata information.
        
        Args:
            query: Original SQL query.
            
        Returns:
            str: Adapted query.
        """
        if not hasattr(self.config, 'metadata') or not self.config.metadata:
            return query
        
        metadata = self.config.metadata
        adapted_query = query
        
        # Replace aliases with real column names
        for alias, real_name in getattr(metadata, '_alias_lookup', {}).items():
            # Use regex for precise replacement
            import re
            pattern = r'(?<![a-zA-Z0-9_])' + re.escape(alias) + r'(?![a-zA-Z0-9_])'
            adapted_query = re.sub(pattern, real_name, adapted_query)
        
        logger.info(f"Query adapted with metadata: {adapted_query}")
        return adapted_query
        
    def _adapt_query_with_semantic_schema(self, query: str) -> str:
        """
        Adapt an SQL query using semantic schema information.
        
        Args:
            query: Original SQL query.
            
        Returns:
            str: Adapted query with semantic adaptations.
        """
        if not hasattr(self.config, 'semantic_schema') or not self.config.semantic_schema:
            return query
            
        schema = self.config.semantic_schema
        adapted_query = query
        
        # No direct query adaptation needed for semantic schema,
        # as we use the ViewLoader for query execution with semantic schema
        
        return adapted_query
    
    def close(self) -> None:
        """
        Close the DuckDB connection.
        """
        # Close the view loader if it exists
        if self.view_loader:
            try:
                self.view_loader.close()
                logger.info("ViewLoader connection closed")
            except Exception as view_error:
                logger.warning(f"Error closing ViewLoader: {str(view_error)}")
                
        if self.connection:
            try:
                # Try to remove the view/table before closing
                try:
                    self.connection.execute(f"DROP VIEW IF EXISTS {self.table_name}")
                except Exception as drop_error:
                    logger.warning(f"Could not remove view: {str(drop_error)}")
                
                # Close the connection
                self.connection.close()
            except Exception as e:
                logger.warning(f"Error closing DuckDB connection: {str(e)}")
            finally:
                self.connection = None
                self.view_loader = None
                logger.info(f"DuckDB connection closed for CSV: {self.config.params.get('path')}")
    
    def is_connected(self) -> bool:
        """
        Check if the connector is active.
        
        Returns:
            bool: True if connected, False otherwise.
        """
        if not self.connection:
            return False
            
        try:
            # Check if the connection is active
            self.connection.execute("SELECT 1")
            return True
        except Exception:
            return False
    
    def get_schema(self) -> pd.DataFrame:
        """
        Return the schema (structure) of the CSV file.
        
        Returns:
            pd.DataFrame: DataFrame with schema information.
        """
        if not self.is_connected():
            raise DataConnectionException("Not connected to data source. Call connect() first.")
            
        try:
            # Get information about column schema
            query = f"DESCRIBE {self.table_name}"
            return self.connection.execute(query).fetchdf()
        except Exception as e:
            logger.warning(f"Error getting schema: {str(e)}")
            
            # Alternative: create schema based on a simple query
            try:
                query = f"SELECT * FROM {self.table_name} LIMIT 1"
                sample = self.connection.execute(query).fetchdf()
                
                schema_data = {
                    'column_name': sample.columns,
                    'column_type': [str(sample[col].dtype) for col in sample.columns]
                }
                return pd.DataFrame(schema_data)
            except Exception as alt_error:
                error_msg = f"Error getting alternative schema: {str(alt_error)}"
                logger.error(error_msg)
                raise DataReadException(error_msg) from e
    
    def sample_data(self, num_rows: int = 5) -> pd.DataFrame:
        """
        Return a sample of the data.
        
        Args:
            num_rows: Number of rows to return.
            
        Returns:
            pd.DataFrame: DataFrame with the sample.
        """
        if not self.is_connected():
            raise DataConnectionException("Not connected to data source. Call connect() first.")
            
        try:
            # If we have a semantic view, use that
            if self.view_loader:
                try:
                    view_df = self.view_loader.construct_view()
                    return view_df.head(num_rows)
                except Exception as view_error:
                    logger.warning(f"Error sampling from semantic view: {str(view_error)}. Using raw table.")
            
            # Otherwise, use the raw table
            query = f"SELECT * FROM {self.table_name} LIMIT {num_rows}"
            return self.connection.execute(query).fetchdf()
        except Exception as e:
            error_msg = f"Error getting data sample: {str(e)}"
            logger.error(error_msg)
            raise DataReadException(error_msg) from e
            
        except ImportError:
            error_msg = "duckdb module not found. Install with: pip install duckdb"
            logger.error(error_msg)
            raise DataConnectionException(error_msg)
        except Exception as e:
            error_msg = f"Error connecting to CSV via DuckDB: {str(e)}"
            logger.error(error_msg)
            raise DataConnectionException(error_msg) from e

class DataConnectorFactory:
    """
    Factory for connectors with metadata and semantic layer support.
    
    Extends the standard factory to create connectors that recognize and
    utilize both column metadata and semantic schema.
    """
    _connectors = {
        'csv': 'CsvConnector',  # Using string to avoid circular imports
        'postgres': 'PostgresConnector',
        'duckdb_csv': 'DuckDBCsvConnector'
    }
    
    @classmethod
    def register_connector(cls, source_type: str, connector_class) -> None:
        """
        Register a new connector type in the factory.
        
        Args:
            source_type: Data source type name.
            connector_class: Connector class to register.
        """
        cls._connectors[source_type] = connector_class
        logger.info(f"Connector registered for type: {source_type}")
    
    @classmethod
    def create_connector(cls, config) -> Any:
        """
        Create a connector with metadata and semantic layer support.
        
        Args:
            config: Data source configuration.
            
        Returns:
            DataConnector: Created connector.
        """
        # Import connectors here to avoid circular imports
        from connector.connectors import CsvConnector, PostgresConnector, DuckDBCsvConnector, DataSourceConfig
        
        # Convert config to DataSourceConfig if necessary
        if not isinstance(config, DataSourceConfig):
            config = DataSourceConfig.from_dict(config)
        
        # Create the appropriate connector based on type
        source_type = config.source_type
        
        if source_type not in cls._connectors:
            raise ValueError(f"Unsupported connector type: {source_type}")
            
        connector_class_name = cls._connectors[source_type]
        
        # If connector_class_name is a string, get the actual class
        if isinstance(connector_class_name, str):
            if connector_class_name == 'CsvConnector':
                connector_class = CsvConnector
            elif connector_class_name == 'PostgresConnector':
                connector_class = PostgresConnector
            elif connector_class_name == 'DuckDBCsvConnector':
                connector_class = DuckDBCsvConnector
            else:
                raise ValueError(f"Unknown connector class name: {connector_class_name}")
        else:
            connector_class = connector_class_name
        
        # Create the connector with the enhanced configuration
        return connector_class(config)
    
    @classmethod
    def create_from_json(cls, json_config: str) -> Dict[str, Any]:
        """
        Create multiple connectors from a JSON configuration.
        
        Args:
            json_config: JSON string with configurations.
            
        Returns:
            Dict[str, DataConnector]: Dictionary with connectors.
        """
        try:
            config_data = json.loads(json_config)
            
            if 'data_sources' not in config_data:
                raise ValueError("Invalid configuration format. Expected 'data_sources' as main key.")
                
            sources_data = config_data['data_sources']
            
            # Process global metadata if exists
            metadata_registry = MetadataRegistry()
            global_metadata = config_data.get('metadata', {})
            
            # Register metadata from files
            for file_path in global_metadata.get('files', []):
                try:
                    if os.path.exists(file_path):
                        metadata_registry.register_from_file(file_path)
                        logger.info(f"Metadata registered from file: {file_path}")
                except Exception as e:
                    logger.warning(f"Error loading metadata from file {file_path}: {str(e)}")
            
            # Register metadata defined inline
            for metadata_dict in global_metadata.get('datasets', []):
                try:
                    metadata_registry.register_from_dict(metadata_dict)
                    logger.info(f"Metadata registered for: {metadata_dict.get('name', 'unknown')}")
                except Exception as e:
                    logger.warning(f"Error registering metadata: {str(e)}")
            
            # Process semantic schemas if exist
            semantic_schemas = {}
            global_schemas = config_data.get('semantic_schemas', {})
            
            # Register schemas from files
            for file_path in global_schemas.get('files', []):
                try:
                    if os.path.exists(file_path):
                        schema = SemanticSchema.load_from_file(file_path)
                        semantic_schemas[schema.name] = schema
                        logger.info(f"Semantic schema registered from file: {file_path}")
                except Exception as e:
                    logger.warning(f"Error loading semantic schema from file {file_path}: {str(e)}")
                    
            # Register schemas defined inline
            for schema_dict in global_schemas.get('schemas', []):
                try:
                    schema = SemanticSchema.from_dict(schema_dict)
                    semantic_schemas[schema.name] = schema
                    logger.info(f"Semantic schema registered for: {schema.name}")
                except Exception as e:
                    logger.warning(f"Error registering semantic schema: {str(e)}")
            
            # Create connectors
            connectors = {}
            for source_config in sources_data:
                source_id = source_config.get('id')
                if not source_id:
                    raise ValueError("Source configuration without ID")
                
                # Check if it already has metadata or needs to fetch from registry
                if 'metadata' not in source_config:
                    dataset_name = source_config.get('dataset_name', source_id)
                    metadata = metadata_registry.get_metadata(dataset_name)
                    if metadata:
                        source_config['metadata'] = metadata.to_dict()
                        logger.info(f"Registry metadata applied to source {source_id}")
                
                # Check if it already has a semantic schema or needs to fetch from registry
                if 'semantic_schema' not in source_config:
                    schema_name = source_config.get('schema_name', source_id)
                    if schema_name in semantic_schemas:
                        source_config['semantic_schema'] = semantic_schemas[schema_name].to_dict()
                        logger.info(f"Semantic schema applied to source {source_id}")
                
                # Create connector
                connector = cls.create_connector(source_config)
                connectors[source_id] = connector
                
            return connectors
                
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON: {str(e)}")

            
def create_view_with_semantic_schema(schema: SemanticSchema, 
                                    sources: Dict[str, Union[pd.DataFrame, str]]) -> pd.DataFrame:
    """
    Create a view from multiple sources using a semantic schema.
    
    This function takes a semantic schema and a dictionary of sources (either
    DataFrames or file paths) and constructs a view according to the schema.
    
    Args:
        schema: Semantic schema for view construction.
        sources: Dictionary of source DataFrames or file paths.
        
    Returns:
        pd.DataFrame: Constructed view DataFrame.
    """
    # Convert file paths to DataFrames
    source_dfs = {}
    
    for name, source in sources.items():
        if isinstance(source, pd.DataFrame):
            # Already a DataFrame
            source_dfs[name] = source
        elif isinstance(source, str) and os.path.exists(source):
            # A file path - load as CSV
            try:
                source_dfs[name] = pd.read_csv(source)
                logger.info(f"Loaded source {name} from file: {source}")
            except Exception as e:
                logger.error(f"Error loading source {name} from file {source}: {str(e)}")
                raise ValueError(f"Could not load source {name} from file {source}: {str(e)}")
        else:
            raise ValueError(f"Unsupported source type for {name}: {type(source)}")
                
    # Create and return the view
    try:
        view_df = create_view_from_sources(schema, source_dfs)
        return view_df
        
    except Exception as e:
        error_msg = f"Error creating semantic view: {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg) from e