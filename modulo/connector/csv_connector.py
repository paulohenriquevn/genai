import os
import glob
import pandas as pd
import logging
from typing import Optional, Union

from connector.data_connector import DataConnector
from connector.datasource_config import DataSourceConfig
from modulo.connector.exceptions import ConfigurationException, DataConnectionException, DataReadException

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("connector")


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
