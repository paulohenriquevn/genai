
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
logger = logging.getLogger("DuckDBCsvConnector")


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
