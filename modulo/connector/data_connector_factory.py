import os
import json
import pandas as pd
import logging
from typing import Any, Dict, Union

from connector.metadata import MetadataRegistry
from connector.semantic_layer_schema import SemanticSchema
from connector.view_loader_and_transformer import create_view_from_sources

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("connector")


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
        from modulo.connector.postgres_connector import CsvConnector, PostgresConnector, DuckDBCsvConnector, DataSourceConfig
        
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