import os
import json
import pandas as pd
import logging
import importlib
from typing import Any, Dict, Union, Type

from connector.metadata import MetadataRegistry
from connector.semantic_layer_schema import SemanticSchema
from connector.view_loader_and_transformer import create_view_from_sources
from connector.data_connector import DataConnector
from connector.datasource_config import DataSourceConfig

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
    # Map of connector types to their module/class information
    _connector_registry = {
        'csv': ('connector.csv_connector', 'CsvConnector'),
        'postgres': ('connector.postgres_connector', 'PostgresConnector'),
        'duckdb_csv': ('connector.duckdb_csv_connector', 'DuckDBCsvConnector')
    }
    
    # Cache of already loaded connector classes
    _connector_classes = {}
    
    @classmethod
    def register_connector(cls, source_type: str, connector_info: tuple) -> None:
        """
        Register a new connector type in the factory.
        
        Args:
            source_type: Data source type name.
            connector_info: Tuple of (module_path, class_name) or connector class
        """
        cls._connector_registry[source_type] = connector_info
        logger.info(f"Connector registered for type: {source_type}")
    
    @classmethod
    def _load_connector_class(cls, module_path: str, class_name: str) -> Type[DataConnector]:
        """
        Dynamically import and load a connector class.
        
        Args:
            module_path: Path to the module containing the connector class
            class_name: Name of the connector class
            
        Returns:
            Type[DataConnector]: The connector class
        """
        cache_key = f"{module_path}.{class_name}"
        
        # Check if we've already loaded this class
        if cache_key in cls._connector_classes:
            return cls._connector_classes[cache_key]
        
        try:
            # Dynamically import the module
            module = importlib.import_module(module_path)
            
            # Get the class from the module
            connector_class = getattr(module, class_name)
            
            # Cache the class for future use
            cls._connector_classes[cache_key] = connector_class
            
            return connector_class
        except (ImportError, AttributeError) as e:
            logger.error(f"Error loading connector class {class_name} from {module_path}: {str(e)}")
            raise ValueError(f"Failed to load connector class: {str(e)}")
    
    @classmethod
    def create_connector(cls, config) -> DataConnector:
        """
        Create a connector with metadata and semantic layer support.
        
        Args:
            config: Data source configuration.
            
        Returns:
            DataConnector: Created connector.
        """
        # Convert config to DataSourceConfig if necessary
        if not isinstance(config, DataSourceConfig):
            config = DataSourceConfig.from_dict(config)
        
        # Create the appropriate connector based on type
        source_type = config.source_type
        
        if source_type not in cls._connector_registry:
            raise ValueError(f"Unsupported connector type: {source_type}")
            
        connector_info = cls._connector_registry[source_type]
        
        # Handle different ways of specifying the connector
        if isinstance(connector_info, tuple) and len(connector_info) == 2:
            # It's a (module_path, class_name) tuple
            module_path, class_name = connector_info
            connector_class = cls._load_connector_class(module_path, class_name)
        elif isinstance(connector_info, type) and issubclass(connector_info, DataConnector):
            # It's already a class
            connector_class = connector_info
        else:
            raise ValueError(f"Invalid connector specification for {source_type}: {connector_info}")
        
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