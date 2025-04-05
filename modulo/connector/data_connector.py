
import pandas as pd
import logging
from typing import Optional

from connector.semantic_layer_schema import TransformationRule, TransformationType

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DataConnector")

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