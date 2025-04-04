import pandas as pd
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("semantic_example")

# Import the necessary components
from connector.semantic_layer_schema import (
    ColumnSchema,
    RelationSchema,
    SemanticSchema, 
    ColumnType, 
    TransformationType, 
    TransformationRule
)
from connector.metadata import ColumnMetadata, DatasetMetadata
from connector.connectors import (
    DataSourceConfig,
    CsvConnector,
    DuckDBCsvConnector,
    DataConnectorFactory
)

# Import the fixed implementation
from connector.connectors import create_view_with_semantic_schema

def example_basic_connector():
    """
    Demonstrates basic connector usage with semantic transformations.
    """
    # Create sample data
    sales_df = pd.DataFrame({
        'sale_id': [1, 2, 3, 4],
        'customer_id': [101, 102, 103, 104],
        'product_name': ['Laptop', 'Phone', 'Tablet', 'Watch'],
        'sale_amount': [1200.0, 800.0, 500.0, None],
        'sale_date': ['2023-01-15', '2023-02-20', '2023-03-10', '2023-04-05']
    })
    
    # Save to a temporary CSV file
    sales_file = "temp_sales.csv"
    sales_df.to_csv(sales_file, index=False)
    
    # Create a semantic schema
    sales_schema = SemanticSchema(
        name='sales_analysis',
        description='Sales data analysis view',
        source_type='csv',
        columns=[
            ColumnSchema(name='sale_id', type=ColumnType.INTEGER, primary_key=True),
            ColumnSchema(name='customer_id', type=ColumnType.INTEGER),
            ColumnSchema(name='product_name', type=ColumnType.STRING),
            ColumnSchema(name='sale_amount', type=ColumnType.FLOAT),
            ColumnSchema(name='sale_date', type=ColumnType.DATE)
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
    
    # Create connector configuration
    config = DataSourceConfig(
        source_id="sales_data",
        source_type="csv",
        semantic_schema=sales_schema,
        path=sales_file
    )
    
    # Create and connect
    connector = CsvConnector(config)
    connector.connect()
    
    # Read data - transformations should be applied automatically
    result_df = connector.read_data()
    
    print("Basic Connector Example:")
    print(result_df.dtypes)
    print(result_df.head())
    
    # Clean up
    connector.close()
    if os.path.exists(sales_file):
        os.remove(sales_file)

def example_view_construction():
    """
    Demonstrates semantic view construction with multiple data sources.
    """
    # Create sample data
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
    
    # Save to temporary CSV files
    sales_file = "temp_sales.csv"
    customers_file = "temp_customers.csv"
    sales_df.to_csv(sales_file, index=False)
    customers_df.to_csv(customers_file, index=False)
    
    # Create a semantic schema with relationships
    sales_view_schema = SemanticSchema(
        name='sales_analysis',
        description='Combined sales and customer data',
        source_type='csv',
        columns=[
            ColumnSchema(name='sale_id', type=ColumnType.INTEGER, primary_key=True),
            ColumnSchema(name='customer_id', type=ColumnType.INTEGER),
            ColumnSchema(name='product_name', type=ColumnType.STRING),
            ColumnSchema(name='sale_amount', type=ColumnType.FLOAT),
            ColumnSchema(name='sale_date', type=ColumnType.DATE),
            ColumnSchema(name='customer_name', type=ColumnType.STRING),
            ColumnSchema(name='customer_city', type=ColumnType.STRING)
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
    
    # Use the fixed function to create the view
    sources = {
        'sales': sales_file,
        'customers': customers_file
    }
    
    # Create the view using the fixed function
    view_df = create_view_with_semantic_schema(sales_view_schema, sources)
    
    print("\nSemantic View Construction Example:")
    print(view_df.columns.tolist())
    print(view_df.head())
    
    # Clean up
    if os.path.exists(sales_file):
        os.remove(sales_file)
    if os.path.exists(customers_file):
        os.remove(customers_file)

def example_duckdb_with_semantic():
    """
    Demonstrates DuckDB connector with semantic layer integration.
    """
    # Create sample data
    sales_df = pd.DataFrame({
        'sale_id': [1, 2, 3, 4, 5],
        'customer_id': [101, 102, 103, 104, 101],
        'product_name': ['Laptop', 'Phone', 'Tablet', 'Watch', 'Headphones'],
        'sale_amount': [1200.0, 800.0, 500.0, 300.0, 150.0],
        'sale_date': ['2023-01-15', '2023-02-20', '2023-03-10', '2023-04-05', '2023-01-30']
    })
    
    # Save to a temporary CSV file
    sales_file = "temp_duckdb_sales.csv"
    sales_df.to_csv(sales_file, index=False)
    
    # Create semantic transformations for analytics
    analytics_schema = SemanticSchema(
        name='sales_analytics',
        description='Sales data with analytical transformations',
        source_type='duckdb_csv',
        columns=[
            ColumnSchema(name='sale_id', type=ColumnType.INTEGER, primary_key=True),
            ColumnSchema(name='customer_id', type=ColumnType.INTEGER),
            ColumnSchema(name='product_name', type=ColumnType.STRING),
            ColumnSchema(name='sale_amount', type=ColumnType.FLOAT),
            ColumnSchema(name='sale_date', type=ColumnType.DATE)
        ],
        transformations=[
            # Convert date 
            TransformationRule(
                type=TransformationType.CONVERT_TYPE, 
                column='sale_date', 
                params={'type': 'datetime', 'format': '%Y-%m-%d'}
            ),
            # Fill NA values
            TransformationRule(
                type=TransformationType.FILLNA, 
                column='sale_amount', 
                params={'value': 0.0}
            )
        ]
    )
    
    # Create connector configuration
    config = DataSourceConfig(
        source_id="sales_analytics",
        source_type="duckdb_csv",
        semantic_schema=analytics_schema,
        path=sales_file,
        delimiter=","
    )
    
    # Create and connect
    connector = DuckDBCsvConnector(config)
    connector.connect()
    
    # Read data with transformations
    result_df = connector.read_data()
    
    print("\nDuckDB with Semantic Layer Example:")
    print(result_df.columns.tolist())
    print(result_df.head())
    
    # Try executing a SQL query that benefits from DuckDB
    query = """
    SELECT 
        product_name,
        SUM(sale_amount) as total_sales,
        AVG(sale_amount) as avg_sales
    FROM csv
    GROUP BY product_name
    ORDER BY total_sales DESC
    """
    
    query_result = connector.read_data(query)
    print("\nSQL Query Result:")
    print(query_result)
    
    # Clean up
    connector.close()
    if os.path.exists(sales_file):
        os.remove(sales_file)

def example_json_config():
    """
    Demonstrates creating connectors from a JSON configuration.
    """
    # Create sample files
    sales_df = pd.DataFrame({
        'sale_id': [1, 2, 3, 4],
        'customer_id': [101, 102, 103, 104],
        'product_name': ['Laptop', 'Phone', 'Tablet', 'Watch'],
        'sale_amount': [1200.0, 800.0, 500.0, None],
        'sale_date': ['2023-01-15', '2023-02-20', '2023-03-10', '2023-04-05']
    })
    
    products_df = pd.DataFrame({
        'product_name': ['Laptop', 'Phone', 'Tablet', 'Watch', 'Headphones'],
        'category': ['Computer', 'Electronics', 'Electronics', 'Accessories', 'Accessories'],
        'price': [1200.0, 800.0, 500.0, 300.0, 150.0]
    })
    
    # Save to temporary CSV files
    sales_file = "json_config_sales.csv"
    products_file = "json_config_products.csv"
    sales_df.to_csv(sales_file, index=False)
    products_df.to_csv(products_file, index=False)
    
    # Create a JSON configuration
    json_config = f"""
    {{
        "data_sources": [
            {{
                "id": "sales",
                "type": "csv",
                "path": "{sales_file}",
                "schema_name": "sales_schema"
            }},
            {{
                "id": "products",
                "type": "duckdb_csv",
                "path": "{products_file}",
                "schema_name": "products_schema"
            }}
        ],
        "semantic_schemas": {{
            "schemas": [
                {{
                    "name": "sales_schema",
                    "description": "Sales data schema",
                    "source_type": "csv",
                    "columns": [
                        {{"name": "sale_id", "type": "int", "primary_key": true}},
                        {{"name": "customer_id", "type": "int"}},
                        {{"name": "product_name", "type": "string"}},
                        {{"name": "sale_amount", "type": "float"}},
                        {{"name": "sale_date", "type": "date"}}
                    ],
                    "transformations": [
                        {{
                            "type": "fill_na",
                            "column": "sale_amount",
                            "params": {{"value": 0.0}}
                        }},
                        {{
                            "type": "convert_type",
                            "column": "sale_date",
                            "params": {{"type": "datetime", "format": "%Y-%m-%d"}}
                        }}
                    ]
                }},
                {{
                    "name": "products_schema",
                    "description": "Products data schema",
                    "source_type": "csv",
                    "columns": [
                        {{"name": "product_name", "type": "string", "primary_key": true}},
                        {{"name": "category", "type": "string"}},
                        {{"name": "price", "type": "float"}}
                    ]
                }}
            ]
        }}
    }}
    """
    
    # Create connectors from JSON
    connectors = DataConnectorFactory.create_from_json(json_config)
    
    print("\nJSON Configuration Example:")
    print(f"Created connectors: {list(connectors.keys())}")
    
    # Connect and read data
    for name, connector in connectors.items():
        connector.connect()
        data = connector.read_data()
        print(f"\nData from {name}:")
        print(data.head())
        connector.close()
    
    # Clean up
    if os.path.exists(sales_file):
        os.remove(sales_file)
    if os.path.exists(products_file):
        os.remove(products_file)

if __name__ == "__main__":
    # Run the examples
    example_basic_connector()
    example_view_construction()
    example_duckdb_with_semantic()
    example_json_config()