"""
Exemplo de uso do sistema de construtores de queries.

Este script demonstra como utilizar o sistema de construtores de queries
em diferentes cenários, incluindo dados locais, bancos de dados SQL e views.
"""

import os
import logging
import pandas as pd
from connector.semantic_layer_schema import (
    SemanticSchema, 
    ColumnSchema,
    ColumnType,
    RelationSchema,
    TransformationType,
    TransformationRule
)
from connector.metadata import (
    DatasetMetadata,
    ColumnMetadata
)
from query_builders import (
    QueryBuilderFacade,
)
from connector.connectors import (
    DataSourceConfig,
    DataConnectorFactory,
)

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Diretório base para arquivos de dados
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dados")


def criar_esquemas_exemplo():
    """
    Cria exemplos de esquemas semânticos para demonstração.
    
    Returns:
        Dict[str, SemanticSchema]: Dicionário de esquemas.
    """
    # 1. Esquema para um arquivo CSV de vendas
    vendas_schema = SemanticSchema(
        name="vendas",
        description="Dados de vendas mensais",
        source_type="csv",
        source_path="vendas.csv",
        columns=[
            ColumnSchema(
                name="id_venda",
                type=ColumnType.INTEGER,
                description="ID único da venda",
                primary_key=True
            ),
            ColumnSchema(
                name="data",
                type=ColumnType.DATE,
                description="Data da venda"
            ),
            ColumnSchema(
                name="valor",
                type=ColumnType.FLOAT,
                description="Valor da venda"
            ),
            ColumnSchema(
                name="id_cliente",
                type=ColumnType.INTEGER,
                description="ID do cliente"
            )
        ],
        transformations=[
            TransformationRule(
                type=TransformationType.CONVERT_TYPE,
                column="data",
                params={"type": "datetime", "format": "%Y-%m-%d"}
            ),
            TransformationRule(
                type=TransformationType.ROUND,
                column="valor",
                params={"decimals": 2}
            )
        ]
    )
    
    # 2. Esquema para um arquivo CSV de clientes
    clientes_schema = SemanticSchema(
        name="clientes",
        description="Dados de clientes",
        source_type="csv",
        source_path="clientes.csv",
        columns=[
            ColumnSchema(
                name="id_cliente",
                type=ColumnType.INTEGER,
                description="ID único do cliente",
                primary_key=True
            ),
            ColumnSchema(
                name="nome",
                type=ColumnType.STRING,
                description="Nome do cliente"
            ),
            ColumnSchema(
                name="cidade",
                type=ColumnType.STRING,
                description="Cidade do cliente"
            ),
            ColumnSchema(
                name="estado",
                type=ColumnType.STRING,
                description="Estado do cliente"
            )
        ],
        transformations=[
            TransformationRule(
                type=TransformationType.UPPERCASE,
                column="estado",
                params={}
            )
        ]
    )
    
    # 3. Esquema para uma view combinando vendas e clientes
    vendas_clientes_schema = SemanticSchema(
        name="vendas_clientes",
        description="Visão integrada de vendas e clientes",
        columns=[
            ColumnSchema(
                name="vendas.data",
                type=ColumnType.DATE,
                description="Data da venda"
            ),
            ColumnSchema(
                name="vendas.valor",
                type=ColumnType.FLOAT,
                description="Valor da venda"
            ),
            ColumnSchema(
                name="clientes.nome",
                type=ColumnType.STRING,
                description="Nome do cliente"
            ),
            ColumnSchema(
                name="clientes.cidade",
                type=ColumnType.STRING,
                description="Cidade do cliente"
            )
        ],
        relations=[
            RelationSchema(
                source_table="vendas",
                source_column="id_cliente",
                target_table="clientes",
                target_column="id_cliente"
            )
        ],
        transformations=[
            TransformationRule(
                type=TransformationType.UPPERCASE,
                column="clientes.nome",
                params={}
            )
        ]
    )
    
    return {
        "vendas": vendas_schema,
        "clientes": clientes_schema,
        "vendas_clientes": vendas_clientes_schema
    }


def criar_exemplo_metadados():
    """
    Cria exemplos de metadados para demonstração.
    
    Returns:
        Dict[str, DatasetMetadata]: Dicionário de metadados.
    """
    # Metadados para o dataset de vendas
    vendas_metadata = DatasetMetadata(
        name="vendas",
        description="Dados de vendas mensais por região",
        source="sistema_erp",
        columns={
            "id_venda": ColumnMetadata(
                name="id_venda",
                description="ID único da venda",
                data_type="int",
                alias=["id", "codigo_venda"]
            ),
            "data": ColumnMetadata(
                name="data",
                description="Data da venda",
                data_type="date",
                format="YYYY-MM-DD",
                alias=["dt_venda", "data_venda"]
            ),
            "valor": ColumnMetadata(
                name="valor",
                description="Valor da venda em reais",
                data_type="float",
                display={"precision": 2, "unit": "R$"},
                aggregations=["sum", "avg", "max", "min"]
            ),
            "id_cliente": ColumnMetadata(
                name="id_cliente",
                description="ID do cliente",
                data_type="int",
                alias=["cliente_id", "cod_cliente"]
            )
        },
        owner="Departamento Comercial"
    )
    
    # Metadados para o dataset de clientes
    clientes_metadata = DatasetMetadata(
        name="clientes",
        description="Cadastro de clientes",
        source="sistema_crm",
        columns={
            "id_cliente": ColumnMetadata(
                name="id_cliente",
                description="ID único do cliente",
                data_type="int",
                alias=["id", "codigo_cliente"]
            ),
            "nome": ColumnMetadata(
                name="nome",
                description="Nome do cliente",
                data_type="str",
                alias=["nome_cliente", "cliente"]
            ),
            "cidade": ColumnMetadata(
                name="cidade",
                description="Cidade do cliente",
                data_type="str",
                alias=["municipio", "localidade"]
            ),
            "estado": ColumnMetadata(
                name="estado",
                description="Sigla do estado",
                data_type="str",
                alias=["uf", "sigla_estado"]
            )
        },
        owner="Atendimento ao Cliente"
    )
    
    return {
        "vendas": vendas_metadata,
        "clientes": clientes_metadata
    }


def configurar_conectores(schemas, metadados):
    """
    Configura conectores de dados para os esquemas.
    
    Args:
        schemas: Dicionário de esquemas.
        metadados: Dicionário de metadados.
        
    Returns:
        Dict[str, Any]: Dicionário de conectores.
    """
    # Cria configurações para os conectores
    configs = {}
    
    # Configuração para vendas
    configs["vendas"] = DataSourceConfig(
        source_id="vendas",
        source_type="duckdb_csv",
        metadata=metadados["vendas"],
        path=os.path.join(DATA_DIR, "vendas.csv"),
        delimiter=",",
        header=True,
        semantic_schema=schemas["vendas"]
    )
    
    # Configuração para clientes
    configs["clientes"] = DataSourceConfig(
        source_id="clientes",
        source_type="duckdb_csv",
        metadata=metadados["clientes"],
        path=os.path.join(DATA_DIR, "clientes.csv"),
        delimiter=",",
        header=True,
        semantic_schema=schemas["clientes"]
    )
    
    # Cria fábrica de conectores
    factory = DataConnectorFactory()
    
    # Cria e conecta os conectores
    conectores = {}
    for nome, config in configs.items():
        conector = factory.create_connector(config)
        try:
            conector.connect()
            conectores[nome] = conector
        except Exception as e:
            logger.error(f"Erro ao conectar {nome}: {str(e)}")
    
    return conectores


def exemplo_query_builder_local():
    """
    Exemplo de uso do construtor de queries para dados locais.
    """
    # Cria esquemas e metadados
    schemas = criar_esquemas_exemplo()
    
    # Cria a fachada
    facade = QueryBuilderFacade(base_path=DATA_DIR)
    
    # Registra os esquemas
    for nome, schema in schemas.items():
        facade.register_schema(schema)
    
    # Constrói queries para o esquema de vendas
    vendas_query = facade.build_query("vendas", "local")
    vendas_head = facade.build_head_query("vendas", 5, "local")
    vendas_count = facade.build_count_query("vendas", "local")
    
    print("\n=== EXEMPLO DE QUERY LOCAL ===")
    print("\nQuery completa para vendas:")
    print(vendas_query)
    print("\nQuery para as primeiras 5 linhas:")
    print(vendas_head)
    print("\nQuery para contagem de linhas:")
    print(vendas_count)


def exemplo_query_builder_view(conectores):
    """
    Exemplo de uso do construtor de queries para views.
    
    Args:
        conectores: Dicionário de conectores.
    """
    # Cria esquemas e metadados
    schemas = criar_esquemas_exemplo()
    
    # Cria loaders simulados para os conectores
    class SimpleLoader:
        def __init__(self, nome, conector):
            self.schema = schemas[nome]
            self.query_builder = conector
    
    # Cria a fachada
    facade = QueryBuilderFacade()
    
    # Registra os esquemas
    for nome, schema in schemas.items():
        facade.register_schema(schema)
    
    # Registra os loaders simulados
    loaders = {}
    for nome, conector in conectores.items():
        loader = SimpleLoader(nome, conector)
        loaders[nome] = loader
        facade.register_loader(nome, loader)
    
    # Constrói uma query para a view
    try:
        view_query = facade.build_query("vendas_clientes", "view")
        view_head = facade.build_head_query("vendas_clientes", 3, "view")
        
        print("\n=== EXEMPLO DE QUERY VIEW ===")
        print("\nQuery completa para a view vendas_clientes:")
        print(view_query)
        print("\nQuery para as primeiras 3 linhas da view:")
        print(view_head)
        
        # Cria definição SQL para a view
        view_sql = facade.create_sql_view("vendas_clientes_view", view_query)
        
        print("\nDefinição SQL para criar a view:")
        print(view_sql)
    except Exception as e:
        logger.error(f"Erro ao construir query para view: {str(e)}")


def exemplo_transformacao_dinamica():
    """
    Exemplo de adição dinâmica de transformações.
    """
    # Cria esquemas
    schemas = criar_esquemas_exemplo()
    
    # Cria a fachada
    facade = QueryBuilderFacade(base_path=DATA_DIR)
    
    # Registra os esquemas
    facade.register_schema(schemas["vendas"])
    
    # Constrói uma query inicial
    query_inicial = facade.build_query("vendas", "local")
    
    print("\n=== EXEMPLO DE TRANSFORMAÇÃO DINÂMICA ===")
    print("\nQuery inicial:")
    print(query_inicial)
    
    # Adiciona novas transformações
    facade.add_transformation_to_schema(
        "vendas",
        "normalize",
        "valor",
        {}
    )
    
    facade.add_transformation_to_schema(
        "vendas",
        "fill_na",
        "id_cliente",
        {"value": 0}
    )
    
    # Constrói a query com as novas transformações
    query_transformada = facade.build_query("vendas", "local")
    
    print("\nQuery após adição de transformações:")
    print(query_transformada)
    
    # Converte a query para outro dialeto
    query_sqlite = facade.transpile_query(
        query_transformada,
        "sqlite",
        "postgres"
    )
    
    print("\nQuery convertida para SQLite:")
    print(query_sqlite)


def criar_dados_exemplo():
    """
    Cria arquivos CSV de exemplo para os testes.
    
    Esta função cria os arquivos necessários no diretório de dados.
    """
    # Criar diretório de dados se não existir
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Dados de vendas
    vendas_data = {
        'id_venda': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'data': ['2023-01-15', '2023-01-20', '2023-02-05', '2023-02-10', 
                 '2023-02-15', '2023-03-01', '2023-03-10', '2023-03-20', 
                 '2023-04-05', '2023-04-15'],
        'valor': [150.50, 200.75, 300.00, 450.25, 180.90, 
                  520.30, 320.45, 290.60, 410.80, 350.20],
        'id_cliente': [101, 102, 101, 103, 104, 105, 102, 103, 104, 105]
    }
    
    # Dados de clientes
    clientes_data = {
        'id_cliente': [101, 102, 103, 104, 105],
        'nome': ['João Silva', 'Maria Santos', 'Carlos Oliveira', 
                'Ana Pereira', 'Paulo Souza'],
        'cidade': ['São Paulo', 'Rio de Janeiro', 'Belo Horizonte', 
                  'Curitiba', 'Brasília'],
        'estado': ['SP', 'RJ', 'MG', 'PR', 'DF']
    }
    
    # Criar DataFrames
    vendas_df = pd.DataFrame(vendas_data)
    clientes_df = pd.DataFrame(clientes_data)
    
    # Salvar como CSV
    vendas_df.to_csv(os.path.join(DATA_DIR, 'vendas.csv'), index=False)
    clientes_df.to_csv(os.path.join(DATA_DIR, 'clientes.csv'), index=False)
    
    print(f"Arquivos CSV criados no diretório: {DATA_DIR}")
    return True


def main():
    """
    Função principal para executar os exemplos.
    """
    print("=== SISTEMA DE CONSTRUÇÃO DE QUERIES AVANÇADO ===")
    
    # Cria dados de exemplo
    print("\nCriando dados de exemplo...")
    if not criar_dados_exemplo():
        print("Erro ao criar dados de exemplo.")
        return
    
    # Cria os esquemas e metadados
    print("\nConfigurando esquemas e metadados...")
    schemas = criar_esquemas_exemplo()
    metadados = criar_exemplo_metadados()
    
    # Configura conectores
    print("\nConfigurando conectores...")
    try:
        conectores = configurar_conectores(schemas, metadados)
    except Exception as e:
        print(f"Erro ao configurar conectores: {str(e)}")
        conectores = {}
    
    # Executa os exemplos
    exemplo_query_builder_local()
    
    if conectores:
        exemplo_query_builder_view(conectores)
    else:
        print("\nExemplo de view ignorado devido a erros nos conectores.")
    
    exemplo_transformacao_dinamica()
    
    print("\n=== EXEMPLOS CONCLUÍDOS ===")


if __name__ == "__main__":
    main()