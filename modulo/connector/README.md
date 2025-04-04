# Documentação do Sistema de Conexão e Transformação de Dados

## Visão Geral

Este sistema oferece uma infraestrutura robusta para conexão, transformação e análise de dados provenientes de múltiplas fontes. A arquitetura foi desenvolvida com foco em flexibilidade, extensibilidade e capacidade de lidar com metadados semânticos, permitindo uma melhor compreensão e utilização dos dados.

## Componentes Principais

O sistema é composto por quatro módulos principais:

1. **Camada Semântica**: Define esquemas e estruturas para descrever dados semanticamente
2. **Conectores de Dados**: Implementa interfaces para diferentes fontes de dados
3. **Metadados**: Gerencia informações descritivas sobre conjuntos de dados
4. **Carregador de Visualizações**: Cria visualizações personalizadas a partir dos dados carregados

### Diagrama de Arquitetura

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Conectores de  │     │     Camada      │     │   Metadados     │
│     Dados       │◄────┤    Semântica    │────►│                 │
└────────┬────────┘     └─────────────────┘     └────────┬────────┘
         │                                               │
         │              ┌─────────────────┐              │
         └─────────────►│   Carregador    │◄─────────────┘
                        │      de         │
                        │  Visualizações  │
                        └─────────────────┘
```

## Camada Semântica (semantic_layer_schema.py)

Este módulo define as estruturas fundamentais para representar dados semanticamente.

### Classes Principais

- **ColumnType**: Enumeração dos tipos de colunas suportados (`STRING`, `INTEGER`, `FLOAT`, etc.)
- **TransformationType**: Enumeração dos tipos de transformações disponíveis (`RENAME`, `FILLNA`, `CONVERT_TYPE`, etc.)
- **ColumnSchema**: Estrutura para metadados de colunas individuais
- **RelationSchema**: Define relacionamentos entre tabelas e colunas
- **TransformationRule**: Define regras de transformação para colunas
- **SemanticSchema**: Esquema semântico completo para fontes de dados

### Exemplo de Uso

```python
# Criando um esquema semântico para uma fonte de dados
schema = SemanticSchema(
    name="vendas_mensais",
    description="Dados de vendas mensais por região",
    source_type="csv",
    source_path="dados/vendas.csv",
    columns=[
        ColumnSchema(
            name="data",
            type=ColumnType.DATE,
            description="Data da venda",
            nullable=False
        ),
        ColumnSchema(
            name="regiao",
            type=ColumnType.STRING,
            description="Região da venda"
        ),
        ColumnSchema(
            name="valor",
            type=ColumnType.FLOAT,
            description="Valor da venda"
        )
    ],
    transformations=[
        TransformationRule(
            type=TransformationType.CONVERT_TYPE,
            column="data",
            params={"type": "date", "format": "%Y-%m-%d"}
        )
    ]
)

# Salvando o esquema em um arquivo
schema.save_to_file("schemas/vendas_schema.json")
```

## Metadados (metadata.py)

Este módulo gerencia metadados detalhados sobre conjuntos de dados e suas colunas.

### Classes Principais

- **ColumnMetadata**: Armazena metadados para uma coluna específica
- **DatasetMetadata**: Armazena metadados para um dataset completo
- **MetadataRegistry**: Registro global de metadados para múltiplos datasets

### Exemplo de Uso

```python
# Criando metadados para um dataset
metadados_vendas = DatasetMetadata(
    name="vendas_mensais",
    description="Dataset com dados de vendas mensais por região",
    source="sistema_erp",
    columns={
        "data": ColumnMetadata(
            name="data",
            description="Data da venda",
            data_type="date",
            format="YYYY-MM-DD",
            alias=["dt_venda", "data_venda"]
        ),
        "regiao": ColumnMetadata(
            name="regiao",
            description="Região onde a venda foi realizada",
            data_type="str",
            alias=["local", "area"]
        ),
        "valor": ColumnMetadata(
            name="valor",
            description="Valor da venda em reais",
            data_type="float",
            display={"precision": 2, "unit": "R$"},
            aggregations=["sum", "avg", "max", "min"]
        )
    },
    owner="Departamento Comercial"
)

# Registrando os metadados
registro = MetadataRegistry()
registro.register_metadata(metadados_vendas)

# Acessando metadados
coluna_info = registro.get_metadata("vendas_mensais").get_column_metadata("valor")
agregacoes = registro.get_metadata("vendas_mensais").get_recommended_aggregations("valor")
```

## Conectores de Dados (connectors.py)

Este módulo implementa interfaces para diferentes fontes de dados.

### Classes Principais

- **DataSourceConfig**: Configuração de fonte de dados com suporte a metadados
- **DataConnector**: Interface base para todos os conectores
- **CsvConnector**: Conector para arquivos CSV
- **PostgresConnector**: Conector para bancos de dados PostgreSQL
- **DuckDBCsvConnector**: Conector otimizado para CSV usando DuckDB
- **DataConnectorFactory**: Fábrica para criação de conectores

### Exemplo de Uso

```python
# Configuração de fonte de dados
config = DataSourceConfig(
    source_id="vendas_csv",
    source_type="csv",
    path="dados/vendas.csv",
    delimiter=";",
    encoding="utf-8"
)

# Criação e uso de um conector
factory = DataConnectorFactory()
conector = factory.create_connector(config)
conector.connect()

# Leitura de dados com uma query
dados = conector.read_data("SELECT regiao, SUM(valor) as total FROM csv GROUP BY regiao")
conector.close()
```

### Configuração via JSON

```python
# Criação de múltiplos conectores via JSON
json_config = """
{
    "data_sources": [
        {
            "id": "vendas_csv",
            "type": "csv",
            "path": "dados/vendas.csv",
            "delimiter": ";",
            "encoding": "utf-8"
        },
        {
            "id": "clientes_db",
            "type": "postgres",
            "host": "localhost",
            "database": "comercial",
            "username": "usuario",
            "password": "senha"
        }
    ]
}
"""

conectores = DataConnectorFactory.create_from_json(json_config)
conector_vendas = conectores["vendas_csv"]
conector_vendas.connect()
```

## Carregador de Visualizações (view_loader_and_transformer.py)

Este módulo permite a criação de visualizações personalizadas dos dados.

### Classes Principais

- **ViewLoader**: Carregador de visualizações com suporte à camada semântica e transformações

### Exemplo de Uso

```python
# Carregando um esquema semântico
schema = SemanticSchema.load_from_file("schemas/vendas_schema.json")

# Criando um carregador de visualizações
view_loader = ViewLoader(schema)

# Registrando fontes de dados
view_loader.register_source("vendas", df_vendas)
view_loader.register_source("clientes", df_clientes)

# Construindo a visualização
view_df = view_loader.construct_view()

# Função auxiliar
df_resultado = create_view_from_sources(schema, {
    "vendas": df_vendas,
    "clientes": df_clientes
})
```

## Fluxo de Trabalho Típico

1. **Definir Esquemas Semânticos**:
   - Criar esquemas que descrevam a estrutura, tipos e relações dos dados
   - Definir transformações necessárias para normalizar e preparar os dados

2. **Preparar Metadados**:
   - Criar metadados detalhados para cada conjunto de dados
   - Registrar os metadados no sistema para uso pelos conectores

3. **Configurar Conectores**:
   - Definir as configurações para cada fonte de dados
   - Criar conectores apropriados para cada fonte

4. **Carregar e Transformar Dados**:
   - Conectar às fontes de dados
   - Ler os dados brutos
   - Aplicar transformações baseadas nos esquemas semânticos

5. **Criar Visualizações**:
   - Utilizar o ViewLoader para construir visualizações personalizadas
   - Aplicar relações e junções entre diferentes fontes de dados

## Exemplo Completo

```python
# 1. Definindo o esquema semântico
from connector.semantic_layer_schema import (
    SemanticSchema, ColumnSchema, RelationSchema, 
    TransformationRule, ColumnType, TransformationType
)

schema = SemanticSchema(
    name="analise_vendas",
    description="Análise integrada de vendas e clientes",
    columns=[
        ColumnSchema(name="id_venda", type=ColumnType.INTEGER, primary_key=True),
        ColumnSchema(name="data_venda", type=ColumnType.DATE),
        ColumnSchema(name="valor", type=ColumnType.FLOAT),
        ColumnSchema(name="id_cliente", type=ColumnType.INTEGER),
        ColumnSchema(name="nome_cliente", type=ColumnType.STRING)
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
            type=TransformationType.CONVERT_TYPE,
            column="data_venda",
            params={"type": "datetime", "format": "%Y-%m-%d"}
        ),
        TransformationRule(
            type=TransformationType.FILLNA,
            column="valor",
            params={"value": 0.0}
        )
    ]
)

# 2. Configurando os conectores
from connector.connectors import DataSourceConfig, DataConnectorFactory

config_vendas = DataSourceConfig(
    source_id="vendas",
    source_type="csv",
    path="dados/vendas.csv"
)

config_clientes = DataSourceConfig(
    source_id="clientes",
    source_type="csv",
    path="dados/clientes.csv"
)

# 3. Carregando os dados
factory = DataConnectorFactory()
conector_vendas = factory.create_connector(config_vendas)
conector_clientes = factory.create_connector(config_clientes)

conector_vendas.connect()
conector_clientes.connect()

df_vendas = conector_vendas.read_data()
df_clientes = conector_clientes.read_data()

# 4. Criando a visualização integrada
from connector.view_loader_and_transformer import create_view_from_sources

df_analise = create_view_from_sources(
    schema,
    {
        "vendas": df_vendas,
        "clientes": df_clientes
    }
)

# 5. Análise dos dados
print(f"Total de vendas: {df_analise['valor'].sum()}")
print(f"Venda média por cliente: {df_analise.groupby('nome_cliente')['valor'].mean()}")
```

## Requisitos de Sistema

- Python 3.7 ou superior
- pandas
- duckdb
- psycopg2 (para conexões PostgreSQL)

## Práticas Recomendadas

1. **Esquemas Semânticos**:
   - Defina esquemas completos e detalhados para seus dados
   - Utilize transformações para normalizar os dados na entrada

2. **Metadados**:
   - Crie metadados descritivos e precisos para todas as colunas
   - Mantenha os metadados atualizados à medida que os dados evoluem

3. **Conectores**:
   - Utilize o tipo de conector mais adequado para cada fonte de dados
   - Para arquivos CSV grandes, prefira o DuckDBCsvConnector

4. **Visualizações**:
   - Defina relacionamentos claros entre tabelas
   - Utilize agregações recomendadas nos metadados

## Solução de Problemas

### Problemas Comuns

1. **Erro ao conectar a uma fonte de dados**:
   - Verifique se o caminho ou credenciais estão corretos
   - Confirme que o arquivo ou banco de dados existe

2. **Erro ao transformar dados**:
   - Verifique se os tipos de dados são compatíveis
   - Certifique-se de que as transformações são aplicáveis aos dados

3. **Erro ao construir visualizações**:
   - Verifique se todas as tabelas necessárias foram registradas
   - Confirme que as relações entre tabelas estão corretamente definidas

4. **Resultados inesperados em consultas**:
   - Utilize o método `sample_data()` para inspecionar os dados brutos
   - Verifique a estrutura das tabelas com `get_schema()`
   - Ative o logging para debug detalhado

### Ativando Logging Detalhado

```python
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

## Extensão do Sistema

O sistema foi projetado para ser facilmente extensível:

1. **Novos Conectores**: Crie uma nova classe que herde de `DataConnector`
2. **Novos Tipos de Transformação**: Adicione novos valores à enumeração `TransformationType`
3. **Novos Tipos de Colunas**: Adicione novos valores à enumeração `ColumnType`

### Exemplo de Extensão: Conector MongoDB

```python
class MongoDBConnector(DataConnector):
    """Conector para MongoDB com suporte à camada semântica."""
    
    def __init__(self, config: DataSourceConfig):
        self.config = config
        self.client = None
        self.db = None
        
        # Validação de parâmetros obrigatórios
        required_params = ['uri', 'database', 'collection']
        missing_params = [p for p in required_params if p not in self.config.params]
        
        if missing_params:
            raise ConfigurationException(
                f"Parâmetros obrigatórios ausentes: {', '.join(missing_params)}"
            )
    
    def connect(self) -> None:
        try:
            from pymongo import MongoClient
            
            self.client = MongoClient(self.config.params['uri'])
            self.db = self.client[self.config.params['database']]
            
        except ImportError:
            raise DataConnectionException("Módulo pymongo não encontrado")
        except Exception as e:
            raise DataConnectionException(f"Erro ao conectar ao MongoDB: {str(e)}")
    
    def read_data(self, query: Optional[str] = None) -> pd.DataFrame:
        if not self.is_connected():
            raise DataConnectionException("Não conectado ao MongoDB")
            
        collection = self.db[self.config.params['collection']]
        
        if query:
            # Converte a string de consulta em um dict
            import json
            query_dict = json.loads(query)
            cursor = collection.find(query_dict)
        else:
            cursor = collection.find()
            
        # Converte o cursor para DataFrame
        df = pd.DataFrame(list(cursor))
        
        # Aplica transformações da camada semântica
        df = self.apply_semantic_transformations(df)
        
        return df
    
    # Implementação dos métodos restantes...

# Registrar o novo conector na fábrica
DataConnectorFactory.register_connector("mongodb", MongoDBConnector)
```

## Conclusão

Este sistema fornece uma arquitetura flexível e poderosa para conexão, transformação e análise de dados de múltiplas fontes. Ao utilizar esquemas semânticos e metadados, ele permite uma compreensão mais profunda dos dados e facilita a criação de visualizações integradas.

A implementação modular permite fácil extensão para novos tipos de fontes de dados e transformações, tornando o sistema adaptável a diferentes necessidades e casos de uso.