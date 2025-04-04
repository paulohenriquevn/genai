# Documentação Técnica: Sistema de Conectores de Dados

## Visão Geral

Este sistema fornece uma infraestrutura robusta para conexão, leitura e transformação de dados de múltiplas fontes, com suporte avançado a metadados. A arquitetura é baseada em classes abstratas e padrões de design que permitem a extensibilidade e a manutenção eficiente.

## Características Principais

- **Suporte a múltiplas fontes de dados**: CSV, PostgreSQL e integração com DuckDB
- **Gestão de metadados**: Suporte para definição, armazenamento e utilização de metadados em nível de dataset e coluna
- **Transformações inteligentes**: Conversão automática de tipos baseada em metadados
- **Suporte a consultas SQL**: Para todas as fontes de dados, incluindo arquivos CSV
- **Operação em diretórios**: Capacidade de processar múltiplos arquivos CSV em um diretório
- **Tratamento de erros robusto**: Exceções específicas e logging detalhado

## Arquitetura do Sistema

O sistema está organizado em dois módulos principais:

1. **Módulo de metadados** (`metadata.py`): Responsável pela definição e gestão de metadados
2. **Módulo de conectores** (`connectors.py`): Implementa os diversos conectores de dados

### Diagrama de Classes

```
┌─────────────────┐      ┌───────────────────┐      ┌─────────────────────┐
│ ColumnMetadata  │      │  DatasetMetadata  │      │  MetadataRegistry   │
└─────────────────┘      └───────────────────┘      └─────────────────────┘
        ▲                        ▲                            ▲
        │                        │                            │
        └──────────┬─────────────┴────────────┐              │
                   │                          │              │
                   ▼                          ▼              ▼
          ┌─────────────────┐         ┌──────────────┐     ┌─────────────────────┐
          │ DataSourceConfig│◄────────┤DataConnector │     │DataConnectorFactory │
          └─────────────────┘         └──────────────┘     └─────────────────────┘
                                             ▲
                                             │
                   ┌───────────┬─────────────┴─────────────┐
                   │           │                           │
                   ▼           ▼                           ▼
          ┌─────────────┐ ┌──────────────┐      ┌───────────────────┐
          │CsvConnector │ │PostgresConn. │      │ DuckDBCsvConnector│
          └─────────────┘ └──────────────┘      └───────────────────┘
```

## Módulo de Metadados

### Classes Principais

#### `ColumnMetadata`

Define metadados para uma coluna específica de dados.

**Atributos principais:**
- `name`: Nome da coluna no dataset
- `description`: Descrição da finalidade/significado da coluna
- `data_type`: Tipo de dados esperado (str, int, float, date, etc)
- `format`: Formato específico (ex: YYYY-MM-DD para datas)
- `alias`: Nomes alternativos para a coluna
- `aggregations`: Agregações recomendadas (sum, avg, etc)
- `validation`: Regras de validação (min, max, etc)
- `display`: Preferências de exibição (precision, unit, etc)
- `tags`: Tags para categorização

**Métodos principais:**
- `from_dict()`: Cria uma instância a partir de um dicionário
- `to_dict()`: Converte os metadados para um dicionário

#### `DatasetMetadata`

Armazena metadados para um dataset completo, incluindo coleções de `ColumnMetadata`.

**Atributos principais:**
- `name`: Nome do dataset
- `description`: Descrição do dataset
- `source`: Origem dos dados
- `columns`: Dicionário de metadados de colunas
- `created_at`/`updated_at`: Timestamps de criação e atualização
- `version`: Versão dos metadados
- `_alias_lookup`: Mapeamento interno de aliases para nomes reais de colunas

**Métodos principais:**
- `from_dict()`, `from_json()`, `from_file()`: Métodos para criar instâncias
- `to_dict()`, `to_json()`, `save_to_file()`: Métodos para serialização
- `get_column_metadata()`: Obtém metadados de uma coluna específica
- `get_columns_by_tag()`, `get_columns_by_type()`: Filtros por tag e tipo
- `resolve_column_name()`: Resolve o nome real a partir de um alias

#### `MetadataRegistry`

Registro global (singleton) que gerencia metadados para múltiplos datasets.

**Métodos principais:**
- `register_metadata()`, `register_from_dict()`, `register_from_json()`, `register_from_file()`: Métodos para registro
- `get_metadata()`: Obtém metadados para um dataset específico
- `remove_metadata()`: Remove metadados para um dataset
- `list_datasets()`: Lista todos os datasets registrados
- `clear()`: Remove todos os metadados

## Módulo de Conectores

### Classes de Exceção

- `DataConnectionException`: Exceção para problemas de conexão
- `DataReadException`: Exceção para problemas de leitura de dados
- `ConfigurationException`: Exceção para problemas com configurações

### Classes Principais

#### `DataSourceConfig`

Configuração de fonte de dados com suporte a metadados.

**Atributos principais:**
- `source_id`: Identificador único da fonte
- `source_type`: Tipo da fonte de dados
- `params`: Parâmetros específicos para o conector
- `metadata`: Metadados do dataset

**Métodos principais:**
- `from_dict()`, `from_json()`: Métodos para criar instâncias
- `resolve_column_name()`: Resolve nome real de coluna a partir de alias
- `get_column_metadata()`: Obtém metadados para uma coluna
- `get_recommended_aggregations()`: Obtém agregações recomendadas
- `get_column_type()`, `get_column_format()`: Acessores para tipo e formato

#### `DataConnector` (Interface Abstrata)

Interface base para todos os conectores de dados.

**Métodos abstratos:**
- `connect()`: Estabelece conexão com a fonte
- `read_data()`: Lê dados da fonte
- `close()`: Fecha a conexão
- `is_connected()`: Verifica se a conexão está ativa

#### `CsvConnector`

Implementação concreta para conexão com arquivos CSV.

**Características:**
- Suporte para metadados de colunas
- Conversão automática de tipos
- Suporte para diretórios com múltiplos arquivos
- Concatenação automática de arquivos
- Suporte para consultas SQL via SQLite em memória

**Métodos principais:**
- `connect()`: Carrega o(s) arquivo(s) CSV
- `read_data()`: Lê e opcionalmente processa os dados
- `_apply_metadata_transformations()`: Aplica transformações baseadas em metadados
- `_convert_column_type()`: Converte uma coluna para o tipo especificado
- `_execute_query_on_directory()`: Executa consulta SQL em diretório

#### `PostgresConnector`

Implementação para conexão com bancos PostgreSQL.

**Características:**
- Conexão com PostgreSQL via psycopg2
- Execução de consultas SQL nativas
- Verificação de conexão ativa

#### `DuckDBCsvConnector`

Conector avançado que usa DuckDB para processar CSV de forma eficiente.

**Características:**
- Performance superior para grandes arquivos
- Suporte para consultas SQL otimizadas
- Processamento de diretórios com múltiplos arquivos
- Criação de visões combinadas
- Adaptação de consulta com base em metadados

**Métodos principais:**
- `connect()`: Estabelece conexão e registra arquivos como tabelas
- `read_data()`: Executa consultas SQL via DuckDB
- `_adapt_query()`: Adapta uma consulta com base em metadados
- `get_schema()`: Retorna o esquema das tabelas
- `sample_data()`: Fornece amostra dos dados

#### `DataConnectorFactory`

Factory para criar instâncias de conectores com base na configuração.

**Métodos principais:**
- `register_connector()`: Registra um novo tipo de conector
- `create_connector()`: Cria um conector com base na configuração
- `create_from_json()`: Cria múltiplos conectores a partir de JSON

## Fluxo de Uso Típico

1. **Configuração e inicialização:**
   ```python
   # Via dicionário
   config = {
       'id': 'vendas_dataset',
       'type': 'csv',
       'path': 'dados/vendas.csv',
       'delimiter': ';'
   }
   
   connector = DataConnectorFactory.create_connector(config)
   
   # Ou via JSON
   json_config = """
   {
     "data_sources": [
       {
         "id": "vendas_dataset",
         "type": "csv",
         "path": "dados/vendas.csv",
         "delimiter": ";"
       }
     ]
   }
   """
   
   connectors = DataConnectorFactory.create_from_json(json_config)
   connector = connectors['vendas_dataset']
   ```

2. **Conexão e leitura:**
   ```python
   # Conectar
   connector.connect()
   
   # Leitura simples
   df = connector.read_data()
   
   # Leitura com SQL
   df = connector.read_data("SELECT produto, SUM(valor) FROM csv GROUP BY produto")
   
   # Fechar conexão
   connector.close()
   ```

3. **Uso com metadados:**
   ```python
   # Definição de metadados
   column_metadata = [
       ColumnMetadata(
           name="data_venda",
           description="Data da venda",
           data_type="date",
           format="%Y-%m-%d",
           alias=["data", "dt_venda"]
       ),
       ColumnMetadata(
           name="valor",
           description="Valor da venda",
           data_type="float",
           aggregations=["sum", "avg", "max", "min"]
       )
   ]
   
   metadata = DatasetMetadata(
       name="vendas_dataset",
       description="Dataset de vendas",
       columns={m.name: m for m in column_metadata}
   )
   
   # Configuração com metadados
   config = DataSourceConfig(
       source_id="vendas_dataset",
       source_type="csv",
       metadata=metadata,
       path="dados/vendas.csv"
   )
   
   connector = DataConnectorFactory.create_connector(config.to_dict())
   ```

## Exemplos de Uso

### Exemplo 1: Leitura Básica de CSV

```python
from connector.connectors import DataConnectorFactory

config = {
    'id': 'vendas',
    'type': 'csv',
    'path': './dados/vendas.csv',
    'delimiter': ';'
}

connector = DataConnectorFactory.create_connector(config)
connector.connect()

# Leitura simples
df = connector.read_data()
print(f"Dados lidos: {len(df)} linhas")

# Leitura com SQL
df_filtrado = connector.read_data("SELECT * FROM csv WHERE valor > 1000")
print(f"Dados filtrados: {len(df_filtrado)} linhas")

connector.close()
```

### Exemplo 2: Uso de DuckDB para CSV Grande

```python
from connector.connectors import DataConnectorFactory

config = {
    'id': 'logs',
    'type': 'duckdb_csv',
    'path': './logs/',
    'pattern': '*.csv',
    'create_combined_view': True
}

connector = DataConnectorFactory.create_connector(config)
connector.connect()

# Consulta agregada
result = connector.read_data("""
    SELECT 
        date_trunc('day', timestamp) as dia,
        COUNT(*) as eventos,
        COUNT(DISTINCT user_id) as usuarios
    FROM csv_data_logs
    GROUP BY 1
    ORDER BY 1
""")

print(result.head())
connector.close()
```

### Exemplo 3: Metadados e Transformações

```python
import json
from connector.connectors import DataConnectorFactory
from connector.metadata import DatasetMetadata, ColumnMetadata

# Define metadados
metadata_json = """
{
    "name": "vendas_dataset",
    "description": "Dados de vendas mensais",
    "columns": [
        {
            "name": "data",
            "description": "Data da venda",
            "data_type": "date",
            "format": "%Y-%m-%d",
            "alias": ["dt_venda", "dia"]
        },
        {
            "name": "produto",
            "description": "Nome do produto",
            "data_type": "str",
            "tags": ["dimensao"]
        },
        {
            "name": "valor",
            "description": "Valor da venda",
            "data_type": "float",
            "aggregations": ["sum", "avg"],
            "tags": ["metrica"]
        }
    ]
}
"""

# Configuração com metadados
config = {
    'id': 'vendas',
    'type': 'csv',
    'path': './dados/vendas.csv',
    'delimiter': ',',
    'metadata': json.loads(metadata_json)
}

connector = DataConnectorFactory.create_connector(config)
connector.connect()

# Os dados já são convertidos automaticamente com base nos metadados
df = connector.read_data()
print(f"Tipo da coluna 'data': {df['data'].dtype}")
print(f"Tipo da coluna 'valor': {df['valor'].dtype}")

# Uso de aliases em consultas
query = "SELECT dia, SUM(valor) as total FROM csv GROUP BY dia"
result = connector.read_data(query)
print(result.head())

connector.close()
```

### Exemplo 4: Múltiplas Fontes de Dados

```python
from connector.connectors import DataConnectorFactory

# Configuração múltipla via JSON
json_config = """
{
    "metadata": {
        "files": ["./metadados/vendas_meta.json", "./metadados/estoque_meta.json"],
        "datasets": []
    },
    "data_sources": [
        {
            "id": "vendas",
            "type": "csv",
            "path": "./dados/vendas/",
            "pattern": "*.csv",
            "dataset_name": "vendas_dataset"
        },
        {
            "id": "estoque",
            "type": "postgres",
            "host": "localhost",
            "database": "warehouse",
            "username": "usuario",
            "password": "senha",
            "dataset_name": "estoque_dataset"
        }
    ]
}
"""

connectors = DataConnectorFactory.create_from_json(json_config)

# Uso do conector de vendas
vendas_conn = connectors['vendas']
vendas_conn.connect()
df_vendas = vendas_conn.read_data("SELECT produto, SUM(quantidade) FROM csv GROUP BY produto")

# Uso do conector de estoque
estoque_conn = connectors['estoque']
estoque_conn.connect()
df_estoque = estoque_conn.read_data("SELECT produto, quantidade FROM estoque")

# Análise combinada
import pandas as pd
df_combinado = pd.merge(df_vendas, df_estoque, on='produto')
print(df_combinado.head())

# Fechamento
for connector in connectors.values():
    connector.close()
```

## Extensão do Sistema

### Adicionando Novos Conectores

Para adicionar um novo tipo de conector:

1. Criar uma classe que implementa a interface `DataConnector`
2. Registrar o conector na factory

```python
class ExcelConnector(DataConnector):
    def __init__(self, config):
        self.config = config
        # ...implementação...
    
    def connect(self):
        # ...implementação...
    
    def read_data(self, query=None):
        # ...implementação...
    
    def close(self):
        # ...implementação...
    
    def is_connected(self):
        # ...implementação...

# Registro na factory
DataConnectorFactory.register_connector('excel', ExcelConnector)
```

### Extensão de Metadados

O sistema de metadados pode ser estendido para incluir mais atributos:

1. Atualizar as classes `ColumnMetadata` e/ou `DatasetMetadata`
2. Atualizar os métodos de serialização/desserialização
3. Implementar lógica para usar os novos atributos

## Considerações de Performance

- Para arquivos CSV pequenos, use `CsvConnector`
- Para arquivos CSV grandes ou consultas complexas, use `DuckDBCsvConnector`
- Para múltiplos arquivos em um diretório, use `DuckDBCsvConnector` com `create_combined_view=True`
- Utilize metadados para conversão automática de tipos, o que evita conversões repetidas

## Logging e Depuração

O sistema utiliza o módulo de logging do Python para registrar informações importantes:

```python
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

Para depuração mais detalhada:

```python
logging.getLogger("connector").setLevel(logging.DEBUG)
```

## Melhores Práticas

1. **Sempre feche os conectores após o uso**:
   ```python
   try:
       connector.connect()
       # ...operações...
   finally:
       connector.close()
   ```

2. **Use with para gerenciamento automático de recursos**:
   ```python
   class DataConnectorContext:
       def __init__(self, connector):
           self.connector = connector
       
       def __enter__(self):
           self.connector.connect()
           return self.connector
       
       def __exit__(self, exc_type, exc_val, exc_tb):
           self.connector.close()
   
   # Uso
   with DataConnectorContext(connector) as conn:
       df = conn.read_data()
   ```

3. **Defina metadados completos para melhor funcionamento**:
   - Sempre especifique `data_type`
   - Para datas, sempre forneça `format`
   - Use `alias` para facilitar consultas

4. **Considere o uso de Cache para consultas frequentes**:
   ```python
   from functools import lru_cache
   
   @lru_cache(maxsize=32)
   def cached_query(connector, query):
       return connector.read_data(query)
   ```

## Limitações Conhecidas

1. **Consultas SQL em CSV**: Limitadas à funcionalidade SQLite/DuckDB
2. **Joins em CsvConnector**: Só possíveis dentro do mesmo arquivo ou com arquivos concatenados
3. **Concorrência**: Não há suporte nativo para acesso concorrente
4. **Autenticação**: Senhas em texto plano nas configurações

## Futuras Melhorias

1. **Conectores adicionais**: Suporte para Parquet, Excel, JSON, APIs
2. **Suporte a credenciais seguras**: Integração com gerenciadores de segredos
3. **Cache inteligente**: Implementação de cache baseado em tempo e alterações
4. **Validação de dados**: Validação automática baseada em metadados
5. **Transformações complexas**: Pipeline de transformações configuráveis
6. **Inferência automática de metadados**: Detecção de tipos e estruturas

## Referências

- DuckDB: https://duckdb.org/docs/
- Pandas: https://pandas.pydata.org/docs/
- PostgreSQL Python: https://www.psycopg.org/docs/
- Python logging: https://docs.python.org/3/library/logging.html
- SQLite: https://docs.python.org/3/library/sqlite3.html