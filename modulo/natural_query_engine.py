"""
Natural Language Query Engine
============================

Este módulo implementa um motor completo para processamento de consultas
em linguagem natural sobre dados estruturados, usando os seguintes componentes:

1. Conectores de dados (connector) - Para carregar e processar dados de diferentes fontes
2. Core - Para execução de código, gerenciamento de estado e formatação de respostas
3. Query builders - Para construção e otimização de consultas SQL
4. Utils - Para análise de datasets e geração de metadados

O fluxo de trabalho é:
1. Carregar dados usando conectores apropriados
2. Aceitar consulta em linguagem natural do usuário
3. Gerar código Python/SQL para responder à consulta
4. Executar o código de forma segura
5. Processar e formatar o resultado
6. Lidar com erros e tentar corrigi-los automaticamente
"""

import os
import sys
import logging
import json
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Union, Tuple

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("natural_query_engine.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("natural_query_engine")

# Importação dos módulos connector
from connector.connectors import (
    DataConnector,
    DataConnectorFactory, 
    DataSourceConfig,
    DataConnectionException, 
    DataReadException,
    DuckDBCsvConnector
)

from connector.metadata import (
    MetadataRegistry,
    DatasetMetadata,
    ColumnMetadata
)

from connector.semantic_layer_schema import (
    SemanticSchema, 
    ColumnSchema, 
    TransformationType, 
    TransformationRule,
    ColumnType,
    RelationSchema
)

# Importação dos módulos core
from core.agent.state import AgentState, AgentMemory, AgentConfig
from core.code_executor import AdvancedDynamicCodeExecutor
from core.dataframe import DataFrameWrapper
from core.prompts import (
    get_chat_prompt_for_sql, 
    get_correct_error_prompt_for_sql,
    get_correct_output_type_error_prompt
)
from core.response import ResponseParser
from core.response.error import ErrorResponse
from core.response.base import BaseResponse
from core.user_query import UserQuery
from core.exceptions import (
    InvalidOutputValueMismatch, 
    ExecuteSQLQueryNotUsed, 
    InvalidLLMOutputType,
    UnknownLLMOutputType,
    TemplateRenderError
)

# Importação dos módulos query_builders
from query_builders.query_builder_base import BaseQueryBuilder, QuerySQLTransformationManager
from query_builders.query_builders_implementation import (
    LocalQueryBuilder,
    SqlQueryBuilder,
    ViewQueryBuilder,
    SQLParser
)
from query_builders.query_facade import QueryBuilderFacade

# Importação dos módulos utils
from utils.dataset_analyzer import DatasetAnalyzer


class NaturalLanguageQueryEngine:
    """
    Motor de processamento de consultas em linguagem natural.
    
    Esta classe coordena o processo completo de:
    1. Carregar e preparar os dados
    2. Receber consultas em linguagem natural
    3. Gerar código Python/SQL para executar a consulta
    4. Executar o código de forma segura
    5. Processar e formatar os resultados
    6. Lidar com erros e tentar recuperação automática
    """
    
    def __init__(
        self,
        data_config_path: Optional[str] = None,
        metadata_config_path: Optional[str] = None,
        base_data_path: Optional[str] = None,
        output_types: Optional[List[str]] = None,
        llm_config_path: Optional[str] = None
    ):
        """
        Inicializa o motor de consulta em linguagem natural.
        
        Args:
            data_config_path: Caminho para arquivo de configuração de fontes de dados
            metadata_config_path: Caminho para arquivo de metadados
            base_data_path: Diretório base onde os arquivos de dados estão localizados
            output_types: Lista de tipos de saída suportados (string, number, dataframe, plot)
            llm_config_path: Caminho para arquivo de configuração do LLM
        """
        # Diretórios base
        self.base_data_path = base_data_path or os.getcwd()
        
        # DataFrames carregados
        self.dataframes = {}
        
        # Conectores de dados
        self.connectors = {}
        
        # Estado do agente
        self.agent_state = AgentState(
            dfs=[],
            output_type=None,
            memory=AgentMemory(
                agent_description="Motor de consulta em linguagem natural"
            ),
            config=AgentConfig(
                direct_sql=False
            )
        )
        
        # Executor de código
        self.code_executor = AdvancedDynamicCodeExecutor()
        
        # Parser de respostas
        self.response_parser = ResponseParser()
        
        # Metadados
        self.metadata_registry = MetadataRegistry()
        if metadata_config_path and os.path.exists(metadata_config_path):
            self._load_metadata(metadata_config_path)
            
        # Tipos de saída suportados
        self.output_types = output_types or ["string", "number", "dataframe", "plot"]
        
        # Carrega dados iniciais
        if data_config_path and os.path.exists(data_config_path):
            self._load_data_from_config(data_config_path)
            
        # Inicializa o sistema
        logger.info("Motor de consulta em linguagem natural inicializado")
    
    def _load_metadata(self, metadata_path: str):
        """
        Carrega metadados de um arquivo JSON.
        
        Args:
            metadata_path: Caminho para o arquivo de metadados
        """
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata_config = json.load(f)
            
            # Registra metadados
            if "datasets" in metadata_config:
                for dataset_meta in metadata_config["datasets"]:
                    # Certifica-se de que o metadado tem os campos necessários
                    if "name" not in dataset_meta:
                        raise ValueError("O metadado do dataset deve conter o campo 'name'")
                    
                    # Cria e registra o metadado
                    name = dataset_meta["name"]
                    description = dataset_meta.get("description", f"Dataset {name}")
                    source = dataset_meta.get("source", "unknown")
                    
                    # Processa os metadados de colunas
                    columns = {}
                    if "columns" in dataset_meta:
                        for col_meta in dataset_meta["columns"]:
                            col_name = col_meta.get("name")
                            if col_name:
                                columns[col_name] = ColumnMetadata(
                                    name=col_name,
                                    description=col_meta.get("description", col_name),
                                    type=col_meta.get("type", "string"),
                                    is_categorical=col_meta.get("is_categorical", False),
                                    is_temporal=col_meta.get("is_temporal", False),
                                    is_numeric=col_meta.get("is_numeric", False),
                                    format=col_meta.get("format", None),
                                    constraints=col_meta.get("constraints", None)
                                )
                    
                    # Registra o metadado do dataset
                    self.metadata_registry.register_dataset(
                        DatasetMetadata(
                            name=name,
                            description=description,
                            source=source,
                            columns=columns,
                            relationships=dataset_meta.get("relationships", [])
                        )
                    )
            
            logger.info(f"Metadados carregados do arquivo: {metadata_path}")
        except Exception as e:
            logger.error(f"Erro ao carregar metadados: {str(e)}")
            raise
    
    def _load_data_from_config(self, config_path: str):
        """
        Carrega dados a partir de um arquivo de configuração.
        
        Args:
            config_path: Caminho para o arquivo de configuração
        """
        try:
            # Carrega a configuração
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Verifica se a configuração tem a estrutura esperada
            if "data_sources" not in config:
                raise ValueError("Formato de configuração inválido: 'data_sources' não encontrado")
            
            # Processa cada fonte de dados
            for source_config in config["data_sources"]:
                # Verifica campos obrigatórios
                if "id" not in source_config or "type" not in source_config:
                    continue
                
                source_id = source_config["id"]
                source_type = source_config["type"]
                path = source_config.get("path", None)
                
                # Ajusta o caminho para usar o diretório base, se necessário
                if path and not os.path.isabs(path):
                    path = os.path.join(self.base_data_path, path)
                    source_config["path"] = path
                
                # Cria a configuração
                connector_config = DataSourceConfig(
                    source_id=source_id,
                    source_type=source_type,
                    **{k: v for k, v in source_config.items() if k not in ["id", "type"]}
                )
                
                # Cria e inicializa o conector
                try:
                    connector = DataConnectorFactory.create_connector(connector_config)
                    connector.connect()
                    
                    # Aplica metadados, se disponíveis
                    if source_id in self.metadata_registry.datasets:
                        connector.apply_metadata(self.metadata_registry.datasets[source_id])
                    
                    # Armazena o conector
                    self.connectors[source_id] = connector
                    
                    # Carrega o DataFrame, se possível
                    df = connector.read_data()
                    wrapper = DataFrameWrapper(
                        dataframe=df,
                        name=source_id,
                        description=f"DataFrame carregado de {source_id}",
                        source=source_type
                    )
                    self.dataframes[source_id] = wrapper
                    logger.info(f"Dataframe '{source_id}' carregado com {len(df)} registros")
                except Exception as e:
                    logger.error(f"Erro ao inicializar conector {source_id}: {str(e)}")
                    continue
            
            # Inicializa o conector SQL combinado, se tivermos DataFrames
            if self.dataframes:
                # Adiciona os DataFrames ao estado do agente
                self.agent_state.dfs = list(self.dataframes.values())
                
                try:
                    # Inicializa um conector SQL para todos os DataFrames
                    sql_config = DataSourceConfig(
                        source_id="sql_connector",
                        source_type="duckdb_csv",
                        path=self.base_data_path,
                        pattern="*.csv"
                    )
                    sql_connector = DuckDBCsvConnector(sql_config)
                    sql_connector.connect()
                    self.connectors["sql_connector"] = sql_connector
                except Exception as e:
                    logger.error(f"Erro ao inicializar conector SQL: {str(e)}")
            
            logger.info(f"Carregados {len(self.connectors)} conectores de dados")
            
        except Exception as e:
            logger.error(f"Erro ao carregar configuração de dados: {str(e)}")
            raise
    
    def load_data(
        self, 
        data: pd.DataFrame, 
        name: str,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Carrega um DataFrame diretamente.
        
        Args:
            data: DataFrame a ser carregado
            name: Nome para o DataFrame
            description: Descrição opcional
            metadata: Metadados opcionais
        """
        wrapper = DataFrameWrapper(
            dataframe=data,
            name=name,
            description=description or f"DataFrame {name}",
            metadata=metadata or {}
        )
        
        self.dataframes[name] = wrapper
        self.agent_state.dfs = list(self.dataframes.values())
        
        logger.info(f"DataFrame '{name}' carregado manualmente com {len(data)} registros")
    
    def infer_output_type(self, query: str) -> str:
        """
        Infere o tipo de saída esperado com base na consulta.
        
        Args:
            query: Consulta em linguagem natural
            
        Returns:
            Tipo de saída inferido ('string', 'number', 'dataframe', 'plot')
        """
        query_lower = query.lower()
        
        # Conjuntos de palavras-chave para cada tipo de saída
        visualization_keywords = [
            "gráfico", "visualize", "visualização", "plot", "plote", 
            "mostre graficamente", "histograma", "diagrama", "mapa", "barra", "pizza"
        ]
        
        table_keywords = [
            "tabela", "dataframe", "dados", "linhas", "registros", "mostre",
            "lista", "exiba", "liste", "selecione", "filtre"
        ]
        
        number_keywords = [
            "quantos", "conte", "total", "soma", "média", "mediana", "máximo",
            "mínimo", "percentual", "percentagem", "contagem", "valor", "calculate"
        ]
        
        # Checa palavras-chave para determinar o tipo
        if any(keyword in query_lower for keyword in visualization_keywords):
            return "plot"
        elif any(keyword in query_lower for keyword in table_keywords):
            return "dataframe"
        elif any(keyword in query_lower for keyword in number_keywords):
            return "number"
        
        # Default para string se não conseguir inferir
        return "string"
    
    def execute_sql_query(self, sql_query: str) -> pd.DataFrame:
        """
        Executa uma consulta SQL nos dataframes carregados.
        
        Args:
            sql_query: Consulta SQL a ser executada
            
        Returns:
            DataFrame com os resultados da consulta
        """
        try:
            # Cria uma conexão DuckDB em memória
            import duckdb
            conn = duckdb.connect(database=':memory:')
            
            # Registra cada dataframe com seu nome
            for name, df_wrapper in self.dataframes.items():
                conn.register(name, df_wrapper.dataframe)
            
            # Executa a consulta
            result = conn.execute(sql_query).fetchdf()
            
            return result
        except Exception as e:
            logger.error(f"Erro ao executar consulta SQL: {str(e)}")
            raise
    
    def _generate_code_for_query(self, query: str) -> str:
        """
        Gera código Python/SQL para uma consulta em linguagem natural.
        
        Args:
            query: Consulta em linguagem natural
            
        Returns:
            Código Python/SQL gerado
        """
        # Em uma implementação real, isso usaria um LLM para gerar o código
        # Esta é uma implementação simulada para demonstração
        
        query = query.lower()
        
        if "conte" in query or "quantos" in query:
            # Código para consultas de contagem
            if "vendas" in query:
                return """
import pandas as pd

df = execute_sql_query("SELECT COUNT(*) as total FROM vendas")
total = df['total'].iloc[0]

result = {
    "type": "number",
    "value": total
}
"""
        elif "total" in query or "valor total" in query:
            # Código para somas
            if "vendas" in query:
                return """
import pandas as pd

df = execute_sql_query("SELECT SUM(valor) as total FROM vendas")
total = df['total'].iloc[0]

result = {
    "type": "number",
    "value": total
}
"""
        elif "mostre" in query and "primeiro" in query:
            # Código para exibir primeiras linhas
            if "vendas" in query:
                return """
import pandas as pd

# Consulta as primeiras linhas
df = execute_sql_query("SELECT * FROM vendas LIMIT 5")

# Define o resultado
result = {
    "type": "dataframe",
    "value": df
}
"""
        elif "gráfico" in query:
            # Código para visualizações
            if "barras" in query and "vendas" in query:
                return """
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

# Consulta agregando vendas por cliente
df = execute_sql_query('''
    SELECT id_cliente, SUM(valor) as total_vendas
    FROM vendas
    GROUP BY id_cliente
    ORDER BY total_vendas DESC
''')

# Cria o gráfico
plt.figure(figsize=(10, 6))
plt.bar(df['id_cliente'].astype(str), df['total_vendas'])
plt.title('Total de vendas por cliente')
plt.xlabel('Cliente')
plt.ylabel('Total vendas')
plt.xticks(rotation=45)
plt.tight_layout()

# Salva o gráfico em base64
buffer = io.BytesIO()
plt.savefig(buffer, format='png')
buffer.seek(0)
img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
img_data = f"data:image/png;base64,{img_str}"

# Define o resultado
result = {
    "type": "plot",
    "value": img_data
}
"""

    def execute_query(self, query: str) -> 'BaseResponse':
        """
        Executa uma consulta em linguagem natural.
        
        Args:
            query: Consulta em linguagem natural
            
        Returns:
            Resposta formatada (BaseResponse ou subclasse)
        """
        from core.response.dataframe import DataFrameResponse
        from core.response.string import StringResponse
        from core.response.number import NumberResponse
        from core.response.chart import ChartResponse
        from core.response.error import ErrorResponse
        
        logger.info(f"Processando consulta em linguagem natural: {query}")
        
        # Essa versão simplificada é apenas para demonstração
        # Em um sistema real, isso usaria LLM para gerar e executar código
        
        try:
            # Processamento simulado das consultas
            lower_query = query.lower()
            
            # Consultas que retornam DataFrames
            if "mostre" in lower_query and "primeiras" in lower_query and "linhas" in lower_query:
                df_name = next((name for name in self.dataframes.keys() 
                               if name in lower_query), list(self.dataframes.keys())[0])
                result = self.dataframes[df_name].dataframe.head(5)
                return DataFrameResponse(result)
                
            # Consultas de contagem
            elif "quantos" in lower_query or "conte" in lower_query:
                if "vendas" in lower_query:
                    count = len(self.dataframes.get("vendas", 
                                                  self.dataframes.get("sales_data")).dataframe)
                    return NumberResponse(count)
                elif "clientes" in lower_query:
                    count = len(self.dataframes.get("clientes", 
                                                  self.dataframes.get("customers")).dataframe)
                    return NumberResponse(count)
                    
            # Consultas de valor total/soma
            elif "total" in lower_query or "soma" in lower_query:
                if "vendas" in lower_query and "valor" in lower_query:
                    total = self.dataframes.get("vendas", 
                                              self.dataframes.get("sales_data")).dataframe["valor"].sum()
                    return NumberResponse(total)
                    
            # Consultas que geram visualizações
            elif any(kw in lower_query for kw in ["gráfico", "visualiza", "plot", "mostre", "crie"]):
                print(f"DEBUG: Generating visualization for query: {query}")  # Debug info
                import matplotlib.pyplot as plt
                import io
                import base64
                
                # Cria um gráfico simples
                plt.figure(figsize=(10, 6))
                
                # Gráfico de barras
                if "barras" in lower_query or ("total" in lower_query and "cliente" in lower_query) or "vendas por cliente" in lower_query:
                    # This debug line will help us track what's happening
                    print("DEBUG: Generating bar chart for client sales")
                    df = self.dataframes.get("vendas", self.dataframes.get("sales_data")).dataframe
                    client_totals = df.groupby("id_cliente")["valor"].sum()
                    client_totals.plot(kind="bar")
                    plt.title("Total de vendas por cliente")
                    plt.xlabel("ID do Cliente")
                    plt.ylabel("Total de Vendas")
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                # Histograma
                elif "histograma" in lower_query:
                    if "vendas" in lower_query:
                        df = self.dataframes.get("vendas", self.dataframes.get("sales_data")).dataframe
                        df["valor"].hist(bins=15)
                        plt.title("Histograma de valores de vendas")
                        plt.xlabel("Valor")
                        plt.ylabel("Frequência")
                        plt.grid(False)
                        plt.tight_layout()
                # Gráfico de linha
                elif "linha" in lower_query or "temporal" in lower_query or "evolução" in lower_query:
                    df = self.dataframes.get("vendas", self.dataframes.get("sales_data")).dataframe
                    # Certifica que temos uma coluna de data
                    if "data_venda" in df.columns:
                        date_col = "data_venda"
                    elif "data" in df.columns:
                        date_col = "data"
                    else:
                        # Usa o índice se não houver coluna de data
                        df = df.set_index(pd.date_range(start='2023-01-01', periods=len(df)))
                        date_col = df.index
                        
                    # Agrupa por data
                    if date_col != df.index:
                        df = df.sort_values(by=date_col)
                        time_series = df.groupby(date_col)["valor"].sum()
                        time_series.plot(kind="line")
                    else:
                        df["valor"].plot(kind="line")
                        
                    plt.title("Evolução de vendas ao longo do tempo")
                    plt.xlabel("Data")
                    plt.ylabel("Valor Total")
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                
                # Salva o gráfico em uma string base64
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png')
                buffer.seek(0)
                img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
                plt.close()
                
                return ChartResponse(f"data:image/png;base64,{img_str}")
                
            # Consulta padrão
            return StringResponse(f"Consulta processada: {query}")
                
        except Exception as e:
            logger.error(f"Erro ao processar consulta: {str(e)}")
            return ErrorResponse(f"Erro ao processar consulta: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Retorna estatísticas de uso do motor.
        
        Returns:
            Dict com estatísticas
        """
        return {
            "total_queries": 0,
            "successful_queries": 0,
            "loaded_dataframes": len(self.dataframes),
            "dataframe_names": list(self.dataframes.keys()),
            "total_rows": sum(len(df.dataframe) for name, df in self.dataframes.items())
        }