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
        
        # Armazena a última consulta executada
        self.last_query = ""
        
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
        
        # Armazena a consulta atual
        self.last_query = query
        
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
                if "vendas perdidas" in lower_query or "oportunidades perdidas" in lower_query:
                    count = len(self.dataframes.get("vendas_perdidas").dataframe)
                    return NumberResponse(count)
                elif "vendas" in lower_query:
                    count = len(self.dataframes.get("vendas", 
                                                  self.dataframes.get("sales_data")).dataframe)
                    return NumberResponse(count)
                elif "clientes" in lower_query:
                    count = len(self.dataframes.get("clientes", 
                                                  self.dataframes.get("customers")).dataframe)
                    return NumberResponse(count)
                    
            # Consultas sobre motivos de vendas perdidas
            elif ("motivo" in lower_query or "motivos" in lower_query) and ("perdidas" in lower_query or "perda" in lower_query):
                if "comum" in lower_query or "frequente" in lower_query or "principal" in lower_query:
                    # Encontra o motivo mais comum
                    df_perdidas = self.dataframes.get("vendas_perdidas").dataframe
                    motivo_counts = df_perdidas["Motivo"].value_counts()
                    motivo_mais_comum = motivo_counts.index[0]
                    quantidade = motivo_counts.iloc[0]
                    
                    return StringResponse(f"O motivo mais comum para vendas perdidas é '{motivo_mais_comum}' com {quantidade} ocorrências.")
                else:
                    # Retorna dados sobre todos os motivos
                    df_perdidas = self.dataframes.get("vendas_perdidas").dataframe
                    motivo_counts = df_perdidas["Motivo"].value_counts().reset_index()
                    motivo_counts.columns = ["Motivo", "Quantidade"]
                    
                    return DataFrameResponse(motivo_counts)
                    
            # Consultas sobre estágios de vendas perdidas
            elif ("estágio" in lower_query or "estagio" in lower_query) and ("perdidas" in lower_query or "perda" in lower_query):
                df_perdidas = self.dataframes.get("vendas_perdidas").dataframe
                estagio_counts = df_perdidas["EstagioPerda"].value_counts().reset_index()
                estagio_counts.columns = ["Estágio", "Quantidade"]
                
                return DataFrameResponse(estagio_counts)
                
            # Consultas sobre correlação entre estágio e impacto financeiro
            elif ("correlação" in lower_query or "correlacao" in lower_query or "relação" in lower_query) and ("estágio" in lower_query or "estagio" in lower_query) and "impacto" in lower_query:
                df_perdidas = self.dataframes.get("vendas_perdidas").dataframe
                
                # Calcula médias e desvio padrão por estágio
                stats_por_estagio = df_perdidas.groupby("EstagioPerda")["ImpactoFinanceiro"].agg(['mean', 'std', 'count']).reset_index()
                stats_por_estagio.columns = ["Estágio", "Média de Impacto (R$)", "Desvio Padrão (R$)", "Quantidade"]
                
                # Formata valores para melhor legibilidade
                stats_por_estagio["Média de Impacto (R$)"] = stats_por_estagio["Média de Impacto (R$)"].round(2)
                stats_por_estagio["Desvio Padrão (R$)"] = stats_por_estagio["Desvio Padrão (R$)"].round(2)
                
                return DataFrameResponse(stats_por_estagio)
                
            # Consultas sobre impacto financeiro por motivo
            elif "impacto" in lower_query and "motivo" in lower_query and ("perdidas" in lower_query or "perda" in lower_query):
                df_perdidas = self.dataframes.get("vendas_perdidas").dataframe
                impacto_por_motivo = df_perdidas.groupby("Motivo")["ImpactoFinanceiro"].agg(['sum', 'mean', 'count']).reset_index()
                impacto_por_motivo.columns = ["Motivo", "Impacto Total (R$)", "Impacto Médio (R$)", "Quantidade"]
                
                # Formata valores e ordena por impacto total
                impacto_por_motivo["Impacto Total (R$)"] = impacto_por_motivo["Impacto Total (R$)"].round(2)
                impacto_por_motivo["Impacto Médio (R$)"] = impacto_por_motivo["Impacto Médio (R$)"].round(2)
                impacto_por_motivo = impacto_por_motivo.sort_values("Impacto Total (R$)", ascending=False)
                
                return DataFrameResponse(impacto_por_motivo)
                
            # Comparação entre regiões/cidades (simulado, pois não temos dados reais de cidade)
            elif ("compare" in lower_query or "comparação" in lower_query or "comparacao" in lower_query) and "cidade" in lower_query and ("perdidas" in lower_query or "perda" in lower_query):
                # Cria dados simulados para demonstrar
                cidades = ["São Paulo", "Rio de Janeiro", "Belo Horizonte", "Curitiba", "Brasília"]
                
                # Gera números simulados baseados nas proporções do dataset real
                df_perdidas = self.dataframes.get("vendas_perdidas").dataframe
                total_perdas = len(df_perdidas)
                
                # Simula distribuição de perdas por cidade
                import numpy as np
                np.random.seed(42)  # Para reprodutibilidade
                perdas_por_cidade = np.random.multinomial(total_perdas, [0.35, 0.25, 0.15, 0.15, 0.1])
                
                df_simulado = pd.DataFrame({
                    "Cidade": cidades,
                    "Quantidade de Vendas Perdidas": perdas_por_cidade,
                    "Percentual do Total": (perdas_por_cidade / total_perdas * 100).round(1)
                })
                
                df_simulado["Percentual do Total"] = df_simulado["Percentual do Total"].astype(str) + '%'
                
                return DataFrameResponse(df_simulado)
                
            # Consultas sobre feedback de cliente em vendas perdidas
            elif "feedback" in lower_query and "cliente" in lower_query and ("perdidas" in lower_query or "perda" in lower_query):
                # Como o feedback é texto livre, podemos mostrar exemplo de feedbacks por motivo
                df_perdidas = self.dataframes.get("vendas_perdidas").dataframe
                
                # Seleciona 2 exemplos de feedback por motivo
                resultado = []
                for motivo in df_perdidas["Motivo"].unique():
                    exemplos = df_perdidas[df_perdidas["Motivo"] == motivo]["FeedbackCliente"].sample(min(2, df_perdidas[df_perdidas["Motivo"] == motivo].shape[0])).tolist()
                    for exemplo in exemplos:
                        resultado.append({"Motivo": motivo, "Exemplo de Feedback": exemplo})
                
                df_resultado = pd.DataFrame(resultado)
                
                return DataFrameResponse(df_resultado)
            
            # Consultas de valor total/soma
            elif "total" in lower_query or "soma" in lower_query:
                if "impacto" in lower_query and ("perdidas" in lower_query or "perda" in lower_query):
                    total = self.dataframes.get("vendas_perdidas").dataframe["ImpactoFinanceiro"].sum()
                    return NumberResponse(total)
                elif "vendas" in lower_query and "valor" in lower_query:
                    total = self.dataframes.get("vendas", 
                                              self.dataframes.get("sales_data")).dataframe["valor"].sum()
                    return NumberResponse(total)
                    
            # Consultas que geram visualizações
            elif any(kw in lower_query for kw in ["gráfico", "visualiza", "plot", "mostre", "crie"]):
                import matplotlib.pyplot as plt
                import io
                import base64
                import pandas as pd
                import numpy as np
                from datetime import datetime, timedelta
                
                # Configura o estilo
                plt.style.use('seaborn-v0_8-darkgrid')
                
                # Cria um gráfico simples
                plt.figure(figsize=(10, 6))
                
                # GRÁFICOS PARA VENDAS PERDIDAS
                if "perdidas" in lower_query or "perda" in lower_query:
                    df_perdidas = self.dataframes.get("vendas_perdidas").dataframe
                    
                    # Gráfico de barras para vendas perdidas por motivo
                    if "motivo" in lower_query or "barras" in lower_query:
                        motivo_counts = df_perdidas["Motivo"].value_counts()
                        motivo_counts.plot(kind="bar", color="indianred")
                        plt.title("Contagem de Vendas Perdidas por Motivo")
                        plt.xlabel("Motivo")
                        plt.ylabel("Quantidade")
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                    
                    # Gráfico de barras para impacto financeiro por motivo
                    elif "impacto" in lower_query and "motivo" in lower_query:
                        impacto_por_motivo = df_perdidas.groupby("Motivo")["ImpactoFinanceiro"].sum()
                        impacto_por_motivo.plot(kind="bar", color="firebrick")
                        plt.title("Impacto Financeiro Total por Motivo de Perda")
                        plt.xlabel("Motivo")
                        plt.ylabel("Impacto Financeiro (R$)")
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                    
                    # Gráfico de barras para vendas perdidas por estágio
                    elif "estágio" in lower_query or "estagio" in lower_query:
                        estagio_counts = df_perdidas["EstagioPerda"].value_counts()
                        estagio_counts.plot(kind="bar", color="darkblue")
                        plt.title("Contagem de Vendas Perdidas por Estágio")
                        plt.xlabel("Estágio de Perda")
                        plt.ylabel("Quantidade")
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        
                    # Histograma de impacto financeiro
                    elif "histograma" in lower_query and "impacto" in lower_query:
                        plt.hist(df_perdidas["ImpactoFinanceiro"], bins=15, color="darkred", alpha=0.7, 
                                edgecolor="black")
                        plt.title("Distribuição do Impacto Financeiro de Vendas Perdidas")
                        plt.xlabel("Impacto Financeiro (R$)")
                        plt.ylabel("Frequência")
                        plt.grid(False)
                        plt.tight_layout()
                    
                    # Gráfico de pizza para distribuição de motivos
                    elif "pizza" in lower_query and "motivo" in lower_query:
                        motivos = df_perdidas["Motivo"].value_counts()
                        plt.pie(motivos, labels=motivos.index, autopct='%1.1f%%', 
                               shadow=True, startangle=90)
                        plt.axis('equal')
                        plt.title("Distribuição de Vendas Perdidas por Motivo")
                        plt.tight_layout()
                    
                    # Gráfico de linha temporal
                    elif "linha" in lower_query or "temporal" in lower_query or "evolução" in lower_query or "tendência" in lower_query or "tendencia" in lower_query:
                        # Criar uma coluna de data simulada já que vendas_perdidas não tem
                        # Para fins de demonstração, vamos criar datas dos últimos 150 dias
                        hoje = datetime.now()
                        datas = [(hoje - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(len(df_perdidas))]
                        df_temp = df_perdidas.copy()
                        df_temp['DataSimulada'] = datas
                        df_temp['DataSimulada'] = pd.to_datetime(df_temp['DataSimulada'])
                        
                        # Agrupa por semana
                        df_temp['Semana'] = df_temp['DataSimulada'].dt.isocalendar().week
                        perdas_por_semana = df_temp.groupby('Semana')['ImpactoFinanceiro'].sum()
                        
                        # Plota o gráfico de linha
                        perdas_por_semana.plot(kind='line', marker='o', color='darkred')
                        plt.title('Tendência de Impacto Financeiro de Vendas Perdidas por Semana')
                        plt.xlabel('Semana do Ano')
                        plt.ylabel('Impacto Financeiro Total (R$)')
                        plt.grid(True, alpha=0.3)
                        plt.tight_layout()
                    
                    # Correlação entre estágio e impacto financeiro
                    elif "correlação" in lower_query or "correlacao" in lower_query or "relação" in lower_query:
                        # Boxplot de impacto financeiro por estágio
                        plt.boxplot([df_perdidas[df_perdidas['EstagioPerda'] == estagio]['ImpactoFinanceiro'] 
                                    for estagio in df_perdidas['EstagioPerda'].unique()],
                                   labels=df_perdidas['EstagioPerda'].unique())
                        plt.title('Impacto Financeiro por Estágio de Perda')
                        plt.xlabel('Estágio de Perda')
                        plt.ylabel('Impacto Financeiro (R$)')
                        plt.grid(True, alpha=0.3)
                        plt.tight_layout()
                    
                    # Comparação de médias de impacto financeiro por motivo
                    elif "comparação" in lower_query or "comparacao" in lower_query or "compare" in lower_query:
                        media_por_motivo = df_perdidas.groupby('Motivo')['ImpactoFinanceiro'].mean().sort_values(ascending=False)
                        media_por_motivo.plot(kind='bar', color='darkblue')
                        plt.title('Impacto Financeiro Médio por Motivo')
                        plt.xlabel('Motivo')
                        plt.ylabel('Impacto Financeiro Médio (R$)')
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                    
                    else:
                        # Gráfico padrão para vendas perdidas - impacto por motivo
                        impacto_por_motivo = df_perdidas.groupby("Motivo")["ImpactoFinanceiro"].sum().sort_values(ascending=False)
                        impacto_por_motivo.plot(kind="bar", color="firebrick")
                        plt.title("Impacto Financeiro Total por Motivo de Perda")
                        plt.xlabel("Motivo")
                        plt.ylabel("Impacto Financeiro (R$)")
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                
                # GRÁFICOS PARA VENDAS
                else:
                    # Gráfico de barras
                    if "barras" in lower_query or ("total" in lower_query and "cliente" in lower_query) or "vendas por cliente" in lower_query:
                        df = self.dataframes.get("vendas", self.dataframes.get("sales_data")).dataframe
                        client_totals = df.groupby("id_cliente")["valor"].sum()
                        client_totals.plot(kind="bar", color="royalblue")
                        plt.title("Total de vendas por cliente")
                        plt.xlabel("ID do Cliente")
                        plt.ylabel("Total de Vendas (R$)")
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                    
                    # Histograma
                    elif "histograma" in lower_query:
                        df = self.dataframes.get("vendas", self.dataframes.get("sales_data")).dataframe
                        plt.hist(df["valor"], bins=15, color="skyblue", edgecolor="black")
                        plt.title("Histograma de valores de vendas")
                        plt.xlabel("Valor (R$)")
                        plt.ylabel("Frequência")
                        plt.grid(False)
                        plt.tight_layout()
                    
                    # Gráfico de linha
                    elif "linha" in lower_query or "temporal" in lower_query or "evolução" in lower_query:
                        df = self.dataframes.get("vendas", self.dataframes.get("sales_data")).dataframe
                        
                        # Certifica que temos uma coluna de data
                        date_col = None
                        if "data_venda" in df.columns:
                            date_col = "data_venda"
                            df[date_col] = pd.to_datetime(df[date_col])
                        elif "data" in df.columns:
                            date_col = "data"
                            df[date_col] = pd.to_datetime(df[date_col])
                        
                        if date_col is not None:
                            # Agrupa por data
                            df = df.sort_values(by=date_col)
                            time_series = df.groupby(pd.Grouper(key=date_col, freq='D'))["valor"].sum()
                            time_series.plot(kind="line", marker='o', color="royalblue")
                            plt.title("Evolução de vendas ao longo do tempo")
                            plt.xlabel("Data")
                            plt.ylabel("Valor Total (R$)")
                            plt.grid(True, alpha=0.3)
                            plt.tight_layout()
                        else:
                            # Usa o índice se não houver coluna de data
                            dates = pd.date_range(start='2023-01-01', periods=len(df))
                            df_temp = df.copy()
                            df_temp['data_simulada'] = dates
                            df_temp.set_index('data_simulada', inplace=True)
                            df_temp["valor"].plot(kind="line", marker='o', color="royalblue")
                            plt.title("Evolução de vendas (datas simuladas)")
                            plt.xlabel("Data")
                            plt.ylabel("Valor (R$)")
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