elif output_type == "plot":
            code += """
import matplotlib.pyplot as plt
import io
import base64

# Limita ao top 10 para melhor visualização
df_plot = df_result.head(10)

# Cria uma visualização do agrupamento
plt.figure(figsize=(12, 6))
plt.bar(df_plot[group_column].astype(str), df_plot[agg_column], color='royalblue')
plt.title(f'{agg_column.capitalize()} por {group_column}')
plt.xlabel(group_column)
plt.ylabel(agg_column.capitalize())
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()

# Salva a figura em memória e converte para base64
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
        else:  # string
            code += """
# Formata uma descrição dos dados agrupados
total_grupos = len(df_result)
top_grupo = df_result.iloc[0][group_column]
top_valor = df_result.iloc[0][agg_column]

# Define o resultado
result = {
    "type": "string",
    "value": f"Análise por {group_column}: {total_grupos} grupos encontrados. O grupo '{top_grupo}' tem o maior valor: {top_valor:,.2f}."
}
"""
        
        return code
    
    def _generate_visualization_code(self, entities: List[str], output_type: str) -> str:
        """Gera código para visualização de dados"""
        entity = entities[0] if entities else "vendas"
        
        # Identifica colunas apropriadas para agrupamento
        group_column = "id_cliente"
        if entity == "vendas_perdidas":
            group_column = "Motivo"
        elif entity == "clientes":
            group_column = "cidade"
        
        # Identifica coluna de valor apropriada
        value_column = "valor"
        if entity == "vendas_perdidas":
            value_column = "ImpactoFinanceiro"
        
        code = f"""
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

# Consulta para dados de visualização
df_result = execute_sql_query('''
    SELECT {group_column}, SUM({value_column}) as total
    FROM {entity}
    GROUP BY {group_column}
    ORDER BY total DESC
    LIMIT 10
''')

# Cria uma visualização dos dados
plt.figure(figsize=(12, 7))
plt.bar(df_result[group_column].astype(str), df_result['total'], color='royalblue')
plt.title(f'Total de {value_column} por {group_column}')
plt.xlabel(group_column)
plt.ylabel(f'Total de {value_column}')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()

# Salva a figura em memória e converte para base64
buffer = io.BytesIO()
plt.savefig(buffer, format='png')
buffer.seek(0)
img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
img_data = f"data:image/png;base64,{{img_str}}"

"""

        # Adiciona resultado de acordo com o tipo de saída
        if output_type == "plot":
            code += """
# Define o resultado
result = {
    "type": "plot",
    "value": img_data
}
"""
        elif output_type == "dataframe":
            code += """
# Define o resultado
result = {
    "type": "dataframe",
    "value": df_result
}
"""
        elif output_type == "number":
            code += """
# Obtém o total geral
total_geral = df_result['total'].sum()

# Define o resultado
result = {
    "type": "number",
    "value": total_geral
}
"""
        else:  # string
            code += """
# Formata uma descrição da visualização
total_grupos = len(df_result)
top_grupo = df_result.iloc[0][group_column]
top_valor = df_result.iloc[0]['total']

# Define o resultado
result = {
    "type": "string",
    "value": f"Visualização de dados com {total_grupos} grupos. O grupo '{top_grupo}' tem o maior valor: {top_valor:,.2f}."
}
"""
        
        return code
    
    def _generate_correction_code(self, prompt: str, output_type: str) -> str:
        """Gera código corrigido com base no prompt de correção"""
        # Verifica se é um erro de função execute_sql_query
        if "use execute_sql_query function" in prompt.lower():
            # Extrai o código original
            import re
            code_match = re.search(r"You generated the following Python code:\n(.*?)However", prompt, re.DOTALL)
            if code_match:
                original_code = code_match.group(1).strip()
                
                # Substitui pandas.read_sql ou similares por execute_sql_query
                corrected_code = re.sub(
                    r"(df|data|result)[\w\s]*=[\w\s]*pd\.read_sql\((.*?)\)",
                    r"\1 = execute_sql_query(\2)",
                    original_code
                )
                
                # Se não houver substituição, adiciona a função manualmente
                if corrected_code == original_code:
                    corrected_code = original_code.replace(
                        "import pandas as pd", 
                        "import pandas as pd\n\n# Usando execute_sql_query"
                    )
                    
                    # Substitui qualquer leitura direta de dataframe
                    data_vars = ["df", "data", "result", "dados"]
                    for var in data_vars:
                        if f"{var} = " in corrected_code:
                            corrected_code = corrected_code.replace(
                                f"{var} = ", 
                                f"{var} = execute_sql_query('''\n    SELECT * FROM vendas\n''') # "
                            )
                            break
                
                return corrected_code
        
        # Verifica se é um erro de tipo de resultado
        if "result type should be" in prompt.lower():
            # Extrai o código original
            import re
            code_match = re.search(r"You generated the following Python code:\n(.*?)However", prompt, re.DOTALL)
            if code_match:
                original_code = code_match.group(1).strip()
                
                # Extrai o tipo esperado
                type_match = re.search(r"result type should be: (\w+)", prompt, re.IGNORECASE)
                expected_type = type_match.group(1) if type_match else output_type
                
                # Procura pelo resultado atual
                result_match = re.search(r'result\s*=\s*{[^}]*"type":\s*"(\w+)"', original_code)
                current_type = result_match.group(1) if result_match else None
                
                if current_type and expected_type and current_type != expected_type:
                    # Substitui o tipo atual pelo esperado
                    corrected_code = re.sub(
                        r'("type":\s*)"(\w+)"',
                        f'\\1"{expected_type}"',
                        original_code
                    )
                    
                    # Ajusta o valor conforme o tipo esperado
                    if expected_type == "number":
                        corrected_code = re.sub(
                            r'("value":\s*)("[^"]*"|\'[^\']*\')',
                            r'\1len(df_result) if "df_result" in locals() else 0',
                            corrected_code
                        )
                    elif expected_type == "string":
                        corrected_code = re.sub(
                            r'("value":\s*)[^,}]+',
                            r'\1f"Resultado da análise: {len(df_result) if "df_result" in locals() else 0} registros encontrados"',
                            corrected_code
                        )
                    elif expected_type == "dataframe":
                        corrected_code = re.sub(
                            r'("value":\s*)[^,}]+',
                            r'\1df_result if "df_result" in locals() else pd.DataFrame()',
                            corrected_code
                        )
                    
                    return corrected_code
        
        # Correção genérica
        return """
import pandas as pd

# Consulta corrigida
df_result = execute_sql_query('''
    SELECT * FROM vendas
    LIMIT 10
''')

# Define o resultado com o tipo correto
result = {
    "type": "string",
    "value": f"Consulta retornou {len(df_result)} registros."
}
"""
    
    def _validate_generated_code(self, code: str) -> bool:
        """
        Valida se o código gerado atende aos requisitos básicos.
        
        Args:
            code: Código gerado
            
        Returns:
            bool: True se o código for válido, False caso contrário
        """
        # Verifica se o código contém a função execute_sql_query
        if "execute_sql_query" not in code:
            logger.warning("Código gerado não utiliza a função execute_sql_query")
            return False
        
        # Verifica se o código define a variável result
        if "result =" not in code:
            logger.warning("Código gerado não define a variável result")
            return False
        
        # Verifica se o resultado tem o formato esperado
        if "\"type\":" not in code or "\"value\":" not in code:
            logger.warning("Código gerado não define o resultado no formato esperado")
            return False
        
        return True
    
    def execute_query(self, query: str) -> BaseResponse:
        """
        Executa uma consulta em linguagem natural.
        
        Este é o método principal que integra todo o processo:
        1. Gera código Python/SQL para a consulta
        2. Executa o código
        3. Processa o resultado
        4. Lida com erros automaticamente
        
        Args:
            query: Consulta em linguagem natural
            
        Returns:
            BaseResponse: Resposta processada
        """
        try:
            # Gera o código
            generated_code = self.generate_code(query)
            logger.info(f"Código gerado:\n{generated_code}")
            
            # Prepara o contexto para execução
            execution_context = {
                'execute_sql_query': self.execute_sql_query,
                'pd': pd,
                'plt': plt,
                'np': __import__('numpy')
            }
            
            # Executa o código com o executor avançado
            execution_result = self.code_executor.execute_code(
                code=generated_code,
                context=execution_context
            )
            
            if not execution_result["success"]:
                # Tenta corrigir o código e executar novamente
                logger.warning(f"Erro na execução do código: {execution_result['error']}")
                corrected_code = self._handle_execution_error(generated_code, execution_result["error"])
                logger.info(f"Código corrigido:\n{corrected_code}")
                
                # Executa o código corrigido
                execution_result = self.code_executor.execute_code(
                    code=corrected_code,
                    context=execution_context
                )
                
                if not execution_result["success"]:
                    raise RuntimeError(f"Erro persistente: {execution_result['error']}")
            
            # Processa o resultado
            result = execution_result["result"]
            
            if not result or not isinstance(result, dict) or "type" not in result or "value" not in result:
                raise ValueError("Resultado inválido. Deve conter 'type' e 'value'.")
            
            # Parse o resultado para o tipo adequado
            response = self.response_parser.parse(result, last_code_executed=generated_code)
            
            # Incrementa contador de sucesso
            self.successful_queries += 1
            
            return response
            
        except Exception as e:
            # Registra e trata o erro
            logger.error(f"Erro ao processar consulta: {str(e)}")
            self.error_queries += 1
            
            return ErrorResponse(
                f"Erro ao processar sua consulta: {str(e)}",
                last_code_executed=generated_code if 'generated_code' in locals() else None,
                error=str(e)
            )
    
    def _handle_execution_error(self, code: str, error: str) -> str:
        """
        Trata erros de execução tentando corrigi-los automaticamente.
        
        Args:
            code: Código original
            error: Mensagem de erro
            
        Returns:
            str: Código corrigido
        """
        # Criamos um prompt para o modelo de IA para corrigir o erro
        correction_prompt = f"""
Você gerou o seguinte código Python:

{code}

No entanto, ocorreu o seguinte erro:

{error}

Por favor, corrija o código Python acima e retorne o novo código corrigido.
"""
        
        # Gera o código corrigido
        corrected_code = self._call_language_model(correction_prompt)
        
        return corrected_code
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Retorna estatísticas de uso do motor.
        
        Returns:
            Dict[str, Any]: Estatísticas de uso
        """
        return {
            "total_queries": self.query_count,
            "successful_queries": self.successful_queries,
            "error_queries": self.error_queries,
            "success_rate": (self.successful_queries / self.query_count) * 100 if self.query_count > 0 else 0,
            "loaded_dataframes": list(self.dataframes.keys()),
            "available_connectors": list(self.connectors.keys())
        }


# Interface simples de linha de comando para teste
def main():
    """Função principal para execução como script"""
    print("\n===== Motor de Consulta em Linguagem Natural =====\n")
    
    # Inicializa o motor
    engine = NaturalLanguageQueryEngine()
    
    print(f"Fontes de dados disponíveis: {', '.join(engine.dataframes.keys())}")
    print("\nDigite suas consultas em linguagem natural. Digite 'sair' para encerrar.\n")
    
    while True:
        # Solicita a consulta
        query = input("\nConsulta > ")
        
        if query.lower() in ['sair', 'exit', 'quit']:
            break
        
        if not query.strip():
            continue
        
        try:
            # Processa a consulta
            print("\nProcessando consulta...\n")
            response = engine.execute_query(query)
            
            # Exibe a resposta formatada
            print("\n----- Resposta -----\n")
            
            if response.type == "dataframe":
                print(response.value.head(10))
                print(f"\n[{len(response.value)} linhas no total]")
            elif response.type == "plot":
                print("[Visualização gerada - Em um ambiente gráfico, isso seria exibido como imagem]")
            else:
                print(response.value)
                
            print("\n-------------------\n")
            
        except Exception as e:
            print(f"\nErro ao processar consulta: {str(e)}\n")
    
    # Exibe estatísticas
    stats = engine.get_stats()
    print("\n===== Estatísticas de Uso =====\n")
    print(f"Total de consultas: {stats['total_queries']}")
    print(f"Consultas bem-sucedidas: {stats['successful_queries']}")
    print(f"Taxa de sucesso: {stats['success_rate']:.1f}%")
    
    print("\nObrigado por usar o Motor de Consulta em Linguagem Natural!")


if __name__ == "__main__":
    main()

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
    Motor principal para processamento de consultas em linguagem natural sobre dados estruturados.

    Esta classe integra todos os componentes do sistema para fornecer uma interface unificada:
    - Carregamento e gerenciamento de dados usando conectores
    - Processamento de consultas em linguagem natural
    - Geração de código Python/SQL
    - Execução segura de código
    - Formatação de respostas
    - Tratamento de erros
    """
    
    def __init__(
        self, 
        data_config_path: Optional[str] = None,
        metadata_config_path: Optional[str] = None,
        output_types: Optional[List[str]] = None,
        model_config: Optional[Dict[str, Any]] = None,
        base_data_path: Optional[str] = None
    ):
        """
        Inicializa o motor de consulta em linguagem natural.
        
        Args:
            data_config_path: Caminho para o arquivo de configuração de dados
            metadata_config_path: Caminho para o arquivo de configuração de metadados
            output_types: Tipos de saída permitidos (string, number, dataframe, plot)
            model_config: Configuração para o modelo de IA
            base_data_path: Caminho base para os arquivos de dados
        """
        # Configurações básicas
        self.data_config_path = data_config_path or os.path.join(os.getcwd(), "datasources.json")
        self.metadata_config_path = metadata_config_path or os.path.join(os.getcwd(), "metadata.json")
        self.base_data_path = base_data_path or os.path.join(os.getcwd(), "dados")
        self.output_types = output_types or ["string", "number", "dataframe", "plot"]
        self.model_config = model_config or {}
        
        # Diretório de saída para gráficos e resultados
        self.output_dir = os.path.join(os.getcwd(), "output")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Inicialização de componentes
        self._init_components()
        
        # Carregamento de dados e metadados
        self._load_data_connectors()
        
        # Inicialização do executor de código
        self.code_executor = AdvancedDynamicCodeExecutor(
            allowed_imports=[
                'numpy', 'pandas', 'matplotlib', 'scipy', 'sympy', 
                'statistics', 're', 'math', 'random', 'datetime', 
                'json', 'itertools', 'collections', 'io', 'base64'
            ],
            timeout=60,
            max_output_size=10 * 1024 * 1024  # 10 MB
        )
        
        # Inicialização do parser de resposta
        self.response_parser = ResponseParser()
        
        # Estatísticas de uso
        self.query_count = 0
        self.successful_queries = 0
        self.error_queries = 0
        
        logger.info("Motor de consulta em linguagem natural inicializado")
    
    def _init_components(self):
        """Inicializa os componentes internos"""
        # Registro de metadados
        self.metadata_registry = MetadataRegistry()
        
        # Conectores de dados
        self.connectors: Dict[str, DataConnector] = {}
        
        # Fachada para construtores de consultas
        self.query_builder = QueryBuilderFacade(base_path=self.base_data_path)
        
        # Analisador de datasets
        self.dataset_analyzer = DatasetAnalyzer()
        
        # Mapeamento de dataframes carregados
        self.dataframes: Dict[str, DataFrameWrapper] = {}
        
        # Estado do agente
        self.memory = AgentMemory("Assistente de análise de dados com SQL")
        self.config = AgentConfig(direct_sql=True)
        self.agent_state = AgentState(
            dfs=[],
            memory=self.memory,
            config=self.config
        )
    
    def _load_data_connectors(self):
        """Carrega conectores de dados a partir dos arquivos de configuração"""
        try:
            # Verifica se os arquivos de configuração existem
            if not os.path.exists(self.data_config_path):
                logger.warning(f"Arquivo de configuração de dados não encontrado: {self.data_config_path}")
                # Cria um arquivo de configuração padrão
                self._create_default_data_config()
            
            # Carrega metadados se disponíveis
            if os.path.exists(self.metadata_config_path):
                try:
                    self.metadata_registry.register_from_file(self.metadata_config_path)
                    logger.info(f"Metadados carregados do arquivo: {self.metadata_config_path}")
                except Exception as e:
                    logger.error(f"Erro ao carregar metadados: {str(e)}")
            
            # Carrega os conectores de dados
            with open(self.data_config_path, "r") as f:
                config_json = f.read()
            
            # Utiliza a factory para criar os conectores
            self.connectors = DataConnectorFactory.create_from_json(config_json)
            logger.info(f"Carregados {len(self.connectors)} conectores de dados")
            
            # Carrega os dataframes
            self._load_dataframes()
            
        except Exception as e:
            logger.error(f"Erro ao carregar conectores de dados: {str(e)}")
            # Cria um conector padrão para não quebrar o fluxo
            self._create_default_connector()
    
    def _create_default_data_config(self):
        """Cria um arquivo de configuração de dados padrão"""
        default_config = {
            "data_sources": [
                {
                    "id": "vendas",
                    "type": "csv",
                    "path": os.path.join(self.base_data_path, "vendas.csv"),
                    "delimiter": ",",
                    "encoding": "utf-8"
                },
                {
                    "id": "clientes",
                    "type": "csv",
                    "path": os.path.join(self.base_data_path, "clientes.csv"),
                    "delimiter": ",",
                    "encoding": "utf-8"
                },
                {
                    "id": "vendas_perdidas",
                    "type": "csv",
                    "path": os.path.join(self.base_data_path, "vendas_perdidas.csv"),
                    "delimiter": ",",
                    "encoding": "utf-8"
                }
            ]
        }
        
        with open(self.data_config_path, "w") as f:
            json.dump(default_config, f, indent=2)
        
        logger.info(f"Criado arquivo de configuração padrão: {self.data_config_path}")
    
    def _create_default_connector(self):
        """Cria um conector padrão para garantir operação mínima"""
        config = DataSourceConfig(
            source_id="vendas",
            source_type="csv",
            path=os.path.join(self.base_data_path, "vendas.csv")
        )
        self.connectors = {"vendas": DataConnectorFactory.create_connector(config)}
        logger.info("Criado conector padrão para garantir operação mínima")
    
    def _load_dataframes(self):
        """Carrega os dataframes a partir dos conectores"""
        self.dataframes = {}
        
        for source_id, connector in self.connectors.items():
            try:
                # Conecta ao conector
                connector.connect()
                
                # Lê os dados
                df = connector.read_data()
                
                # Cria um wrapper para o dataframe
                wrapper = DataFrameWrapper(df, source_id)
                
                # Adiciona ao dicionário de dataframes
                self.dataframes[source_id] = wrapper
                
                # Fecha a conexão
                connector.close()
                
                logger.info(f"Dataframe '{source_id}' carregado com {len(df)} registros")
                
            except Exception as e:
                logger.error(f"Erro ao carregar dataframe '{source_id}': {str(e)}")
    
    def update_agent_state(self, query: str):
        """
        Atualiza o estado do agente com base na consulta.
        
        Args:
            query: Consulta em linguagem natural
        """
        # Atualiza a memória do agente
        self.memory.add_message(query)
        
        # Define os dataframes disponíveis
        self.agent_state.dfs = list(self.dataframes.values())
        
        # Define o tipo de saída esperado
        output_type = self._infer_output_type(query)
        self.agent_state.output_type = output_type
        
        # Registra a consulta
        self.query_count += 1
        
        logger.info(f"Estado do agente atualizado para consulta: '{query}'")
        logger.info(f"Tipo de saída inferido: {output_type}")
    
    def _infer_output_type(self, query: str) -> Optional[str]:
        """
        Infere o tipo de saída esperado com base na consulta.
        
        Args:
            query: Consulta em linguagem natural
            
        Returns:
            str: Tipo de saída esperado ou None
        """
        query_lower = query.lower()
        
        # Palavras-chave para cada tipo de saída
        viz_keywords = ["mostre", "gráfico", "visualização", "visualize", "plote", "plot", "chart", "figura", "exibe"]
        table_keywords = ["tabela", "lista", "listar", "dataframe", "dados"]
        number_keywords = ["quantos", "quanto", "média", "mediana", "total", "soma", "máximo", "mínimo", "count", "calcule"]
        
        # Verifica o tipo de saída esperado
        if any(keyword in query_lower for keyword in viz_keywords):
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
            
            # Fecha a conexão
            conn.close()
            
            return result
            
        except Exception as e:
            logger.error(f"Erro ao executar consulta SQL: {str(e)}")
            return pd.DataFrame()  # Retorna um dataframe vazio em caso de erro
    
    def generate_code(self, query: str) -> str:
        """
        Gera código Python/SQL para responder à consulta.
        
        Args:
            query: Consulta em linguagem natural
            
        Returns:
            str: Código Python gerado
        """
        # Atualiza o estado do agente
        self.update_agent_state(query)
        
        # Gera o prompt para o modelo de IA
        prompt = get_chat_prompt_for_sql(self.agent_state)
        prompt_text = prompt.to_string()
        
        logger.info(f"Prompt gerado para o modelo de IA:\n{prompt_text[:300]}...")
        
        # Aqui seria a chamada para o modelo de IA real
        # Por enquanto, usaremos uma implementação de exemplo
        generated_code = self._call_language_model(prompt_text)
        
        # Valida o código gerado
        if not self._validate_generated_code(generated_code):
            logger.warning("Código gerado inválido. Tentando novamente com prompt modificado.")
            # Tenta gerar novamente com um prompt modificado
            modified_prompt = prompt_text + "\n\nPor favor, certifique-se de usar a função execute_sql_query para consultar os dados e definir a variável result com o tipo correto."
            generated_code = self._call_language_model(modified_prompt)
        
        # Atualiza o estado com o código gerado
        self.agent_state.set("last_code_generated", generated_code)
        
        return generated_code
    
    def _call_language_model(self, prompt: str) -> str:
        """
        Simulação de chamada para o modelo de linguagem.
        
        Em uma implementação real, esta função chamaria uma API de modelo de linguagem.
        
        Args:
            prompt: Prompt para o modelo de linguagem
            
        Returns:
            str: Código Python gerado
        """
        # Esta é uma implementação de exemplo que gera um código básico
        # Em um sistema real, isso seria substituído pela chamada ao modelo de IA
        
        # Verifica se é um prompt de correção
        is_correction = "Fix the python code" in prompt or "resulted in the following error" in prompt
        
        # Verifica o tipo de saída esperado
        output_type = "string"  # Padrão
        if "{ \"type\": \"number\"" in prompt:
            output_type = "number"
        elif "{ \"type\": \"dataframe\"" in prompt:
            output_type = "dataframe"
        elif "{ \"type\": \"plot\"" in prompt:
            output_type = "plot"
        
        # Palavras-chave na consulta para determinar o tipo de análise
        has_group_by = "group by" in prompt.lower() or "agrupados por" in prompt.lower()
        has_count = "count" in prompt.lower() or "contagem" in prompt.lower() or "quantos" in prompt.lower()
        has_sum = "sum" in prompt.lower() or "soma" in prompt.lower() or "total" in prompt.lower()
        has_avg = "average" in prompt.lower() or "média" in prompt.lower() or "media" in prompt.lower()
        has_plot = "plot" in prompt.lower() or "gráfico" in prompt.lower() or "visualize" in prompt.lower()
        
        # Detecta quais entidades estão sendo mencionadas
        entities = []
        for entity in ["vendas", "clientes", "produtos", "vendas_perdidas"]:
            if entity in prompt.lower():
                entities.append(entity)
        
        # Se não foi mencionada nenhuma entidade, usa a primeira disponível
        if not entities and self.dataframes:
            entities = [next(iter(self.dataframes.keys()))]
        
        # Gera código com base nos parâmetros detectados
        if is_correction:
            return self._generate_correction_code(prompt, output_type)
        elif has_plot:
            return self._generate_visualization_code(entities, output_type)
        elif has_group_by:
            return self._generate_group_by_code(entities, has_count, has_sum, has_avg, output_type)
        elif has_count:
            return self._generate_count_code(entities, output_type)
        elif has_sum:
            return self._generate_sum_code(entities, output_type)
        elif has_avg:
            return self._generate_average_code(entities, output_type)
        else:
            return self._generate_basic_query_code(entities, output_type)
    
    def _generate_basic_query_code(self, entities: List[str], output_type: str) -> str:
        """Gera código para uma consulta básica"""
        entity = entities[0] if entities else "vendas"
        
        code = f"""
import pandas as pd

# Consulta básica para obter os dados
df_result = execute_sql_query('''
    SELECT * FROM {entity}
    LIMIT 10
''')

# Formata a resposta de acordo com o tipo de saída esperado
"""

        # Adiciona formatação específica por tipo de saída
        if output_type == "number":
            code += """
# Quantidade de registros
total = len(df_result)

# Define o resultado
result = {
    "type": "number",
    "value": total
}
"""
        elif output_type == "dataframe":
            code += """
# Define o resultado
result = {
    "type": "dataframe",
    "value": df_result
}
"""
# Define o resultado
result = {
    "type": "dataframe",
    "value": df_result
}
"""
        elif output_type == "plot":
            code += """
import matplotlib.pyplot as plt
import io
import base64

# Cria uma visualização básica
plt.figure(figsize=(10, 6))
plt.bar(range(len(df_result)), df_result.iloc[:, 1].values)
plt.title('Visualização Básica dos Dados')
plt.xlabel('Índice')
plt.ylabel('Valor')
plt.tight_layout()

# Salva a figura em memória e converte para base64
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
        else:  # string
            code += """
# Formata uma descrição dos dados
description = f"Os dados contêm {len(df_result)} registros com as seguintes colunas: {', '.join(df_result.columns)}"

# Define o resultado
result = {
    "type": "string",
    "value": description
}
"""
        
        return code
    
    def _generate_count_code(self, entities: List[str], output_type: str) -> str:
        """Gera código para uma contagem de registros"""
        entity = entities[0] if entities else "vendas"
        
        code = f"""
import pandas as pd

# Consulta para contar registros
df_result = execute_sql_query('''
    SELECT COUNT(*) as total_registros FROM {entity}
''')

# Extrai o valor total
total = df_result['total_registros'].iloc[0]

"""

        # Adiciona formatação específica por tipo de saída
        if output_type == "number":
            code += """
# Define o resultado
result = {
    "type": "number",
    "value": total
}
"""
        elif output_type == "dataframe":
            code += """
# Define o resultado
result = {
    "type": "dataframe",
    "value": df_result
}
"""
        elif output_type == "plot":
            code += """
import matplotlib.pyplot as plt
import io
import base64

# Cria uma visualização de contagem
plt.figure(figsize=(8, 6))
plt.bar(['Total de Registros'], [total], color='skyblue')
plt.title('Contagem Total de Registros')
plt.ylabel('Quantidade')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()

# Salva a figura em memória e converte para base64
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
        else:  # string
            code += """
# Define o resultado
result = {
    "type": "string",
    "value": f"Total de registros: {total:,}"
}
"""
        
        return code
    
    def _generate_sum_code(self, entities: List[str], output_type: str) -> str:
        """Gera código para soma de valores"""
        entity = entities[0] if entities else "vendas"
        
        # Tenta detectar a coluna de valor apropriada para soma
        value_column = "valor"
        if entity == "vendas_perdidas":
            value_column = "ImpactoFinanceiro"
        
        code = f"""
import pandas as pd

# Consulta para calcular o total
df_result = execute_sql_query('''
    SELECT SUM({value_column}) as valor_total FROM {entity}
''')

# Extrai o valor total
total = df_result['valor_total'].iloc[0]

"""

        # Adiciona formatação específica por tipo de saída
        if output_type == "number":
            code += """
# Define o resultado
result = {
    "type": "number",
    "value": total
}
"""
        elif output_type == "dataframe":
            code += """
# Define o resultado
result = {
    "type": "dataframe",
    "value": df_result
}
"""
        elif output_type == "plot":
            code += """
import matplotlib.pyplot as plt
import io
import base64

# Cria uma visualização do total
plt.figure(figsize=(8, 6))
plt.bar(['Valor Total'], [total], color='green')
plt.title('Soma Total')
plt.ylabel('Valor')
plt.grid(axis='y', alpha=0.3)
formatter = plt.FuncFormatter(lambda x, pos: f'R$ {x:,.2f}')
plt.gca().yaxis.set_major_formatter(formatter)
plt.tight_layout()

# Salva a figura em memória e converte para base64
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
        else:  # string
            code += """
# Define o resultado
result = {
    "type": "string",
    "value": f"Valor total: R$ {total:,.2f}"
}
"""
        
        return code
    
    def _generate_average_code(self, entities: List[str], output_type: str) -> str:
        """Gera código para cálculo de média"""
        entity = entities[0] if entities else "vendas"
        
        # Tenta detectar a coluna de valor apropriada para média
        value_column = "valor"
        if entity == "vendas_perdidas":
            value_column = "ImpactoFinanceiro"
        
        code = f"""
import pandas as pd

# Consulta para calcular a média
df_result = execute_sql_query('''
    SELECT AVG({value_column}) as valor_medio FROM {entity}
''')

# Extrai o valor médio
media = df_result['valor_medio'].iloc[0]

"""

        # Adiciona formatação específica por tipo de saída
        if output_type == "number":
            code += """
# Define o resultado
result = {
    "type": "number",
    "value": media
}
"""
        elif output_type == "dataframe":
            code += """
# Define o resultado
result = {
    "type": "dataframe",
    "value": df_result
}
"""
        elif output_type == "plot":
            code += """
import matplotlib.pyplot as plt
import io
import base64

# Cria uma visualização da média
plt.figure(figsize=(8, 6))
plt.bar(['Valor Médio'], [media], color='orange')
plt.title('Valor Médio')
plt.ylabel('Valor')
plt.grid(axis='y', alpha=0.3)
formatter = plt.FuncFormatter(lambda x, pos: f'R$ {x:,.2f}')
plt.gca().yaxis.set_major_formatter(formatter)
plt.tight_layout()

# Salva a figura em memória e converte para base64
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
        else:  # string
            code += """
# Define o resultado
result = {
    "type": "string",
    "value": f"Valor médio: R$ {media:,.2f}"
}
"""
        
        return code
    
    def _generate_group_by_code(self, entities: List[str], has_count: bool, has_sum: bool, has_avg: bool, output_type: str) -> str:
        """Gera código para consultas com agrupamento"""
        entity = entities[0] if entities else "vendas"
        
        # Identifica colunas apropriadas para agrupamento
        group_column = "id_cliente"
        if entity == "vendas_perdidas":
            group_column = "Motivo"
        elif entity == "clientes":
            group_column = "cidade"
        
        # Identifica coluna de valor apropriada
        value_column = "valor"
        if entity == "vendas_perdidas":
            value_column = "ImpactoFinanceiro"
        
        # Identifica a agregação apropriada
        if has_count:
            agg_function = "COUNT(*)"
            agg_column = "contagem"
        elif has_sum:
            agg_function = f"SUM({value_column})"
            agg_column = "total"
        elif has_avg:
            agg_function = f"AVG({value_column})"
            agg_column = "media"
        else:
            agg_function = f"SUM({value_column})"
            agg_column = "total"
        
        code = f"""
import pandas as pd

# Consulta com agrupamento
df_result = execute_sql_query('''
    SELECT {group_column}, {agg_function} as {agg_column}
    FROM {entity}
    GROUP BY {group_column}
    ORDER BY {agg_column} DESC
''')

"""

        # Adiciona formatação específica por tipo de saída
        if output_type == "number":
            code += """
# Conta quantos grupos foram gerados
total_grupos = len(df_result)

# Define o resultado
result = {
    "type": "number",
    "value": total_grupos
}
"""
        elif output_type == "dataframe":
            code += """