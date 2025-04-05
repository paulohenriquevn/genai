"""
Motor de análise principal que integra todos os componentes do sistema.
"""

import os
import logging
import pandas as pd
import time
from typing import Dict, List, Optional, Any, Union

# Importação dos componentes core
from core.code_executor import AdvancedDynamicCodeExecutor
from core.agent.state import AgentState, AgentMemory, AgentConfig
from core.prompts.generate_python_code_with_sql import GeneratePythonCodeWithSQLPrompt
from core.response.parser import ResponseParser
from core.response.base import BaseResponse
from core.response.dataframe import DataFrameResponse
from core.response.number import NumberResponse
from core.response.string import StringResponse
from core.response.chart import ChartResponse
from core.response.error import ErrorResponse
from core.user_query import UserQuery
from core.exceptions import QueryExecutionError

# Importação dos módulos refatorados
from core.engine.dataset import Dataset
from core.engine.sql_executor import SQLExecutor
from core.engine.alternative_flow import AlternativeFlow
from core.engine.feedback_manager import FeedbackManager

# Importação do módulo de integração com LLMs
from llm_integration import LLMIntegration, LLMQueryGenerator

# Configura o logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("analysis_engine.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("analysis_engine")


class AnalysisEngine:
    """
    Motor de análise que integra componentes core para processamento de consultas em linguagem natural.
    
    Esta classe implementa:
    - Carregamento e gerenciamento de datasets
    - Execução segura de código
    - Geração de prompts para LLM
    - Processamento de consultas em linguagem natural
    - Tratamento de respostas e conversão de formatos
    
    O design adota o padrão de fachada (Facade), orquestrando componentes especializados:
    - Dataset: Gerencia dados e metadados
    - SQLExecutor: Processa consultas SQL com adaptação de dialetos
    - AlternativeFlow: Provê fluxos alternativos para erros e sugestões
    - FeedbackManager: Gerencia feedback do usuário e otimização de consultas
    """
    
    def __init__(
        self,
        agent_description: str = "Assistente de Análise de Dados Inteligente",
        default_output_type: str = "dataframe",
        direct_sql: bool = False,
        timeout: int = 30,
        max_output_size: int = 1024 * 1024,  # 1 MB
        model_type: str = "mock",
        model_name: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        """
        Inicializa o motor de análise com configurações personalizadas.
        
        Args:
            agent_description: Descrição do agente para o LLM
            default_output_type: Tipo padrão de saída (dataframe, string, number, plot)
            direct_sql: Se True, executa SQL diretamente sem código Python
            timeout: Tempo limite para execução de código (segundos)
            max_output_size: Tamanho máximo da saída
            model_type: Tipo de modelo LLM (openai, anthropic, huggingface, local, mock)
            model_name: Nome específico do modelo LLM
            api_key: Chave de API para o modelo LLM
        """
        logger.info(f"Inicializando AnalysisEngine com output_type={default_output_type}, model_type={model_type}")
        
        # Inicialização dos componentes core
        self.code_executor = AdvancedDynamicCodeExecutor(
            timeout=timeout,
            max_output_size=max_output_size,
            allowed_imports=[
                "numpy", 
                "pandas", 
                "matplotlib", 
                "scipy", 
                "sympy", 
                "statistics", 
                "re", 
                "math", 
                "random", 
                "datetime", 
                "json", 
                "itertools", 
                "collections", 
                "io", 
                "base64"
            ]
        )
        
        # Configuração do agente
        agent_config = AgentConfig(direct_sql=direct_sql)
        agent_memory = AgentMemory(agent_description=agent_description)
        
        # Estado do agente (armazena datasets, memória e configurações)
        self.agent_state = AgentState(
            dfs=[],  # Será populado com objetos Dataset
            memory=agent_memory,
            config=agent_config,
            output_type=default_output_type
        )
        
        # Parser de respostas para validação e conversão
        self.response_parser = ResponseParser()
        
        # Armazena o último código gerado
        self.last_code_generated = ""
        
        # Dataset carregados (nome -> Dataset)
        self.datasets = {}
        
        # Inicializa componentes modulares
        self.feedback_manager = FeedbackManager()
        
        # Inicializa o módulo de fluxo alternativo (usado para tratamento de erros e sugestões)
        self.alternative_flow = None  # Será inicializado depois que tivermos datasets
        
        # Inicializa o executor SQL (será configurado após carregar datasets)
        self.sql_executor = None
        
        # Inicializa o gerador de código LLM
        try:
            # Cria a integração LLM
            llm_integration = LLMIntegration(
                model_type=model_type,
                model_name=model_name,
                api_key=api_key
            )
            
            # Cria o gerador de consultas
            self.query_generator = LLMQueryGenerator(llm_integration=llm_integration)
            logger.info(f"Gerador LLM inicializado com modelo {model_type}" + (f" ({model_name})" if model_name else ""))
        except Exception as e:
            # Em caso de erro, usa o modo mock
            logger.warning(f"Erro ao inicializar LLM: {str(e)}. Usando modo mock.")
            self.query_generator = LLMQueryGenerator()
    
    def load_data(
        self, 
        data: Union[pd.DataFrame, str], 
        name: str, 
        description: str = None,
        schema: Dict[str, str] = None
    ) -> None:
        """
        Carrega um DataFrame ou arquivo CSV no motor de análise.
        
        Args:
            data: DataFrame ou caminho para arquivo CSV
            name: Nome do dataset
            description: Descrição do dataset (opcional)
            schema: Dicionário de metadados das colunas (opcional)
        """
        try:
            # Carrega dados se for um caminho de arquivo
            if isinstance(data, str):
                logger.info(f"Carregando dados do arquivo: {data}")
                
                # Determina o tipo de arquivo pela extensão
                if data.endswith('.csv'):
                    df = pd.read_csv(data)
                elif data.endswith(('.xls', '.xlsx')):
                    df = pd.read_excel(data)
                elif data.endswith('.json'):
                    df = pd.read_json(data)
                elif data.endswith('.parquet'):
                    df = pd.read_parquet(data)
                else:
                    raise ValueError(f"Formato de arquivo não suportado: {data}")
            else:
                # Usa DataFrame diretamente
                df = data
            
            # Define descrição padrão se não fornecida
            if description is None:
                if isinstance(data, str):
                    description = f"Dataset carregado de {os.path.basename(data)}"
                else:
                    description = f"Dataset {name}"
            
            # Preprocessa o DataFrame para garantir compatibilidade com SQL
            df = self._preprocess_dataframe_for_sql(df, name)
            
            # Cria objeto Dataset
            dataset = Dataset(dataframe=df, name=name, description=description, schema=schema)
            
            # Armazena para uso futuro e adiciona ao estado do agente
            self.datasets[name] = dataset
            
            # Atualiza a lista no estado do agente com objetos Dataset
            self.agent_state.dfs.append(dataset)
            
            # Inicializa ou atualiza componentes dependentes
            # SQLExecutor precisa dos datasets para configuração
            self.sql_executor = SQLExecutor(self.datasets)
            
            # AlternativeFlow precisa dos datasets para gerar alternativas
            self.alternative_flow = AlternativeFlow(self.datasets, self.query_generator)
            
            logger.info(f"Dataset '{name}' carregado com {len(df)} linhas e {len(df.columns)} colunas")
        
        except Exception as e:
            logger.error(f"Erro ao carregar dados: {str(e)}")
            raise
    
    def _preprocess_dataframe_for_sql(self, df: pd.DataFrame, name: str) -> pd.DataFrame:
        """
        Prepara um DataFrame para uso em consultas SQL, garantindo compatibilidade com DuckDB.
        
        Args:
            df: DataFrame a ser preprocessado
            name: Nome do dataset (para logging)
            
        Returns:
            DataFrame preprocessado
        """
        try:
            # Cria cópia para evitar alterações no original
            processed_df = df.copy()
            
            # Converte colunas de data para o formato correto
            for col in processed_df.columns:
                # Verifica se a coluna parece ser uma data
                if processed_df[col].dtype == 'object':
                    try:
                        # Tenta usar expressão regular para identificar padrões de data
                        if processed_df[col].str.contains(r'\d{4}-\d{2}-\d{2}').any():
                            logger.info(f"Convertendo coluna {col} para datetime no dataset {name}")
                            processed_df[col] = pd.to_datetime(processed_df[col], errors='ignore')
                    except (AttributeError, TypeError):
                        # Ignora erros para colunas que não são strings ou com valores mistos
                        pass
            
            # Remove caracteres especiais dos nomes das colunas
            rename_map = {}
            for col in processed_df.columns:
                # Substitui espaços e caracteres especiais por underscores
                new_col = col
                if ' ' in col or any(c in col for c in '!@#$%^&*()-+?_=,<>/\\|{}[]'):
                    new_col = ''.join(c if c.isalnum() else '_' for c in col)
                    rename_map[col] = new_col
            
            # Renomeia colunas se necessário
            if rename_map:
                logger.info(f"Renomeando colunas com caracteres especiais no dataset {name}: {rename_map}")
                processed_df = processed_df.rename(columns=rename_map)
            
            # Verifica e corrige tipos de dados problemáticos
            for col in processed_df.columns:
                # Tenta converter colunas mistas para string quando apropriado
                if processed_df[col].dtype == 'object' and not pd.api.types.is_datetime64_any_dtype(processed_df[col]):
                    # Se a coluna tem valores mistos, converte para string
                    try:
                        unique_types = processed_df[col].apply(type).nunique()
                        if unique_types > 1:
                            logger.info(f"Convertendo coluna {col} com tipos mistos para string no dataset {name}")
                            processed_df[col] = processed_df[col].astype(str)
                    except:
                        # Em caso de erro, força para string
                        processed_df[col] = processed_df[col].astype(str)
            
            return processed_df
            
        except Exception as e:
            logger.warning(f"Erro durante preprocessamento do DataFrame {name}: {str(e)}")
            # Em caso de erro, retorna o DataFrame original
            return df
    
    def get_dataset(self, name: str) -> Optional[Dataset]:
        """
        Obtém um dataset pelo nome.
        
        Args:
            name: Nome do dataset
            
        Returns:
            Dataset ou None se não encontrado
        """
        return self.datasets.get(name)
    
    def list_datasets(self) -> List[str]:
        """
        Lista os nomes de todos os datasets carregados.
        
        Returns:
            Lista de nomes de datasets
        """
        return list(self.datasets.keys())
    
    def _generate_prompt(self, query: str) -> str:
        """
        Gera um prompt para o LLM com base na consulta do usuário.
        
        Args:
            query: Consulta em linguagem natural
            
        Returns:
            Prompt formatado para o LLM
        """
        # Adiciona a consulta à memória do agente
        self.agent_state.memory.add_message(query)
        
        # Cria o prompt usando a classe GeneratePythonCodeWithSQLPrompt
        prompt = GeneratePythonCodeWithSQLPrompt(
            context=self.agent_state,
            output_type=self.agent_state.output_type,
            last_code_generated=self.last_code_generated
        )
        
        # Renderiza o prompt completo
        rendered_prompt = prompt.render()
        logger.debug(f"Prompt gerado: {rendered_prompt[:500]}...")
        
        return rendered_prompt
    
    def process_query(self, query: str, retry_count: int = 0, max_retries: int = 2, feedback: str = None) -> BaseResponse:
        """
        Processa uma consulta em linguagem natural.
        
        Args:
            query: Consulta em linguagem natural
            retry_count: Contador de tentativas de rephrasing (uso interno)
            max_retries: Número máximo de tentativas antes de oferecer opções alternativas
            feedback: Feedback do usuário para melhorar a resposta (opcional)
            
        Returns:
            Objeto BaseResponse com o resultado da consulta
        """
        logger.info(f"Processando consulta: {query} (tentativa {retry_count+1})")
        
        # Se houver feedback do usuário, armazena para uso em futuras melhorias
        if feedback:
            self.feedback_manager.store_user_feedback(query, feedback)
            logger.info(f"Feedback recebido para a consulta: '{feedback}'")
        
        try:
            # Cria objeto UserQuery
            user_query = UserQuery(query)
            
            # Verifica se há datasets carregados
            if not self.datasets:
                return ErrorResponse("Nenhum dataset carregado. Carregue dados antes de executar consultas.")
            
            # Verifica menções a dados inexistentes usando o AlternativeFlow
            missing_entity_response = self.alternative_flow.check_missing_entities(query)
            if missing_entity_response:
                return missing_entity_response
            
            # Gera o prompt para o LLM
            prompt = self._generate_prompt(query)
            
            # Gera código Python usando o LLM
            start_time = time.time()
            generated_code = self.query_generator.generate_code(prompt)
            generation_time = time.time() - start_time
            
            logger.info(f"Código gerado em {generation_time:.2f}s")
            self.last_code_generated = generated_code
            
            # Contexto para execução inclui os datasets
            execution_context = {
                'query': query,
                'datasets': {name: ds.dataframe for name, ds in self.datasets.items()},
                'retry_count': retry_count
            }
            
            # Configuração da função execute_sql_query
            if self.sql_executor and len(self.datasets) > 0:
                execution_context['execute_sql_query'] = self.sql_executor.create_sql_executor()
            
            # Executa o código gerado
            execution_result = self.code_executor.execute_code(
                generated_code,
                context=execution_context,
                output_type=self.agent_state.output_type
            )
            
            # Verifica se a execução foi bem-sucedida
            if not execution_result["success"]:
                error_msg = execution_result["error"]
                logger.error(f"Erro na execução de código: {error_msg}")
                
                # Verifica se o erro menciona tabelas inexistentes
                if "tabela" in error_msg.lower() and ("não encontrada" in error_msg.lower() or "não existe" in error_msg.lower()):
                    return self.alternative_flow.handle_missing_table_error(error_msg)
                
                # Tenta corrigir o erro (opcional)
                if "correction_attempt" not in execution_context:
                    correction_result = self.alternative_flow.attempt_error_correction(
                        query,
                        generated_code,
                        error_msg,
                        execution_context,
                        self.code_executor,
                        self.response_parser,
                        self.agent_state.output_type
                    )
                    
                    # Se a correção também falhou e ainda não esgotamos as tentativas
                    if correction_result.type == "error" and retry_count < max_retries:
                        # Tenta reformular a consulta
                        rephrased_query = self.alternative_flow.rephrase_query(query, error_msg)
                        logger.info(f"Consulta reformulada: {rephrased_query}")
                        
                        # Reinicia o processamento com a consulta reformulada
                        return self.process_query(rephrased_query, retry_count + 1, max_retries)
                    
                    # Se tentamos correção e ainda não funcionou, mas foi o último retry
                    if correction_result.type == "error" and retry_count >= max_retries:
                        # Oferece opções predefinidas
                        return self.alternative_flow.offer_predefined_options(query, error_msg)
                    
                    return correction_result
                
                # Se chegou aqui, é uma falha após todas as tentativas
                if retry_count >= max_retries:
                    return self.alternative_flow.offer_predefined_options(query, error_msg)
                
                return ErrorResponse(f"Erro ao processar consulta: {error_msg}")
            
            # Obtém o resultado da execução
            result = execution_result["result"]
            
            # Valida e processa a resposta
            try:
                # Formata o resultado para o formato esperado pelo parser
                formatted_result = self._format_result_for_parser(result)
                
                # Parse a resposta para o tipo apropriado
                response = self.response_parser.parse(
                    formatted_result, 
                    self.last_code_generated
                )
                
                # Armazena a consulta bem-sucedida para uso futuro
                self.feedback_manager.store_successful_query(query, self.last_code_generated)
                
                logger.info(f"Consulta processada com sucesso. Tipo de resposta: {response.type}")
                return response
                
            except Exception as e:
                logger.error(f"Erro ao processar resposta: {str(e)}")
                
                # Se ainda temos tentativas disponíveis
                if retry_count < max_retries:
                    # Tenta reformular a consulta
                    rephrased_query = self.alternative_flow.rephrase_query(query, str(e))
                    logger.info(f"Consulta reformulada após erro de processamento: {rephrased_query}")
                    
                    # Reinicia o processamento com a consulta reformulada
                    return self.process_query(rephrased_query, retry_count + 1, max_retries)
                
                return ErrorResponse(f"Erro no processamento da resposta: {str(e)}")
        
        except Exception as e:
            logger.error(f"Erro ao processar consulta: {str(e)}")
            
            # Se ainda temos tentativas disponíveis
            if retry_count < max_retries:
                # Tenta reformular a consulta
                rephrased_query = self.alternative_flow.rephrase_query(query, str(e))
                logger.info(f"Consulta reformulada após exceção: {rephrased_query}")
                
                # Reinicia o processamento com a consulta reformulada
                return self.process_query(rephrased_query, retry_count + 1, max_retries)
            
            return ErrorResponse(f"Erro ao processar consulta: {str(e)}")
    
    def process_query_with_feedback(self, query: str, feedback: str = None) -> BaseResponse:
        """
        Processa uma consulta e inclui feedback do usuário quando disponível.
        
        Args:
            query: Consulta em linguagem natural
            feedback: Feedback opcional do usuário sobre consultas anteriores
            
        Returns:
            Objeto BaseResponse com o resultado da consulta
        """
        return self.process_query(query, feedback=feedback)
    
    def _format_result_for_parser(self, result: Any) -> Dict[str, Any]:
        """
        Formata o resultado da execução para o formato esperado pelo parser.
        
        Args:
            result: Resultado da execução
            
        Returns:
            Dicionário com 'type' e 'value'
        """
        # Se já estiver no formato esperado
        if isinstance(result, dict) and "type" in result and "value" in result:
            # Verifica se o tipo é 'plot' e o valor não é uma string de caminho válido
            if result["type"] == "plot":
                value = result["value"]
                if not isinstance(value, str) or (not value.endswith(('.png', '.jpg', '.svg', '.pdf')) and "data:image" not in value):
                    # Tenta salvar a imagem se for uma figura matplotlib
                    try:
                        import matplotlib.pyplot as plt
                        if isinstance(value, plt.Figure):
                            filename = f"plot_{int(time.time())}.png"
                            value.savefig(filename)
                            result["value"] = filename
                            logger.info(f"Figura matplotlib salva automaticamente como {filename}")
                        else:
                            # Fallback para string se não for um caminho ou figura válida
                            logger.warning(f"Valor inválido para tipo 'plot'. Convertendo para string.")
                            return {"type": "string", "value": "Não foi possível gerar uma visualização válida. Valor não é um caminho para imagem ou figura."}
                    except Exception as e:
                        logger.error(f"Erro ao processar visualização: {str(e)}")
                        return {"type": "string", "value": f"Erro ao processar visualização: {str(e)}"}
            
            return result
        
        # Infere o tipo com base no valor
        if isinstance(result, pd.DataFrame):
            return {"type": "dataframe", "value": result}
        elif isinstance(result, (int, float)):
            return {"type": "number", "value": result}
        elif isinstance(result, str):
            # Verifica se parece ser um caminho para um plot
            if result.endswith(('.png', '.jpg', '.svg', '.pdf')) or "data:image" in result:
                return {"type": "plot", "value": result}
            else:
                return {"type": "string", "value": result}
        else:
            # Verifica se é uma figura matplotlib
            try:
                import matplotlib.pyplot as plt
                if hasattr(result, 'savefig') or isinstance(result, plt.Figure):
                    filename = f"plot_{int(time.time())}.png"
                    plt.savefig(filename)
                    plt.close()
                    return {"type": "plot", "value": filename}
            except:
                pass
                
            # Tentativa genérica para outros tipos
            return {"type": "string", "value": str(result)}
    
    def execute_direct_query(
        self, 
        query: str, 
        dataset_name: Optional[str] = None
    ) -> BaseResponse:
        """
        Executa uma consulta SQL diretamente em um dataset.
        
        Args:
            query: Consulta SQL
            dataset_name: Nome do dataset alvo (opcional se houver apenas um)
            
        Returns:
            Resultado da consulta
        """
        logger.info(f"Executando consulta SQL direta: {query}")
        
        try:
            if not self.sql_executor:
                return ErrorResponse("Executor SQL não inicializado. Carregue datasets primeiro.")
                
            # Usa o SQLExecutor para executar a consulta direta
            result_df = self.sql_executor.execute_direct_query(query, dataset_name)
            
            # Retorna como DataFrameResponse
            return DataFrameResponse(result_df)
        
        except Exception as e:
            logger.error(f"Erro ao executar consulta SQL: {str(e)}")
            return ErrorResponse(f"Erro ao executar consulta SQL: {str(e)}")
    
    def generate_chart(
        self, 
        data: Union[pd.DataFrame, pd.Series], 
        chart_type: str, 
        x: Optional[str] = None, 
        y: Optional[str] = None,
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> ChartResponse:
        """
        Gera uma visualização a partir de um DataFrame.
        
        Args:
            data: DataFrame ou Series para visualização
            chart_type: Tipo de gráfico (bar, line, scatter, hist, etc.)
            x: Coluna para eixo x (opcional)
            y: Coluna para eixo y (opcional)
            title: Título do gráfico (opcional)
            save_path: Caminho para salvar o gráfico (opcional)
            
        Returns:
            ChartResponse com a visualização
        """
        try:
            import matplotlib.pyplot as plt
            
            # Configura o gráfico
            plt.figure(figsize=(10, 6))
            
            # Determina o tipo de gráfico
            if chart_type == 'bar':
                if x and y:
                    data.plot(kind='bar', x=x, y=y)
                else:
                    data.plot(kind='bar')
            elif chart_type == 'line':
                if x and y:
                    data.plot(kind='line', x=x, y=y)
                else:
                    data.plot(kind='line')
            elif chart_type == 'scatter':
                if x and y:
                    data.plot(kind='scatter', x=x, y=y)
                else:
                    # Scatter requer x e y
                    raise ValueError("Scatter plot requer especificação de x e y")
            elif chart_type == 'hist':
                if y:
                    data[y].plot(kind='hist')
                else:
                    data.plot(kind='hist')
            elif chart_type == 'boxplot':
                data.boxplot()
            elif chart_type == 'pie':
                if y:
                    data.plot(kind='pie', y=y)
                else:
                    data.plot(kind='pie')
            else:
                raise ValueError(f"Tipo de gráfico não suportado: {chart_type}")
            
            # Adiciona título se fornecido
            if title:
                plt.title(title)
            
            # Ajusta o layout
            plt.tight_layout()
            
            # Determina caminho para salvar
            if not save_path:
                # Gera nome baseado no tipo e título
                title_slug = "chart" if not title else title.replace(" ", "_").lower()
                save_path = f"{title_slug}_{chart_type}.png"
            
            # Salva o gráfico
            plt.savefig(save_path)
            plt.close()
            
            # Retorna resposta com o caminho
            logger.info(f"Gráfico gerado e salvo em: {save_path}")
            return ChartResponse(save_path)
            
        except Exception as e:
            logger.error(f"Erro ao gerar gráfico: {str(e)}")
            raise ValueError(f"Falha ao gerar gráfico: {str(e)}")
    
    def sanitize_query(self, query: str) -> str:
        """
        Sanitiza uma consulta do usuário removendo conteúdo potencialmente perigoso.
        
        Args:
            query: Consulta do usuário
            
        Returns:
            Consulta sanitizada
        """
        # Remove comandos SQL perigosos
        dangerous_patterns = [
            r'DROP\s+TABLE',
            r'DELETE\s+FROM',
            r'TRUNCATE\s+TABLE',
            r'ALTER\s+TABLE',
            r'CREATE\s+TABLE',
            r'UPDATE\s+.+\s+SET',
            r'INSERT\s+INTO',
            r'EXECUTE\s+',
            r'EXEC\s+',
            r';.*--'
        ]
        
        sanitized_query = query
        
        # Verifica e remove padrões perigosos
        for pattern in dangerous_patterns:
            import re
            sanitized_query = re.sub(pattern, "[REMOVIDO]", sanitized_query, flags=re.IGNORECASE)
        
        return sanitized_query