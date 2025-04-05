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

# Importação do módulo de integração com LLMs
from llm_integration import LLMIntegration, LLMQueryGenerator

# Configura o logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("core_integration.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("core_integration")


class Dataset:
    """
    Representa um dataset com metadados e descrição para uso no motor de análise.
    """
    
    def __init__(
        self, 
        dataframe: pd.DataFrame, 
        name: str, 
        description: str = "", 
        schema: Dict[str, str] = None
    ):
        """
        Inicializa um objeto Dataset.
        
        Args:
            dataframe: DataFrame Pandas com os dados
            name: Nome do dataset
            description: Descrição do conjunto de dados
            schema: Dicionário de metadados sobre as colunas (opcional)
        """
        self.dataframe = dataframe
        self.name = name
        self.description = description
        self.schema = schema or {}
    
    def to_json(self) -> Dict[str, Any]:
        """
        Converte o dataset para um formato JSON para uso em prompts.
        
        Returns:
            Dict com informações sobre o dataset
        """
        # Cria uma representação simplificada para o LLM
        columns = []
        for col in self.dataframe.columns:
            col_type = str(self.dataframe[col].dtype)
            sample = str(self.dataframe[col].iloc[0]) if len(self.dataframe) > 0 else ""
            
            # Tenta obter descrição do schema se disponível
            description = self.schema.get(col, f"Column {col} of type {col_type}")
            
            columns.append({
                "name": col,
                "type": col_type,
                "sample": sample,
                "description": description
            })
        
        # Estrutura completa
        return {
            "name": self.name,
            "description": self.description,
            "row_count": len(self.dataframe),
            "column_count": len(self.dataframe.columns),
            "columns": columns,
            "sample": self.dataframe.head(3).to_dict(orient="records")
        }
        
    def serialize_dataframe(self) -> Dict[str, Any]:
        """
        Serializa o dataframe para uso no prompt template.
        Método requerido pela integração com o template de prompt.
        
        Returns:
            Dict com informações do dataframe
        """
        return {
            "name": self.name,
            "description": self.description,
            "dataframe": self.dataframe
        }


class AnalysisEngine:
    """
    Motor de análise que integra componentes core para processamento de consultas em linguagem natural.
    
    Esta classe implementa:
    - Carregamento e gerenciamento de datasets
    - Execução segura de código
    - Geração de prompts para LLM
    - Processamento de consultas em linguagem natural
    - Tratamento de respostas e conversão de formatos
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
    
    def process_query(self, query: str) -> BaseResponse:
        """
        Processa uma consulta em linguagem natural.
        
        Args:
            query: Consulta em linguagem natural
            
        Returns:
            Objeto BaseResponse com o resultado da consulta
        """
        logger.info(f"Processando consulta: {query}")
        
        try:
            # Cria objeto UserQuery
            user_query = UserQuery(query)
            
            # Verifica se há datasets carregados
            if not self.datasets:
                return ErrorResponse("Nenhum dataset carregado. Carregue dados antes de executar consultas.")
            
            # Verifica menções a dados inexistentes
            # Lista de palavras-chave que indicam consultas sobre produtos
            product_keywords = ['produtos', 'produto', 'estoque', 'inventário', 'item', 'itens', 'mercadoria']
            
            # Verifica se a consulta menciona produtos (que não existem nos dados)
            if any(keyword in query.lower() for keyword in product_keywords) and not any('produto' in ds.name.lower() for ds in self.datasets.values()):
                datasets_desc = ", ".join([f"{name} ({', '.join(ds.dataframe.columns[:3])}...)" for name, ds in self.datasets.items()])
                return StringResponse(
                    f"Não há dados sobre produtos disponíveis. Os datasets disponíveis são: {datasets_desc}. "
                    f"Por favor, reformule sua consulta para usar os dados disponíveis."
                )
            
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
                'datasets': {name: ds.dataframe for name, ds in self.datasets.items()}
            }
            
            # Configuração da função execute_sql_query
            if len(self.datasets) > 0:
                execution_context['execute_sql_query'] = self._create_sql_executor()
            
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
                    datasets_desc = ", ".join([f"{name} ({', '.join(ds.dataframe.columns[:3])}...)" for name, ds in self.datasets.items()])
                    return StringResponse(
                        f"Não foi possível executar a consulta porque os dados mencionados não estão disponíveis. "
                        f"Os datasets disponíveis são: {datasets_desc}. "
                        f"Por favor, reformule sua consulta para usar apenas os dados disponíveis."
                    )
                
                # Tenta corrigir o erro (opcional)
                if "correction_attempt" not in execution_context:
                    return self._attempt_error_correction(query, generated_code, error_msg, execution_context)
                
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
                
                logger.info(f"Consulta processada com sucesso. Tipo de resposta: {response.type}")
                return response
                
            except Exception as e:
                logger.error(f"Erro ao processar resposta: {str(e)}")
                return ErrorResponse(f"Erro no processamento da resposta: {str(e)}")
        
        except Exception as e:
            logger.error(f"Erro ao processar consulta: {str(e)}")
            return ErrorResponse(f"Erro ao processar consulta: {str(e)}")
    
    def _create_sql_executor(self):
        """
        Cria uma função para executar consultas SQL em datasets.
        
        Returns:
            Função que executa SQL em DataFrames com suporte a funções SQL compatíveis
        """
        # Integração com DuckDB para execução SQL mais robusta
        try:
            import duckdb
            import re
            from datetime import datetime
            
            def adapt_sql_query(sql_query: str) -> str:
                """
                Adapta uma consulta SQL para compatibilidade com DuckDB.
                
                Args:
                    sql_query: Consulta SQL original
                    
                Returns:
                    Consulta SQL adaptada para DuckDB
                """
                # Verificação de tabelas existentes
                table_names = list(self.datasets.keys())
                
                # Verifica se a consulta referencia tabelas inexistentes
                for table in re.findall(r'FROM\s+(\w+)', sql_query, re.IGNORECASE):
                    if table not in table_names:
                        logger.warning(f"Tabela '{table}' não encontrada nos datasets carregados")
                
                # Substitui funções de data incompatíveis
                # DATE_FORMAT(campo, '%Y-%m-%d') -> strftime('%Y-%m-%d', campo)
                sql_query = re.sub(
                    r'DATE_FORMAT\s*\(\s*([^,]+)\s*,\s*[\'"]([^\'"]+)[\'"]\s*\)',
                    r"strftime('\2', \1)",
                    sql_query
                )
                
                # TO_DATE(string) -> DATE(string)
                sql_query = re.sub(
                    r'TO_DATE\s*\(\s*([^)]+)\s*\)',
                    r'DATE(\1)',
                    sql_query
                )
                
                # Funções de string
                # CONCAT(a, b) -> a || b
                sql_query = re.sub(
                    r'CONCAT\s*\(\s*([^,]+)\s*,\s*([^)]+)\s*\)',
                    r'(\1 || \2)',
                    sql_query
                )
                
                # SUBSTRING(x, start, len) -> SUBSTR(x, start, len)
                sql_query = re.sub(
                    r'SUBSTRING\s*\(',
                    r'SUBSTR(',
                    sql_query
                )
                
                # Funções de agregação
                # GROUP_CONCAT -> STRING_AGG
                sql_query = re.sub(
                    r'GROUP_CONCAT\s*\(',
                    r'STRING_AGG(',
                    sql_query
                )
                
                logger.debug(f"Consulta SQL adaptada: {sql_query}")
                return sql_query
            
            def check_table_existence(sql_query: str) -> None:
                """Verifica se as tabelas referenciadas existem."""
                table_refs = re.findall(r'FROM\s+(\w+)', sql_query, re.IGNORECASE)
                table_refs.extend(re.findall(r'JOIN\s+(\w+)', sql_query, re.IGNORECASE))
                
                for table in table_refs:
                    if table not in self.datasets:
                        raise ValueError(f"Tabela '{table}' não encontrada nos datasets carregados. " + 
                                       f"Datasets disponíveis: {', '.join(self.datasets.keys())}")
            
            def register_custom_sql_functions(con: duckdb.DuckDBPyConnection) -> None:
                """
                Registra funções SQL personalizadas no DuckDB para ampliar a compatibilidade
                com outros dialetos SQL, usando abordagem simplificada.
                
                Args:
                    con: Conexão DuckDB
                """
                try:
                    # Função utilitária para criar SQL functions de forma segura
                    def safe_create_function(sql):
                        try:
                            con.execute(sql)
                        except Exception as e:
                            logger.warning(f"Erro ao criar função SQL: {str(e)}")
                    
                    # GROUP_CONCAT para compatibilidade com MySQL
                    safe_create_function("CREATE OR REPLACE MACRO GROUP_CONCAT(x) AS STRING_AGG(x, ',')")
                    
                    # DATE_FORMAT simplificada (casos mais comuns)
                    safe_create_function("""
                    CREATE OR REPLACE MACRO DATE_FORMAT(d, f) AS
                    CASE 
                        WHEN f = '%Y-%m-%d' THEN strftime('%Y-%m-%d', d)
                        WHEN f = '%Y-%m' THEN strftime('%Y-%m', d)
                        WHEN f = '%Y' THEN strftime('%Y', d)
                        ELSE strftime('%Y-%m-%d', d)
                    END
                    """)
                    
                    # TO_DATE para converter para data
                    safe_create_function("CREATE OR REPLACE MACRO TO_DATE(d) AS TRY_CAST(d AS DATE)")
                    
                    # String concatenation helpers
                    safe_create_function("CREATE OR REPLACE MACRO CONCAT(a, b) AS a || b")
                    
                    # Concat with separator (simplified version)
                    safe_create_function("""
                    CREATE OR REPLACE MACRO CONCAT_WS(sep, a, b) AS
                    CASE 
                        WHEN a IS NULL AND b IS NULL THEN NULL
                        WHEN a IS NULL THEN b
                        WHEN b IS NULL THEN a
                        ELSE a || sep || b
                    END
                    """)
                    
                    # Register extract functions for date parts
                    safe_create_function("""
                    CREATE OR REPLACE MACRO YEAR(d) AS EXTRACT(YEAR FROM d)
                    """)
                    
                    safe_create_function("""
                    CREATE OR REPLACE MACRO MONTH(d) AS EXTRACT(MONTH FROM d)
                    """)
                    
                    safe_create_function("""
                    CREATE OR REPLACE MACRO DAY(d) AS EXTRACT(DAY FROM d)
                    """)
                    
                    logger.info("Funções SQL personalizadas registradas com sucesso")
                    
                except Exception as e:
                    logger.warning(f"Erro ao registrar funções SQL personalizadas: {str(e)}")
            
            def execute_sql(sql_query: str) -> pd.DataFrame:
                """Executa uma consulta SQL usando DuckDB com adaptações de compatibilidade."""
                try:
                    # Verifica se tabelas existem antes de executar
                    check_table_existence(sql_query)
                    
                    # Adapta a consulta para compatibilidade com DuckDB
                    adapted_query = adapt_sql_query(sql_query)
                    
                    # Estabelece conexão com todos os dataframes
                    con = duckdb.connect(database=':memory:')
                    
                    # Registra funções SQL personalizadas
                    register_custom_sql_functions(con)
                    
                    # Registra todos os datasets
                    for name, dataset in self.datasets.items():
                        # Registra o dataframe
                        con.register(name, dataset.dataframe)
                        
                        # Cria visualizações otimizadas para funções de data
                        con.execute(f"""
                        CREATE OR REPLACE VIEW {name}_date_view AS 
                        SELECT * FROM {name}
                        """)
                    
                    # Executa a consulta
                    result = con.execute(adapted_query).fetchdf()
                    
                    # Registra a consulta SQL para debugging
                    sql_logger = logging.getLogger("sql_logger")
                    sql_logger.info(f"Consulta SQL executada: {adapted_query}")
                    
                    return result
                except Exception as e:
                    logger.error(f"Erro SQL: {str(e)}")
                    raise QueryExecutionError(f"Erro ao executar SQL: {str(e)}")
        
        except ImportError:
            # Fallback para pandas se DuckDB não estiver disponível
            logger.warning("DuckDB não encontrado, usando pandas para consultas SQL")
            
            def execute_sql(sql_query: str) -> pd.DataFrame:
                """Executa uma consulta SQL básica usando pandas."""
                try:
                    # Para o modo pandas, suporta apenas SELECT * FROM dataset
                    import re
                    match = re.search(r'FROM\s+(\w+)', sql_query, re.IGNORECASE)
                    
                    if not match:
                        raise ValueError("Consulta SQL inválida. Formato esperado: SELECT * FROM dataset")
                    
                    dataset_name = match.group(1)
                    
                    if dataset_name not in self.datasets:
                        raise ValueError(f"Dataset '{dataset_name}' não encontrado")
                    
                    # Registra a consulta SQL para debugging
                    sql_logger = logging.getLogger("sql_logger")
                    sql_logger.info(f"Consulta SQL simulada: {sql_query}")
                    
                    # Retorna o dataset inteiro (limitação do modo pandas)
                    return self.datasets[dataset_name].dataframe
                except Exception as e:
                    logger.error(f"Erro SQL: {str(e)}")
                    raise QueryExecutionError(f"Erro ao executar SQL: {str(e)}")
        
        return execute_sql
    
    def _attempt_error_correction(self, query: str, original_code: str, error_msg: str, context: Dict[str, Any]) -> BaseResponse:
        """
        Tenta corrigir um código com erro usando o LLM, com suporte especial para erros de SQL.
        
        Args:
            query: Consulta original
            original_code: Código com erro
            error_msg: Mensagem de erro
            context: Contexto de execução
            
        Returns:
            Resposta após tentativa de correção
        """
        logger.info(f"Tentando corrigir erro: {error_msg}")
        
        # Verifica se é um erro relacionado a SQL
        is_sql_error = any(keyword in error_msg.lower() for keyword in 
                          ['sql', 'query', 'syntax', 'column', 'table', 'from', 'select', 
                           'date', 'function', 'duckdb', 'type', 'conversion'])
        
        # Lista de datasets disponíveis
        datasets_list = ", ".join(self.datasets.keys())
        
        # Adiciona sugestões específicas para erros de SQL
        sql_correction_tips = ""
        if is_sql_error:
            sql_correction_tips = f"""
            DICAS PARA CORREÇÃO DE SQL:
            
            1. Datasets disponíveis: {datasets_list}
            2. Use strftime('%Y-%m-%d', coluna) em vez de DATE_FORMAT
            3. Use DATE(string) em vez de TO_DATE
            4. Use coluna1 || coluna2 em vez de CONCAT
            5. Verifique se todas as tabelas mencionadas no SQL existem
            6. DuckDB é sensível a tipos - converta dados quando necessário
            7. Certifique-se de que as colunas referenciadas existem nas tabelas
            8. Verifique a sintaxe SQL - DuckDB segue o padrão PostgreSQL
            """
        
        # Cria um prompt para correção
        correction_prompt = f"""
        O código gerado para a consulta "{query}" falhou com o seguinte erro:
        
        ERROR:
        {error_msg}
        
        CÓDIGO ORIGINAL:
        {original_code}
        
        {sql_correction_tips}
        
        Por favor, corrija o código levando em conta o erro. Forneça apenas o código Python corrigido,
        não explicações. Lembre-se que o resultado deve ser um dicionário no formato:
        result = {{"type": tipo, "value": valor}}
        onde o tipo pode ser "string", "number", "dataframe", ou "plot".
        
        Se a consulta mencionar dados que não existem (como 'produtos'), adapte para usar dados disponíveis
        ou retorne uma mensagem explicando que esses dados não estão disponíveis.
        """
        
        try:
            # Gera código corrigido
            corrected_code = self.query_generator.generate_code(correction_prompt)
            logger.info("Código corrigido gerado")
            
            # Se for um erro de SQL, tenta extrair a consulta para validação
            if is_sql_error:
                import re
                sql_matches = re.findall(r'execute_sql_query\([\'"](.+?)[\'"]\)', corrected_code)
                
                if sql_matches:
                    # Pega a primeira consulta SQL encontrada
                    sql_query = sql_matches[0]
                    logger.info(f"Validando consulta SQL corrigida: {sql_query}")
                    
                    # Verifica se a consulta menciona tabelas inexistentes
                    for table in re.findall(r'FROM\s+(\w+)', sql_query, re.IGNORECASE):
                        if table not in self.datasets:
                            # Se a tabela não existir, modifica o código para retornar uma mensagem amigável
                            logger.warning(f"Correção ainda referencia tabela inexistente: {table}")
                            corrected_code = f"""
                            result = {{
                                "type": "string",
                                "value": "Não foi possível processar a consulta porque a tabela '{table}' não está disponível. Tabelas disponíveis: {datasets_list}"
                            }}
                            """
                            break
            
            # Marca o contexto para evitar loop infinito
            context_with_flag = context.copy()
            context_with_flag['correction_attempt'] = True
            
            # Executa o código corrigido
            execution_result = self.code_executor.execute_code(
                corrected_code,
                context=context_with_flag,
                output_type=self.agent_state.output_type
            )
            
            # Verifica se a correção foi bem-sucedida
            if not execution_result["success"]:
                # Se a primeira correção falhar, tenta uma correção mais simples para casos graves
                error_msg = execution_result["error"]
                logger.error(f"Primeira correção falhou: {error_msg}")
                
                # Tentativa de fallback - gera uma resposta mais simples
                simplified_correction = f"""
                # A consulta apresentou problemas técnicos. Criando uma resposta simplificada.
                
                result = {{
                    "type": "string",
                    "value": "Não foi possível processar a consulta '{query}' devido a limitações técnicas. Erro: {error_msg}"
                }}
                """
                
                # Executa a versão simplificada
                fallback_result = self.code_executor.execute_code(
                    simplified_correction,
                    context=context_with_flag,
                    output_type=self.agent_state.output_type
                )
                
                if not fallback_result["success"]:
                    return ErrorResponse(f"Erro ao processar consulta (após todas as tentativas de correção): {error_msg}")
                else:
                    # Usa o resultado do fallback
                    result = fallback_result["result"]
                    formatted_result = self._format_result_for_parser(result)
                    response = self.response_parser.parse(formatted_result, simplified_correction)
                    return response
            
            # Processa o resultado da execução corrigida
            result = execution_result["result"]
            formatted_result = self._format_result_for_parser(result)
            response = self.response_parser.parse(formatted_result, corrected_code)
            
            logger.info(f"Consulta corrigida e processada com sucesso. Tipo de resposta: {response.type}")
            return response
            
        except Exception as e:
            logger.error(f"Erro durante tentativa de correção: {str(e)}")
            
            # Em caso de erro na correção, cria uma resposta de erro mais amigável
            try:
                simplified_response = f"""
                result = {{
                    "type": "string",
                    "value": "Não foi possível processar a consulta devido a problemas técnicos. Por favor, tente reformular sua pergunta de forma mais simples ou verifique se os dados mencionados existem."
                }}
                """
                
                # Tenta executar a resposta simplificada
                fallback_execution = self.code_executor.execute_code(
                    simplified_response,
                    context=context,
                    output_type=self.agent_state.output_type
                )
                
                if fallback_execution["success"]:
                    result = fallback_execution["result"]
                    formatted_result = self._format_result_for_parser(result)
                    return self.response_parser.parse(formatted_result, simplified_response)
            except:
                pass
                
            # Se tudo falhar, retorna erro
            return ErrorResponse(f"Erro ao processar consulta: {error_msg} (Correção falhou: {str(e)})")
    
    def _generate_prompt(self, query: str) -> str:
        """
        Gera um prompt detalhado para o LLM com informações sobre datasets disponíveis.
        
        Args:
            query: Consulta em linguagem natural
            
        Returns:
            Prompt formatado
        """
        # Adiciona a consulta ao histórico
        self.agent_state.memory.add_message(query)
        
        # Informações sobre datasets disponíveis
        datasets_info = "\n".join([
            f"Dataset '{name}':\n" +
            f"  - Descrição: {dataset.description}\n" +
            f"  - Registros: {len(dataset.dataframe)}\n" +
            f"  - Colunas: {', '.join(dataset.dataframe.columns)}\n" +
            f"  - Tipos: {', '.join([f'{col}: {dataset.dataframe[col].dtype}' for col in dataset.dataframe.columns])}\n"
            for name, dataset in self.datasets.items()
        ])
        
        # Exemplos de valores para cada dataset
        dataset_samples = "\n".join([
            f"Exemplos de '{name}':\n{dataset.dataframe.head(2).to_string()}\n"
            for name, dataset in self.datasets.items()
        ])
        
        # Informações sobre funções SQL suportadas
        sql_functions_info = """
        ## Funções SQL Suportadas
        
        O sistema usa DuckDB para executar consultas SQL e foi expandido para suportar funções de vários dialetos SQL:
        
        ### Funções de Data
        - DATE_FORMAT(coluna, formato) - Formata data/hora no estilo MySQL (ex: DATE_FORMAT(data, '%Y-%m-%d'))
        - strftime(formato, coluna) - Formata data/hora no estilo SQLite (ex: strftime('%Y-%m-%d', data))
        - DATE(string) - Converte string para data (ex: DATE '2023-01-01' ou DATE(coluna))
        - TO_DATE(string) - Converte string para data no estilo PostgreSQL
        - DATE_PART(parte, data) - Extrai parte específica de data (ex: DATE_PART('year', data))
        - DATEADD(parte, n, data) - Adiciona intervalo de tempo a uma data no estilo SQL Server
        - EXTRACT(parte FROM data) - Extrai parte de data no estilo PostgreSQL
        
        ### Funções de String
        - CONCAT(a, b) - Concatena strings no estilo MySQL/PostgreSQL
        - a || b - Concatena strings no estilo SQLite/PostgreSQL
        - CONCAT_WS(separador, a, b, ...) - Concatena strings com separador
        - SUBSTR(string, inicio, tamanho) - Extrai substring
        - SUBSTRING(string, inicio, tamanho) - Mesmo que SUBSTR
        - LOWER(string) - Converte para minúsculas
        - UPPER(string) - Converte para maiúsculas
        - TRIM(string) - Remove espaços do início e fim
        
        ### Funções de Agregação
        - COUNT(), SUM(), AVG(), MIN(), MAX() - Funções de agregação padrão
        - GROUP_CONCAT(coluna) - Concatena valores agrupados com vírgula (estilo MySQL)
        - STRING_AGG(coluna, separador) - Concatena valores com separador (estilo PostgreSQL)
        
        ### Funções de Casting e Conversão
        - CAST(valor AS tipo) - Converte para outro tipo de dados
        - valor::tipo - Converte para outro tipo no estilo PostgreSQL
        - CONVERT(tipo, valor) - Converte para outro tipo no estilo SQL Server/MySQL
        
        ### Exemplos válidos de consultas SQL
        - SELECT * FROM vendas WHERE data >= DATE '2023-01-01'
        - SELECT DATE_FORMAT(data, '%Y-%m') as mes, SUM(valor) FROM vendas GROUP BY mes
        - SELECT strftime('%Y-%m', data) as mes, AVG(valor) FROM vendas GROUP BY mes
        - SELECT c.nome, SUM(v.valor) FROM vendas v JOIN clientes c ON v.id_cliente = c.id_cliente GROUP BY c.nome
        - SELECT EXTRACT(YEAR FROM data) as ano, EXTRACT(MONTH FROM data) as mes, SUM(valor) FROM vendas GROUP BY ano, mes
        - SELECT id_cliente, GROUP_CONCAT(id_venda) as vendas FROM vendas GROUP BY id_cliente
        """
        
        # Construindo o prompt
        prompt = f"""
        # Instruções para Geração de Código Python

        Você deve gerar código Python para responder à seguinte consulta:
        
        CONSULTA: "{query}"
        
        ## Datasets Disponíveis

        {datasets_info}
        
        ## Exemplos de Dados

        {dataset_samples}
        
        {sql_functions_info}
        
        ## Requisitos

        1. Use a função `execute_sql_query(sql_query)` para executar consultas SQL
        2. A função execute_sql_query retorna um DataFrame pandas
        3. O código DEVE definir uma variável `result` no formato: {{"type": tipo, "value": valor}}
        4. Tipos válidos são: "string", "number", "dataframe", ou "plot"
        5. Para visualizações, use matplotlib e salve o gráfico com plt.savefig()

        ## Importante

        - Importe apenas as bibliotecas necessárias
        - Use SQL para consultas e agregações sempre que possível
        - Para visualizações, use o tipo "plot" e salve o gráfico em um arquivo
        - Defina result = {{"type": "tipo_aqui", "value": valor_aqui}} ao final
        - NÃO inclua comentários explicativos, apenas o código funcional
        - Use apenas os datasets indicados acima, NÃO tente usar tabelas inexistentes como 'produtos'
        - Adapte consultas SQL para compatibilidade com DuckDB usando as funções listadas acima
        """
        
        return prompt
    
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
            # Determina qual dataset usar
            if dataset_name:
                if dataset_name not in self.datasets:
                    return ErrorResponse(f"Dataset '{dataset_name}' não encontrado")
                df = self.datasets[dataset_name].dataframe
            elif len(self.datasets) == 1:
                # Se há apenas um dataset, usa ele
                df = next(iter(self.datasets.values())).dataframe
            else:
                # Se há múltiplos datasets e nenhum especificado
                return ErrorResponse("Múltiplos datasets disponíveis. Especifique qual usar.")
            
            # Executa a consulta SQL usando pandas
            # Em uma implementação real, usaríamos o DuckDB ou SQLite para suporte SQL real
            result_df = pd.read_sql_query(query, df)
            
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