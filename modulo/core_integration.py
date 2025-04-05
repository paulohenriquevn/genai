import os
import logging
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple

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
        max_output_size: int = 1024 * 1024  # 1 MB
    ):
        """
        Inicializa o motor de análise com configurações personalizadas.
        
        Args:
            agent_description: Descrição do agente para o LLM
            default_output_type: Tipo padrão de saída (dataframe, string, number, plot)
            direct_sql: Se True, executa SQL diretamente sem código Python
            timeout: Tempo limite para execução de código (segundos)
            max_output_size: Tamanho máximo da saída
        """
        logger.info(f"Inicializando AnalysisEngine com output_type={default_output_type}")
        
        # Inicialização dos componentes core
        self.code_executor = AdvancedDynamicCodeExecutor(
            timeout=timeout,
            max_output_size=max_output_size
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
            
            # Em uma implementação real, enviaríamos o prompt para o LLM
            # Mas neste caso, vamos simular a geração de código baseado na consulta
            
            # Simula código gerado para as consultas
            simulated_code = self._generate_simulated_code(query)
            self.last_code_generated = simulated_code
            
            # Contexto para execução inclui os datasets
            execution_context = {
                'query': query,
                'datasets': {name: ds.dataframe for name, ds in self.datasets.items()}
            }
            
            # Executa o código gerado
            execution_result = self.code_executor.execute_code(
                simulated_code,
                context=execution_context,
                output_type=self.agent_state.output_type
            )
            
            # Verifica se a execução foi bem-sucedida
            if not execution_result["success"]:
                error_msg = execution_result["error"]
                logger.error(f"Erro na execução de código: {error_msg}")
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
            
    def _generate_simulated_code(self, query: str) -> str:
        """
        Gera código Python simulado com base na consulta.
        Isso é temporário e em uma implementação real seria substituído 
        por código gerado por um LLM.
        
        Args:
            query: Consulta em linguagem natural
            
        Returns:
            Código Python simulado
        """
        query_lower = query.lower()
        
        # Vendas por cliente
        if "total" in query_lower and "vendas" in query_lower and "cliente" in query_lower:
            return """
import pandas as pd

# Acessa o dataset de vendas
vendas_df = datasets['vendas']

# Agrupa vendas por cliente e calcula o total
total_por_cliente = vendas_df.groupby('id_cliente')['valor'].sum().reset_index()
total_por_cliente.columns = ['ID Cliente', 'Total Vendas']

# Formata o resultado
result = {"type": "dataframe", "value": total_por_cliente}
"""

        # Motivos de vendas perdidas
        elif "motivos" in query_lower and "vendas perdidas" in query_lower:
            return """
import pandas as pd

# Acessa o dataset de vendas perdidas
vendas_perdidas_df = datasets['vendas_perdidas']

# Conta ocorrências por motivo e ordena
motivos = vendas_perdidas_df['Motivo'].value_counts().reset_index(name='Contagem')
motivos.columns = ['Motivo', 'Contagem']
top_motivos = motivos.head(3)

# Formata o resultado
result = {"type": "dataframe", "value": top_motivos}
"""

        # Gráfico de barras para impacto financeiro
        elif "gráfico" in query_lower and "barras" in query_lower and "impacto" in query_lower:
            return """
import pandas as pd
import matplotlib.pyplot as plt

# Acessa o dataset de vendas perdidas
vendas_perdidas_df = datasets['vendas_perdidas']

# Agrupa por motivo e soma o impacto financeiro
impacto_por_motivo = vendas_perdidas_df.groupby('Motivo')['ImpactoFinanceiro'].sum().reset_index()

# Cria o gráfico de barras
plt.figure(figsize=(10, 6))
plt.bar(impacto_por_motivo['Motivo'], impacto_por_motivo['ImpactoFinanceiro'])
plt.title('Impacto Financeiro por Motivo de Venda Perdida')
plt.xlabel('Motivo')
plt.ylabel('Impacto Financeiro Total (R$)')
plt.xticks(rotation=45)
plt.tight_layout()

# Salva o gráfico
plt.savefig('impacto_por_motivo.png')

# Formata o resultado
result = {"type": "plot", "value": "impacto_por_motivo.png"}
"""

        # Valor médio de vendas
        elif "valor médio" in query_lower and "vendas" in query_lower:
            return """
import pandas as pd

# Acessa o dataset de vendas
vendas_df = datasets['vendas']

# Calcula a média de valores
valor_medio = vendas_df['valor'].mean()

# Formata o resultado
result = {"type": "number", "value": valor_medio}
"""

        # Clientes de São Paulo
        elif "clientes" in query_lower and "são paulo" in query_lower:
            return """
import pandas as pd

# Acessa o dataset de clientes
clientes_df = datasets['clientes']

# Filtra clientes de São Paulo
clientes_sp = clientes_df[clientes_df['cidade'].str.contains('São Paulo', case=False)]

# Formata o resultado
result = {"type": "dataframe", "value": clientes_sp}
"""

        # Consulta genérica
        else:
            return """
import pandas as pd

# Determina qual dataset usar com base na consulta
if 'vendas perdidas' in query.lower():
    df = datasets['vendas_perdidas']
elif 'clientes' in query.lower():
    df = datasets['clientes']
else:
    df = datasets['vendas']

# Retorna os primeiros registros do dataset como resposta genérica
result = {"type": "dataframe", "value": df.head(5)}
"""
    
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