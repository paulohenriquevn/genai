"""
Módulo core do sistema contendo componentes fundamentais.
Inclui mecanismos de execução de código, gerenciamento de estado de agente,
geração de prompts e processamento de respostas.
"""

# Importações diretas para componentes principais
from .code_executor import AdvancedDynamicCodeExecutor
from .exceptions import QueryExecutionError
from .user_query import UserQuery

# Submódulos organizados - importamos do módulo agent para evitar erros de importação
from .agent.state import AgentState, AgentMemory, AgentConfig

# Imports de outros componentes
from .prompts.base import BasePrompt
from .prompts.generate_python_code_with_sql import GeneratePythonCodeWithSQLPrompt

from .response.base import BaseResponse
from .response.dataframe import DataFrameResponse
from .response.string import StringResponse
from .response.number import NumberResponse
from .response.chart import ChartResponse
from .response.error import ErrorResponse
from .response.parser import ResponseParser

# Lista de componentes exportados pelo módulo core
__all__ = [
    # Componentes principais
    "AdvancedDynamicCodeExecutor",
    "QueryExecutionError",
    "UserQuery",
    
    # Componentes de estado do agente
    "AgentState",
    "AgentMemory",
    "AgentConfig",
    
    # Componentes de prompt
    "BasePrompt",
    "GeneratePythonCodeWithSQLPrompt",
    
    # Componentes de resposta
    "BaseResponse",
    "DataFrameResponse",
    "StringResponse",
    "NumberResponse",
    "ChartResponse",
    "ErrorResponse",
    "ResponseParser"
]