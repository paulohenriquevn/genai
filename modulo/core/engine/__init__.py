"""
Módulo de componentes core do motor de análise.

Este módulo contém os componentes centrais do sistema de análise,
cada um com uma responsabilidade bem definida:

- Dataset: Gerencia dados e metadados
- SQLExecutor: Processa consultas SQL com adaptação de dialetos
- AlternativeFlow: Provê fluxos alternativos para erros e sugestões
- FeedbackManager: Gerencia feedback do usuário e otimização de consultas
"""

from core.engine.dataset import Dataset
from core.engine.sql_executor import SQLExecutor
from core.engine.alternative_flow import AlternativeFlow
from core.engine.feedback_manager import FeedbackManager

__all__ = [
    'Dataset',
    'SQLExecutor', 
    'AlternativeFlow',
    'FeedbackManager'
]