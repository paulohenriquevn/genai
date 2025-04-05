
"""
Módulo de construtores de consultas (query builders).
Fornece interfaces e implementações para criar e transformar consultas SQL e
outras formas de acesso a dados.
"""

# Importações da base primeiro para evitar dependências circulares
from .query_builder_base import (
    BaseQueryBuilder,
    QuerySQLTransformationManager
)

# Implementações específicas
from .query_builders_implementation import (
    LocalQueryBuilder,
    SqlQueryBuilder,
    ViewQueryBuilder,
    SQLParser
)

# Classe fachada que coordena os builders
from .query_facade import QueryBuilderFacade

# Lista explícita do que é exportado pelo módulo
__all__ = [
    # Interfaces base
    "BaseQueryBuilder",
    "QuerySQLTransformationManager",
    
    # Implementações
    "LocalQueryBuilder",
    "SqlQueryBuilder",
    "ViewQueryBuilder",
    "SQLParser",
    
    # Fachada
    "QueryBuilderFacade"
]