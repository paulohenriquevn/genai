"""
Modelos de dados do sistema GenBI.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

class DataType(Enum):
    """Tipos de dados suportados pelo sistema"""
    STRING = "string"
    NUMBER = "number"
    DATE = "date"
    DATETIME = "datetime"
    BOOLEAN = "boolean"
    JSON = "json"
    RELATIONSHIP = "relationship"

class RelationshipType(Enum):
    """Tipos de relacionamentos entre modelos"""
    ONE_TO_ONE = "ONE_TO_ONE"
    ONE_TO_MANY = "ONE_TO_MANY" 
    MANY_TO_ONE = "MANY_TO_ONE"
    MANY_TO_MANY = "MANY_TO_MANY"

class JoinType(Enum):
    """Tipos de JOIN SQL"""
    INNER = "INNER"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    FULL = "FULL"

class TimeGrain(Enum):
    """Grãos temporais para análise"""
    YEAR = "YEAR"
    QUARTER = "QUARTER"
    MONTH = "MONTH"
    WEEK = "WEEK"
    DAY = "DAY"
    HOUR = "HOUR"
    MINUTE = "MINUTE"

@dataclass
class Column:
    """Definição de uma coluna em um modelo"""
    name: str
    data_type: DataType
    description: Optional[str] = None
    relationship: Optional[str] = None
    is_calculated: bool = False
    expression: Optional[str] = None
    format: Optional[str] = None
    semantic_type: Optional[str] = None
    
    def __post_init__(self):
        if isinstance(self.data_type, str):
            self.data_type = DataType(self.data_type)
            
        if self.data_type == DataType.RELATIONSHIP and not self.relationship:
            raise ValueError("Columns of type RELATIONSHIP must have a relationship specified")
            
        if self.is_calculated and not self.expression:
            raise ValueError("Calculated columns must have an expression")
            
    def to_dict(self) -> Dict[str, Any]:
        """Converte a coluna para um dicionário"""
        return {
            "name": self.name,
            "type": self.data_type.value,
            "description": self.description,
            "relationship": self.relationship,
            "isCalculated": self.is_calculated,
            "expression": self.expression,
            "format": self.format,
            "semanticType": self.semantic_type
        }

@dataclass
class Source:
    """Fonte de dados para um modelo"""
    type: str  # table, query, view
    value: str  # nome da tabela ou consulta SQL
    catalog: Optional[str] = None
    schema: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte a fonte para um dicionário"""
        return {
            "type": self.type,
            "value": self.value,
            "catalog": self.catalog,
            "schema": self.schema
        }

@dataclass
class Cache:
    """Configuração de cache para um modelo"""
    enabled: bool = False
    refresh_interval: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte a configuração de cache para um dicionário"""
        return {
            "enabled": self.enabled,
            "refreshInterval": self.refresh_interval
        }

@dataclass
class Model:
    """Modelo de dados (tabela ou consulta)"""
    name: str
    source: Source
    columns: List[Column]
    description: Optional[str] = None
    primary_key: Optional[List[str]] = None
    cache: Optional[Cache] = None
    tags: Optional[List[str]] = None
    
    def __post_init__(self):
        # Converter dicionários em objetos
        if isinstance(self.source, dict):
            self.source = Source(**self.source)
            
        if isinstance(self.columns, list) and self.columns and isinstance(self.columns[0], dict):
            self.columns = [Column(**col) if isinstance(col, dict) else col for col in self.columns]
            
        if isinstance(self.cache, dict):
            self.cache = Cache(**self.cache)
        elif self.cache is None:
            self.cache = Cache()
            
    def get_column(self, name: str) -> Optional[Column]:
        """Obtém uma coluna pelo nome"""
        for col in self.columns:
            if col.name == name:
                return col
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte o modelo para um dicionário"""
        return {
            "name": self.name,
            "description": self.description,
            "source": self.source.to_dict(),
            "columns": [col.to_dict() for col in self.columns],
            "primaryKey": self.primary_key,
            "cache": self.cache.to_dict() if self.cache else None,
            "tags": self.tags
        }

@dataclass
class Relationship:
    """Relacionamento entre modelos"""
    name: str
    type: RelationshipType
    models: List[str]  # Lista de 2 modelos relacionados
    condition: str  # Condição SQL para o JOIN
    description: Optional[str] = None
    join_type: JoinType = JoinType.INNER
    
    def __post_init__(self):
        if isinstance(self.type, str):
            self.type = RelationshipType(self.type)
            
        if isinstance(self.join_type, str):
            self.join_type = JoinType(self.join_type)
            
        if len(self.models) != 2:
            raise ValueError("Relationships must have exactly 2 models")
            
    def to_dict(self) -> Dict[str, Any]:
        """Converte o relacionamento para um dicionário"""
        return {
            "name": self.name,
            "description": self.description,
            "type": self.type.value,
            "models": self.models,
            "condition": self.condition,
            "joinType": self.join_type.value
        }

@dataclass
class TimeGrainConfig:
    """Configuração de grão temporal para métricas"""
    column: str
    grains: List[TimeGrain]
    
    def __post_init__(self):
        if isinstance(self.grains[0], str):
            self.grains = [TimeGrain(grain) for grain in self.grains]
            
    def to_dict(self) -> Dict[str, Any]:
        """Converte a configuração de grão temporal para um dicionário"""
        return {
            "column": self.column,
            "grains": [grain.value for grain in self.grains]
        }

@dataclass
class Filter:
    """Filtro pré-definido para métricas"""
    name: str
    column: str
    operator: str
    value: Any = None
    default: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte o filtro para um dicionário"""
        return {
            "name": self.name,
            "column": self.column,
            "operator": self.operator,
            "value": self.value,
            "default": self.default
        }

@dataclass
class Metric:
    """Métrica para análise de dados"""
    name: str
    base_model: str
    dimensions: List[Column]
    measures: List[Column]
    description: Optional[str] = None
    time_grains: Optional[List[TimeGrainConfig]] = None
    filters: Optional[List[Filter]] = None
    
    def __post_init__(self):
        # Converter dicionários em objetos
        if isinstance(self.dimensions[0], dict):
            self.dimensions = [Column(**dim) if isinstance(dim, dict) else dim for dim in self.dimensions]
            
        if isinstance(self.measures[0], dict):
            self.measures = [Column(**measure) if isinstance(measure, dict) else measure for measure in self.measures]
            
        if self.time_grains and isinstance(self.time_grains[0], dict):
            self.time_grains = [TimeGrainConfig(**tg) if isinstance(tg, dict) else tg for tg in self.time_grains]
            
        if self.filters and isinstance(self.filters[0], dict):
            self.filters = [Filter(**f) if isinstance(f, dict) else f for f in self.filters]
            
    def to_dict(self) -> Dict[str, Any]:
        """Converte a métrica para um dicionário"""
        return {
            "name": self.name,
            "description": self.description,
            "baseModel": self.base_model,
            "dimensions": [dim.to_dict() for dim in self.dimensions],
            "measures": [measure.to_dict() for measure in self.measures],
            "timeGrains": [tg.to_dict() for tg in self.time_grains] if self.time_grains else [],
            "filters": [f.to_dict() for f in self.filters] if self.filters else []
        }