import json
from typing import List, Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime

class ColumnType(Enum):
    """Supported column data types."""
    STRING = 'string'
    INTEGER = 'int'
    FLOAT = 'float'
    BOOLEAN = 'bool'
    DATE = 'date'
    DATETIME = 'datetime'
    CATEGORICAL = 'categorical'

class TransformationType(Enum):
    """Supported data transformation types."""
    RENAME = 'rename'
    FILLNA = 'fill_na'
    DROP_NA = 'drop_na'
    CONVERT_TYPE = 'convert_type'
    MAP_VALUES = 'map_values'
    CLIP = 'clip'
    NORMALIZE = 'normalize'
    STANDARDIZE = 'standardize'
    ENCODE_CATEGORICAL = 'encode_categorical'
    EXTRACT_DATE = 'extract_date'
    ROUND = 'round'
    UPPERCASE = 'to_uppercase'  # Adicione essa linha
    REPLACE = 'replace'

@dataclass
class ColumnSchema:
    """Schema for defining column metadata."""
    name: str
    type: ColumnType
    description: Optional[str] = None
    nullable: bool = True
    primary_key: bool = False
    unique: bool = False
    default: Any = None
    constraints: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

@dataclass
class RelationSchema:
    """Defines relationships between tables/columns."""
    source_table: str
    source_column: str
    target_table: str
    target_column: str
    relationship_type: str = 'one_to_many'

@dataclass
class TransformationRule:
    """Defines a data transformation rule."""
    type: TransformationType
    column: str
    params: Dict[str, Any]

@dataclass
class SemanticSchema:
    """Comprehensive semantic schema for data sources."""
    name: str
    description: Optional[str] = None
    source_type: str = 'csv'  # csv, postgres, mysql, etc.
    source_path: Optional[str] = None
    columns: List[ColumnSchema] = field(default_factory=list)
    relations: List[RelationSchema] = field(default_factory=list)
    transformations: List[TransformationRule] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    version: str = '1.0.0'
    tags: List[str] = field(default_factory=list)

    def validate(self) -> bool:
        """
        Validates the semantic schema for consistency.
        
        Returns:
            bool: True if schema is valid, False otherwise.
        """
        # Check for duplicate column names
        column_names = [col.name for col in self.columns]
        if len(column_names) != len(set(column_names)):
            return False
        
        # Validate relations
        for relation in self.relations:
            source_cols = [col.name for col in self.columns if col.name == relation.source_column]
            target_cols = [col.name for col in self.columns if col.name == relation.target_column]
            
            if not source_cols or not target_cols:
                return False
        
        return True

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the semantic schema to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the schema.
        """
        return {
            'name': self.name,
            'description': self.description,
            'source_type': self.source_type,
            'source_path': self.source_path,
            'columns': [
                {
                    'name': col.name,
                    'type': col.type.value,
                    'description': col.description,
                    'nullable': col.nullable,
                    'primary_key': col.primary_key,
                    'unique': col.unique,
                    'default': col.default,
                    'constraints': col.constraints,
                    'tags': col.tags
                } for col in self.columns
            ],
            'relations': [
                {
                    'source_table': rel.source_table,
                    'source_column': rel.source_column,
                    'target_table': rel.target_table,
                    'target_column': rel.target_column,
                    'relationship_type': rel.relationship_type
                } for rel in self.relations
            ],
            'transformations': [
                {
                    'type': trans.type.value,
                    'column': trans.column,
                    'params': trans.params
                } for trans in self.transformations
            ],
            'created_at': self.created_at.isoformat(),
            'version': self.version,
            'tags': self.tags
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SemanticSchema':
        """
        Creates a SemanticSchema from a dictionary.
        
        Args:
            data (Dict[str, Any]): Dictionary representation of the schema.
        
        Returns:
            SemanticSchema: Constructed semantic schema.
        """
        columns = [
            ColumnSchema(
                name=col['name'],
                type=ColumnType(col['type']),
                description=col.get('description'),
                nullable=col.get('nullable', True),
                primary_key=col.get('primary_key', False),
                unique=col.get('unique', False),
                default=col.get('default'),
                constraints=col.get('constraints', {}),
                tags=col.get('tags', [])
            ) for col in data.get('columns', [])
        ]

        relations = [
            RelationSchema(
                source_table=rel['source_table'],
                source_column=rel['source_column'],
                target_table=rel['target_table'],
                target_column=rel['target_column'],
                relationship_type=rel.get('relationship_type', 'one_to_many')
            ) for rel in data.get('relations', [])
        ]

        transformations = [
            TransformationRule(
                type=TransformationType(trans['type']),
                column=trans['column'],
                params=trans['params']
            ) for trans in data.get('transformations', [])
        ]

        return cls(
            name=data['name'],
            description=data.get('description'),
            source_type=data.get('source_type', 'csv'),
            source_path=data.get('source_path'),
            columns=columns,
            relations=relations,
            transformations=transformations,
            created_at=datetime.fromisoformat(data.get('created_at', datetime.now().isoformat())),
            version=data.get('version', '1.0.0'),
            tags=data.get('tags', [])
        )

    def save_to_file(self, filepath: str) -> None:
        """
        Saves the semantic schema to a JSON file.
        
        Args:
            filepath (str): Path to save the JSON file.
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_file(cls, filepath: str) -> 'SemanticSchema':
        """
        Loads a semantic schema from a JSON file.
        
        Args:
            filepath (str): Path to the JSON file.
        
        Returns:
            SemanticSchema: Loaded semantic schema.
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)