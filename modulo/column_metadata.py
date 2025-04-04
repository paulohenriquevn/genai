import json
import logging
from typing import Dict, Any, List, Optional


logger = logging.getLogger("column_metadata")

class ColumnMetadata:
    """
    Armazena metadados para uma coluna específica.
    
    Attributes:
        name (str): Nome da coluna no dataset.
        description (str): Descrição da finalidade/significado da coluna.
        data_type (str): Tipo de dados esperado (str, int, float, date, etc).
        format (str): Formato específico (ex: YYYY-MM-DD para datas).
        alias (List[str]): Nomes alternativos para a coluna.
        aggregations (List[str]): Agregações recomendadas (sum, avg, etc).
        validation (Dict): Regras de validação (min, max, etc).
        display (Dict): Preferências de exibição (precision, unit, etc).
        tags (List[str]): Tags para categorização.
    """
    
    def __init__(
        self,
        name: str,
        description: str = None,
        data_type: str = None,
        format: str = None,
        alias: List[str] = None,
        aggregations: List[str] = None,
        validation: Dict[str, Any] = None,
        display: Dict[str, Any] = None,
        tags: List[str] = None
    ):
        """
        Inicializa metadados para uma coluna.
        
        Args:
            name: Nome da coluna.
            description: Descrição/propósito da coluna.
            data_type: Tipo de dados esperado.
            format: Formato específico.
            alias: Nomes alternativos para a coluna.
            aggregations: Agregações recomendadas.
            validation: Regras de validação.
            display: Configurações de exibição.
            tags: Tags para categorização.
        """
        self.name = name
        self.description = description
        self.data_type = data_type
        self.format = format
        self.alias = alias or []
        self.aggregations = aggregations or []
        self.validation = validation or {}
        self.display = display or {}
        self.tags = tags or []
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ColumnMetadata':
        """
        Cria uma instância de ColumnMetadata a partir de um dicionário.
        
        Args:
            data: Dicionário com os metadados da coluna.
            
        Returns:
            ColumnMetadata: Nova instância.
        """
        if 'name' not in data:
            raise ValueError("O metadado da coluna deve conter o campo 'name'")
        
        return cls(
            name=data['name'],
            description=data.get('description'),
            data_type=data.get('data_type'),
            format=data.get('format'),
            alias=data.get('alias'),
            aggregations=data.get('aggregations'),
            validation=data.get('validation'),
            display=data.get('display'),
            tags=data.get('tags')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Converte os metadados para um dicionário.
        
        Returns:
            Dict: Representação em dicionário.
        """
        result = {'name': self.name}
        
        if self.description:
            result['description'] = self.description
        if self.data_type:
            result['data_type'] = self.data_type
        if self.format:
            result['format'] = self.format
        if self.alias:
            result['alias'] = self.alias
        if self.aggregations:
            result['aggregations'] = self.aggregations
        if self.validation:
            result['validation'] = self.validation
        if self.display:
            result['display'] = self.display
        if self.tags:
            result['tags'] = self.tags
            
        return result

class DatasetMetadata:
    """
    Armazena metadados para um dataset completo.
    
    Attributes:
        name (str): Nome do dataset.
        description (str): Descrição do dataset.
        source (str): Origem dos dados.
        columns (Dict[str, ColumnMetadata]): Metadados de cada coluna.
        created_at (str): Data de criação.
        updated_at (str): Data da última atualização.
        version (str): Versão dos metadados.
        tags (List[str]): Tags para categorização.
        owner (str): Proprietário do dataset.
        custom (Dict): Campos personalizados adicionais.
    """
    
    def __init__(
        self,
        name: str,
        description: str = None,
        source: str = None,
        columns: Dict[str, ColumnMetadata] = None,
        created_at: str = None,
        updated_at: str = None,
        version: str = None,
        tags: List[str] = None,
        owner: str = None,
        custom: Dict[str, Any] = None
    ):
        """
        Inicializa metadados para um dataset.
        
        Args:
            name: Nome do dataset.
            description: Descrição do dataset.
            source: Origem dos dados.
            columns: Metadados de cada coluna.
            created_at: Data de criação.
            updated_at: Data da última atualização.
            version: Versão dos metadados.
            tags: Tags para categorização.
            owner: Proprietário do dataset.
            custom: Campos personalizados adicionais.
        """
        self.name = name
        self.description = description
        self.source = source
        self.columns = columns or {}
        self.created_at = created_at
        self.updated_at = updated_at
        self.version = version
        self.tags = tags or []
        self.owner = owner
        self.custom = custom or {}
        
        # Cria lookup para nomes alternativos (alias) das colunas
        self._alias_lookup = {}
        for column_name, metadata in self.columns.items():
            for alias in metadata.alias:
                self._alias_lookup[alias.lower()] = column_name
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DatasetMetadata':
        """
        Cria uma instância de DatasetMetadata a partir de um dicionário.
        
        Args:
            data: Dicionário com os metadados do dataset.
            
        Returns:
            DatasetMetadata: Nova instância.
        """
        if 'name' not in data:
            raise ValueError("O metadado do dataset deve conter o campo 'name'")
        
        # Processa metadados de colunas
        columns = {}
        if 'columns' in data and isinstance(data['columns'], list):
            for col_data in data['columns']:
                col_metadata = ColumnMetadata.from_dict(col_data)
                columns[col_metadata.name] = col_metadata
        
        return cls(
            name=data['name'],
            description=data.get('description'),
            source=data.get('source'),
            columns=columns,
            created_at=data.get('created_at'),
            updated_at=data.get('updated_at'),
            version=data.get('version'),
            tags=data.get('tags'),
            owner=data.get('owner'),
            custom=data.get('custom')
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> 'DatasetMetadata':
        """
        Cria uma instância de DatasetMetadata a partir de uma string JSON.
        
        Args:
            json_str: String JSON com os metadados.
            
        Returns:
            DatasetMetadata: Nova instância.
        """
        try:
            data = json.loads(json_str)
            return cls.from_dict(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON inválido: {str(e)}")
    
    @classmethod
    def from_file(cls, file_path: str) -> 'DatasetMetadata':
        """
        Cria uma instância de DatasetMetadata a partir de um arquivo JSON.
        
        Args:
            file_path: Caminho para o arquivo JSON.
            
        Returns:
            DatasetMetadata: Nova instância.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return cls.from_json(f.read())
        except FileNotFoundError:
            raise ValueError(f"Arquivo não encontrado: {file_path}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Converte os metadados para um dicionário.
        
        Returns:
            Dict: Representação em dicionário.
        """
        result = {'name': self.name}
        
        if self.description:
            result['description'] = self.description
        if self.source:
            result['source'] = self.source
        
        if self.columns:
            result['columns'] = [col.to_dict() for col in self.columns.values()]
            
        if self.created_at:
            result['created_at'] = self.created_at
        if self.updated_at:
            result['updated_at'] = self.updated_at
        if self.version:
            result['version'] = self.version
        if self.tags:
            result['tags'] = self.tags
        if self.owner:
            result['owner'] = self.owner
        if self.custom:
            result['custom'] = self.custom
            
        return result
    
    def to_json(self, indent: int = 2) -> str:
        """
        Converte os metadados para uma string JSON.
        
        Args:
            indent: Número de espaços para indentação.
            
        Returns:
            str: Representação JSON.
        """
        return json.dumps(self.to_dict(), indent=indent)
    
    def save_to_file(self, file_path: str, indent: int = 2) -> None:
        """
        Salva os metadados em um arquivo JSON.
        
        Args:
            file_path: Caminho para o arquivo.
            indent: Número de espaços para indentação.
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(self.to_json(indent))
    
    def get_column_metadata(self, column_name: str) -> Optional[ColumnMetadata]:
        """
        Obtém os metadados de uma coluna específica.
        
        Args:
            column_name: Nome da coluna.
            
        Returns:
            Optional[ColumnMetadata]: Metadados da coluna ou None se não encontrado.
        """
        # Verifica nome exato
        if column_name in self.columns:
            return self.columns[column_name]
        
        # Verifica alias
        column_lower = column_name.lower()
        if column_lower in self._alias_lookup:
            actual_name = self._alias_lookup[column_lower]
            return self.columns[actual_name]
        
        return None
    
    def get_columns_by_tag(self, tag: str) -> List[str]:
        """
        Obtém os nomes das colunas com uma tag específica.
        
        Args:
            tag: Tag para filtrar.
            
        Returns:
            List[str]: Lista de nomes de colunas.
        """
        return [name for name, metadata in self.columns.items() 
                if tag in metadata.tags]
    
    def get_columns_by_type(self, data_type: str) -> List[str]:
        """
        Obtém os nomes das colunas com um tipo específico.
        
        Args:
            data_type: Tipo de dados para filtrar.
            
        Returns:
            List[str]: Lista de nomes de colunas.
        """
        return [name for name, metadata in self.columns.items() 
                if metadata.data_type == data_type]
    
    def get_recommended_aggregations(self, column_name: str) -> List[str]:
        """
        Obtém as agregações recomendadas para uma coluna.
        
        Args:
            column_name: Nome da coluna.
            
        Returns:
            List[str]: Lista de agregações recomendadas.
        """
        metadata = self.get_column_metadata(column_name)
        return metadata.aggregations if metadata else []
    
    def resolve_column_name(self, name_or_alias: str) -> Optional[str]:
        """
        Resolve o nome real de uma coluna a partir de um nome ou alias.
        
        Args:
            name_or_alias: Nome ou alias da coluna.
            
        Returns:
            Optional[str]: Nome real da coluna ou None se não encontrado.
        """
        if name_or_alias in self.columns:
            return name_or_alias
        
        name_lower = name_or_alias.lower()
        if name_lower in self._alias_lookup:
            return self._alias_lookup[name_lower]
        
        return None

class MetadataRegistry:
    """
    Registro global de metadados para datasets.
    
    Esta classe gerencia metadados para múltiplos datasets e fornece
    métodos para registrar, recuperar e utilizar esses metadados.
    """
    
    _instance = None
    
    def __new__(cls):
        """Implementação de Singleton para o registro."""
        if cls._instance is None:
            cls._instance = super(MetadataRegistry, cls).__new__(cls)
            cls._instance._datasets = {}
        return cls._instance
    
    def register_metadata(self, metadata: DatasetMetadata) -> None:
        """
        Registra metadados para um dataset.
        
        Args:
            metadata: Objeto DatasetMetadata.
        """
        self._datasets[metadata.name] = metadata
        logger.info(f"Metadados registrados para dataset: {metadata.name}")
    
    def register_from_dict(self, metadata_dict: Dict[str, Any]) -> None:
        """
        Registra metadados a partir de um dicionário.
        
        Args:
            metadata_dict: Dicionário com metadados.
        """
        metadata = DatasetMetadata.from_dict(metadata_dict)
        self.register_metadata(metadata)
    
    def register_from_json(self, json_str: str) -> None:
        """
        Registra metadados a partir de uma string JSON.
        
        Args:
            json_str: String JSON com metadados.
        """
        metadata = DatasetMetadata.from_json(json_str)
        self.register_metadata(metadata)
    
    def register_from_file(self, file_path: str) -> None:
        """
        Registra metadados a partir de um arquivo JSON.
        
        Args:
            file_path: Caminho para o arquivo JSON.
        """
        metadata = DatasetMetadata.from_file(file_path)
        self.register_metadata(metadata)
    
    def get_metadata(self, dataset_name: str) -> Optional[DatasetMetadata]:
        """
        Obtém metadados para um dataset específico.
        
        Args:
            dataset_name: Nome do dataset.
            
        Returns:
            Optional[DatasetMetadata]: Metadados do dataset ou None se não encontrado.
        """
        return self._datasets.get(dataset_name)
    
    def remove_metadata(self, dataset_name: str) -> bool:
        """
        Remove metadados para um dataset.
        
        Args:
            dataset_name: Nome do dataset.
            
        Returns:
            bool: True se removido com sucesso, False caso contrário.
        """
        if dataset_name in self._datasets:
            del self._datasets[dataset_name]
            logger.info(f"Metadados removidos para dataset: {dataset_name}")
            return True
        return False
    
    def list_datasets(self) -> List[str]:
        """
        Lista todos os datasets registrados.
        
        Returns:
            List[str]: Lista de nomes de datasets.
        """
        return list(self._datasets.keys())
    
    def clear(self) -> None:
        """Remove todos os metadados registrados."""
        self._datasets.clear()
        logger.info("Registro de metadados limpo")