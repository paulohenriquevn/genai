"""
Integração de Metadados com o Módulo Principal de Conectores
===========================================================

Este módulo integra o sistema de metadados com o módulo principal de conectores,
permitindo que os conectores base reconheçam e utilizem metadados de colunas.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, Union, List

# Importação do módulo principal de conectores
from connectors import (
    DataConnector,
    DataSourceConfig,
    DataConnectorFactory,
    DataConnectionException,
    DataReadException,
    ConfigurationException,
    CsvConnector,
    PostgresConnector
)

# Importação do módulo de metadados
from column_metadata import (
    DatasetMetadata,
    ColumnMetadata,
    MetadataRegistry
)

# Configuração de logging
logger = logging.getLogger("metadata_integration")

class MetadataEnabledDataSourceConfig(DataSourceConfig):
    """
    Configuração de fonte de dados com suporte a metadados.
    
    Estende a classe base DataSourceConfig para incluir informações
    de metadados sobre as colunas do dataset.
    
    Attributes:
        source_id (str): Identificador único da fonte de dados.
        source_type (str): Tipo da fonte de dados.
        params (Dict): Parâmetros específicos para o conector.
        metadata (Optional[DatasetMetadata]): Metadados do dataset.
    """
    
    def __init__(self, source_id: str, source_type: str, metadata: Optional[Union[Dict, DatasetMetadata]] = None, **params):
        """
        Inicializa a configuração com metadados.
        
        Args:
            source_id: Identificador único da fonte de dados.
            source_type: Tipo da fonte de dados.
            metadata: Metadados do dataset.
            **params: Parâmetros adicionais para o conector.
        """
        super().__init__(source_id, source_type, **params)
        
        # Processa os metadados
        if metadata is None:
            self.metadata = None
        elif isinstance(metadata, DatasetMetadata):
            self.metadata = metadata
        elif isinstance(metadata, dict):
            try:
                self.metadata = DatasetMetadata.from_dict(metadata)
            except Exception as e:
                logger.warning(f"Erro ao processar metadados: {str(e)}")
                self.metadata = None
        else:
            logger.warning(f"Formato de metadados não suportado: {type(metadata)}")
            self.metadata = None
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MetadataEnabledDataSourceConfig':
        """
        Cria uma instância a partir de um dicionário.
        
        Args:
            config_dict: Dicionário de configuração.
            
        Returns:
            MetadataEnabledDataSourceConfig: Nova instância.
        """
        source_id = config_dict.get('id')
        source_type = config_dict.get('type')
        metadata = config_dict.get('metadata')
        
        if not source_id:
            raise ConfigurationException("ID da fonte de dados não especificado")
        if not source_type:
            raise ConfigurationException("Tipo da fonte de dados não especificado")
            
        # Remove chaves especiais
        params = {k: v for k, v in config_dict.items() if k not in ('id', 'type', 'metadata')}
        
        return cls(source_id, source_type, metadata=metadata, **params)
    
    def resolve_column_name(self, name_or_alias: str) -> Optional[str]:
        """
        Resolve o nome real de uma coluna a partir de um nome ou alias.
        
        Args:
            name_or_alias: Nome ou alias da coluna.
            
        Returns:
            Optional[str]: Nome real da coluna ou None.
        """
        if self.metadata is None:
            return None
            
        return self.metadata.resolve_column_name(name_or_alias)
    
    def get_column_metadata(self, column_name: str) -> Optional[ColumnMetadata]:
        """
        Obtém metadados para uma coluna específica.
        
        Args:
            column_name: Nome da coluna.
            
        Returns:
            Optional[ColumnMetadata]: Metadados da coluna ou None.
        """
        if self.metadata is None:
            return None
            
        return self.metadata.get_column_metadata(column_name)
    
    def get_recommended_aggregations(self, column_name: str) -> List[str]:
        """
        Obtém as agregações recomendadas para uma coluna.
        
        Args:
            column_name: Nome da coluna.
            
        Returns:
            List[str]: Lista de agregações recomendadas.
        """
        if self.metadata is None:
            return []
            
        metadata = self.metadata.get_column_metadata(column_name)
        return metadata.aggregations if metadata else []
    
    def get_column_type(self, column_name: str) -> Optional[str]:
        """
        Obtém o tipo de dados de uma coluna.
        
        Args:
            column_name: Nome da coluna.
            
        Returns:
            Optional[str]: Tipo de dados da coluna ou None.
        """
        if self.metadata is None:
            return None
            
        metadata = self.metadata.get_column_metadata(column_name)
        return metadata.data_type if metadata else None
    
    def get_column_format(self, column_name: str) -> Optional[str]:
        """
        Obtém o formato de uma coluna.
        
        Args:
            column_name: Nome da coluna.
            
        Returns:
            Optional[str]: Formato da coluna ou None.
        """
        if self.metadata is None:
            return None
            
        metadata = self.metadata.get_column_metadata(column_name)
        return metadata.format if metadata else None


class MetadataEnabledDataConnectorFactory(DataConnectorFactory):
    """
    Factory de conectores com suporte a metadados.
    
    Estende a factory padrão para criar conectores que reconhecem e
    utilizam metadados de colunas.
    """
    
    @classmethod
    def create_connector(cls, config: Union[Dict, DataSourceConfig, MetadataEnabledDataSourceConfig]) -> DataConnector:
        """
        Cria um conector com suporte a metadados.
        
        Args:
            config: Configuração da fonte de dados.
            
        Returns:
            DataConnector: Conector criado.
        """
        # Converte o config para MetadataEnabledDataSourceConfig se necessário
        if isinstance(config, dict):
            config = MetadataEnabledDataSourceConfig.from_dict(config)
        elif isinstance(config, DataSourceConfig) and not isinstance(config, MetadataEnabledDataSourceConfig):
            # Cria uma versão com metadados mantendo os parâmetros originais
            config = MetadataEnabledDataSourceConfig(
                config.source_id,
                config.source_type,
                metadata=None,
                **config.params
            )
        
        # Cria o conector apropriado com base no tipo
        source_type = config.source_type
        
        if source_type not in cls._connectors:
            raise ConfigurationException(f"Tipo de conector não suportado: {source_type}")
            
        connector_class = cls._connectors[source_type]
        
        # Se o conector é o CSV, cria uma versão com metadados
        if source_type == 'csv' and connector_class == CsvConnector:
            return MetadataEnabledCsvConnector(config)
        
        # Para outros conectores, usa a classe original
        return connector_class(config)
    
    @classmethod
    def create_from_json(cls, json_config: str) -> Dict[str, DataConnector]:
        """
        Cria múltiplos conectores a partir de uma configuração JSON.
        
        Args:
            json_config: String JSON com configurações.
            
        Returns:
            Dict[str, DataConnector]: Dicionário com conectores.
        """
        try:
            config_data = json.loads(json_config)
            
            if 'data_sources' not in config_data:
                raise ConfigurationException("Formato de configuração inválido. Esperava 'data_sources' como chave principal.")
                
            sources_data = config_data['data_sources']
            
            # Processa metadados globais se existirem
            metadata_registry = MetadataRegistry()
            global_metadata = config_data.get('metadata', {})
            
            # Registra metadados de arquivos
            for file_path in global_metadata.get('files', []):
                try:
                    if os.path.exists(file_path):
                        metadata_registry.register_from_file(file_path)
                        logger.info(f"Metadados registrados do arquivo: {file_path}")
                except Exception as e:
                    logger.warning(f"Erro ao carregar metadados do arquivo {file_path}: {str(e)}")
            
            # Registra metadados definidos inline
            for metadata_dict in global_metadata.get('datasets', []):
                try:
                    metadata_registry.register_from_dict(metadata_dict)
                    logger.info(f"Metadados registrados para: {metadata_dict.get('name', 'desconhecido')}")
                except Exception as e:
                    logger.warning(f"Erro ao registrar metadados: {str(e)}")
            
            # Cria os conectores
            connectors = {}
            for source_config in sources_data:
                source_id = source_config.get('id')
                if not source_id:
                    raise ConfigurationException("Configuração de fonte sem ID")
                
                # Verifica se já tem metadados ou se precisa buscar do registro
                if 'metadata' not in source_config:
                    dataset_name = source_config.get('dataset_name', source_id)
                    metadata = metadata_registry.get_metadata(dataset_name)
                    if metadata:
                        source_config['metadata'] = metadata.to_dict()
                        logger.info(f"Metadados do registro aplicados à fonte {source_id}")
                
                # Cria o conector
                connector = cls.create_connector(source_config)
                connectors[source_id] = connector
                
            return connectors
                
        except json.JSONDecodeError as e:
            raise ConfigurationException(f"Erro ao decodificar JSON: {str(e)}")


class MetadataEnabledCsvConnector(CsvConnector):
    """
    Conector CSV com suporte a metadados.
    
    Estende o CsvConnector padrão para utilizar informações de metadados
    para melhorar a interpretação e transformação dos dados.
    """
    
    def __init__(self, config: Union[DataSourceConfig, MetadataEnabledDataSourceConfig]):
        """
        Inicializa o conector.
        
        Args:
            config: Configuração da fonte de dados.
        """
        super().__init__(config)
        
        # Assegura que self.config seja um MetadataEnabledDataSourceConfig
        if not isinstance(self.config, MetadataEnabledDataSourceConfig):
            # Converte para um config com metadados, mas sem metadados reais
            old_config = self.config
            self.config = MetadataEnabledDataSourceConfig(
                old_config.source_id,
                old_config.source_type,
                metadata=None,
                **old_config.params
            )
    
    def connect(self) -> None:
        """
        Conecta à fonte de dados CSV e aplica transformações baseadas em metadados.
        """
        # Usa a implementação base para conectar
        super().connect()
        
        # Se conectou com sucesso e tem metadados, aplica transformações
        if self._connected and self.data is not None and hasattr(self.config, 'metadata') and self.config.metadata:
            self._apply_metadata_transformations()
    
    def _apply_metadata_transformations(self) -> None:
        """
        Aplica transformações baseadas em metadados aos dados carregados.
        """
        if self.data is None or not hasattr(self.config, 'metadata') or not self.config.metadata:
            return
        
        metadata = self.config.metadata
        
        # Aplica conversões de tipo para cada coluna com metadados
        for column_name, column_metadata in metadata.columns.items():
            if column_name in self.data.columns and column_metadata.data_type:
                try:
                    # Conversão baseada no tipo definido nos metadados
                    self._convert_column_type(column_name, column_metadata)
                except Exception as e:
                    logger.warning(f"Erro ao converter coluna {column_name}: {str(e)}")
    
    def _convert_column_type(self, column_name: str, metadata: ColumnMetadata) -> None:
        """
        Converte uma coluna para o tipo especificado nos metadados.
        
        Args:
            column_name: Nome da coluna a converter.
            metadata: Metadados da coluna.
        """
        import pandas as pd
        
        if self.data is None:
            return
            
        data_type = metadata.data_type
        format_str = metadata.format
        
        try:
            # Conversão de acordo com o tipo
            if data_type == 'int':
                self.data[column_name] = pd.to_numeric(self.data[column_name], errors='coerce').astype('Int64')
                logger.info(f"Coluna {column_name} convertida para inteiro")
                
            elif data_type == 'float':
                self.data[column_name] = pd.to_numeric(self.data[column_name], errors='coerce')
                logger.info(f"Coluna {column_name} convertida para float")
                
            elif data_type == 'date':
                self.data[column_name] = pd.to_datetime(self.data[column_name], format=format_str, errors='coerce')
                logger.info(f"Coluna {column_name} convertida para data")
                
            elif data_type == 'bool':
                # Trata valores booleanos representados como strings
                true_values = ['true', 'yes', 'y', '1', 'sim', 's']
                false_values = ['false', 'no', 'n', '0', 'não', 'nao']
                
                def to_bool(x):
                    if isinstance(x, str):
                        x = x.lower()
                        if x in true_values:
                            return True
                        if x in false_values:
                            return False
                    return x
                
                self.data[column_name] = self.data[column_name].apply(to_bool)
                logger.info(f"Coluna {column_name} convertida para booleano")
                
        except Exception as e:
            logger.warning(f"Erro ao converter coluna {column_name} para {data_type}: {str(e)}")
    
    def read_data(self, query: Optional[str] = None) -> 'pd.DataFrame':
        """
        Lê dados, opcionalmente aplicando uma consulta SQL.
        
        Args:
            query: Consulta SQL opcional.
            
        Returns:
            pd.DataFrame: DataFrame com os dados resultantes.
        """
        # Se tiver uma query e metadados, adapta a query
        if query and hasattr(self.config, 'metadata') and self.config.metadata:
            adapted_query = self._adapt_query_with_metadata(query)
            return super().read_data(adapted_query)
        
        # Caso contrário, usa a implementação padrão
        return super().read_data(query)
    
    def _adapt_query_with_metadata(self, query: str) -> str:
        """
        Adapta uma consulta SQL usando informações de metadados.
        
        Args:
            query: Consulta SQL original.
            
        Returns:
            str: Consulta adaptada.
        """
        if not hasattr(self.config, 'metadata') or not self.config.metadata:
            return query
        
        metadata = self.config.metadata
        adapted_query = query
        
        # Substitui aliases por nomes reais de colunas
        for alias, real_name in getattr(metadata, '_alias_lookup', {}).items():
            # Uso de regex para substituição precisa
            import re
            pattern = r'(?<![a-zA-Z0-9_])' + re.escape(alias) + r'(?![a-zA-Z0-9_])'
            adapted_query = re.sub(pattern, real_name, adapted_query)
        
        logger.info(f"Query adaptada com metadados: {adapted_query}")
        return adapted_query


# Exemplo de utilização
if __name__ == "__main__":
    # Define metadados para um dataset
    metadata_json = """
    {
        "name": "vendas",
        "description": "Dados de vendas mensais",
        "columns": [
            {
                "name": "data",
                "description": "Data da venda",
                "data_type": "date",
                "format": "%Y-%m-%d",
                "alias": ["dt", "date", "data_venda"]
            },
            {
                "name": "valor",
                "description": "Valor da venda em reais",
                "data_type": "float",
                "alias": ["revenue", "montante", "receita"],
                "aggregations": ["sum", "avg", "min", "max"]
            },
            {
                "name": "quantidade",
                "description": "Quantidade de produtos vendidos",
                "data_type": "int",
                "alias": ["qty", "qtd", "quantity"]
            }
        ]
    }
    """
    
    # Cria um config com metadados
    config = MetadataEnabledDataSourceConfig(
        "vendas_perdidas_csv",
        "csv",
        metadata=json.loads(metadata_json),
        path="vendas_perdidas.csv",
        delimiter=","
    )
    
    # Cria o conector usando a factory
    connector = MetadataEnabledDataConnectorFactory.create_connector(config)
    
    try:
        # Conecta e lê os dados
        connector.connect()
        
        # Lê todos os dados (com as conversões de tipo aplicadas)
        all_data = connector.read_data()
        print(f"Total de registros: {len(all_data)}")
        print(f"Colunas: {list(all_data.columns)}")
        print(f"Tipos das colunas: {all_data.dtypes}")
        
        # Executa uma consulta usando aliases
        try:
            # Estas consultas são equivalentes graças aos metadados
            result1 = connector.read_data("SELECT SUM(ImpactoFinanceiro) FROM csv")
            print(f"\nSoma de ImpactoFinanceiro: {result1['SUM(ImpactoFinanceiro)'].iloc[0]}")
            
            result2 = connector.read_data("SELECT SUM(ImpactoFinanceiro) FROM csv")
            print(f"Soma de ImpactoFinanceiro: {result2['SUM(ImpactoFinanceiro)'].iloc[0]}")
        except Exception as e:
            print(f"Erro na consulta: {str(e)}")
        
        # Fecha o conector
        connector.close()
        
    except Exception as e:
        print(f"Erro: {str(e)}")