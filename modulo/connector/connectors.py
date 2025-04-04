import os
import glob
import json
import pandas as pd
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
from typing import Optional, Dict, List, Any

from connector.metadata import ColumnMetadata, DatasetMetadata, MetadataRegistry

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("connector")


class DataConnectionException(Exception):
    """Exceção para problemas de conexão com fontes de dados."""
    pass


class DataReadException(Exception):
    """Exceção para problemas de leitura de dados."""
    pass


class ConfigurationException(Exception):
    """Exceção para problemas com configurações."""
    pass


class DataSourceConfig:
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
        self.source_id = source_id
        self.source_type = source_type
        self.params = params
        
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
    
    @classmethod
    def from_json(cls, json_str: str) -> 'DataSourceConfig':
        """
        Cria uma instância de configuração a partir de uma string JSON.
        
        Args:
            json_str: String JSON com configurações.
            
        Returns:
            DataSourceConfig: Nova instância de configuração.
        """
        try:
            config_dict = json.loads(json_str)
            return cls.from_dict(config_dict)
        except json.JSONDecodeError as e:
            raise ConfigurationException(f"Erro ao decodificar JSON: {str(e)}")
        
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


class DataConnector(ABC):
    """
    Interface base para todos os conectores de dados.
    """
    
    @abstractmethod
    def connect(self) -> None:
        """
        Estabelece conexão com a fonte de dados.
        
        Raises:
            DataConnectionException: Se a conexão falhar.
        """
        pass
    
    @abstractmethod
    def read_data(self, query: Optional[str] = None) -> pd.DataFrame:
        """
        Lê dados da fonte conforme a query especificada.
        
        Args:
            query: Consulta para filtrar/transformar os dados.
            
        Returns:
            pd.DataFrame: DataFrame com os dados lidos.
            
        Raises:
            DataReadException: Se a leitura falhar.
        """
        pass
    
    @abstractmethod
    def close(self) -> None:
        """
        Fecha a conexão com a fonte de dados.
        
        Raises:
            DataConnectionException: Se o fechamento da conexão falhar.
        """
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """
        Verifica se a conexão está ativa.
        
        Returns:
            bool: True se conectado, False caso contrário.
        """
        pass


class CsvConnector(DataConnector):
    """
    Conector CSV com suporte a metadados.
    
    Estende o CsvConnector padrão para utilizar informações de metadados
    para melhorar a interpretação e transformação dos dados.
    Suporta a leitura de um diretório contendo múltiplos arquivos CSV.
    """
    
    def __init__(self, config: Union[DataSourceConfig]):
        """
        Inicializa o conector.
        
        Args:
            config: Configuração da fonte de dados.
        """
        self.config = config
        self.data = None
        self._connected = False
        self.is_directory = False
        self.csv_files = []
        self.dataframes = {}
        
        # Validação de parâmetros obrigatórios
        if 'path' not in self.config.params:
            raise ConfigurationException("Parâmetro 'path' é obrigatório para fontes CSV")

    def connect(self) -> None:
        """
        Carrega o arquivo CSV ou diretório de CSVs na memória.
        """
        try:
            path = self.config.params['path']
            delimiter = self.config.params.get('delimiter', ',')
            encoding = self.config.params.get('encoding', 'utf-8')
            
            # Verifica se o caminho é um diretório
            if os.path.isdir(path):
                self.is_directory = True
                pattern = self.config.params.get('pattern', '*.csv')
                logger.info(f"Conectando ao diretório de CSVs: {path} com padrão {pattern}")
                
                # Lista todos os arquivos CSV no diretório
                self.csv_files = glob.glob(os.path.join(path, pattern))
                
                if not self.csv_files:
                    logger.warning(f"Nenhum arquivo CSV encontrado no diretório: {path}")
                    self._connected = False
                    return
                
                # Carrega cada arquivo CSV em um DataFrame separado
                for csv_file in self.csv_files:
                    try:
                        file_name = os.path.basename(csv_file)
                        logger.info(f"Carregando arquivo CSV: {file_name}")
                        
                        df = pd.read_csv(
                            csv_file,
                            delimiter=delimiter,
                            encoding=encoding
                        )
                        
                        # Aplica transformações baseadas em metadados para cada DataFrame
                        if hasattr(self.config, 'metadata') and self.config.metadata:
                            df = self._apply_metadata_transformations(df)
                        
                        self.dataframes[file_name] = df
                        
                    except Exception as e:
                        logger.error(f"Erro ao carregar arquivo CSV {file_name}: {str(e)}")
                
                # Se pelo menos um arquivo foi carregado com sucesso, consideramos conectado
                if self.dataframes:
                    self._connected = True
                    
                    # Concatena todos os DataFrames para consultas simples (sem joins)
                    # Esta é uma abordagem simples que pode ser refinada depois
                    if self.config.params.get('auto_concat', True):
                        try:
                            self.data = pd.concat(self.dataframes.values(), ignore_index=True)
                            logger.info(f"DataFrames concatenados com sucesso. Total de {len(self.data)} linhas.")
                        except Exception as e:
                            logger.warning(f"Não foi possível concatenar os DataFrames: {str(e)}")
                            # Usa o primeiro DataFrame como fallback
                            self.data = next(iter(self.dataframes.values()))
                else:
                    self._connected = False
                
            else:
                # Comportamento original para um único arquivo
                logger.info(f"Conectando ao CSV: {path}")
                self.data = pd.read_csv(
                    path, 
                    delimiter=delimiter, 
                    encoding=encoding
                )
                self._connected = True
                logger.info(f"Conectado com sucesso ao CSV: {path}")
                
                # Se conectou com sucesso e tem metadados, aplica transformações
                if self._connected and self.data is not None and hasattr(self.config, 'metadata') and self.config.metadata:
                    self.data = self._apply_metadata_transformations(self.data)
                
        except Exception as e:
            self._connected = False
            error_msg = f"Erro ao conectar com CSV {self.config.params.get('path')}: {str(e)}"
            logger.error(error_msg)
            raise DataConnectionException(error_msg) from e

    def _apply_metadata_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica transformações baseadas em metadados aos dados carregados.
        
        Args:
            df: DataFrame a ser transformado
            
        Returns:
            pd.DataFrame: DataFrame transformado
        """
        if df is None or not hasattr(self.config, 'metadata') or not self.config.metadata:
            return df
        
        result_df = df.copy()
        metadata = self.config.metadata
        
        # Aplica conversões de tipo para cada coluna com metadados
        for column_name, column_metadata in metadata.columns.items():
            if column_name in result_df.columns and column_metadata.data_type:
                try:
                    # Conversão baseada no tipo definido nos metadados
                    result_df = self._convert_column_type(result_df, column_name, column_metadata)
                except Exception as e:
                    logger.warning(f"Erro ao converter coluna {column_name}: {str(e)}")
        
        return result_df

    def _convert_column_type(self, df: pd.DataFrame, column_name: str, metadata: ColumnMetadata) -> pd.DataFrame:
        """
        Converte uma coluna para o tipo especificado nos metadados.
        
        Args:
            df: DataFrame contendo a coluna
            column_name: Nome da coluna a converter
            metadata: Metadados da coluna
            
        Returns:
            pd.DataFrame: DataFrame com a coluna convertida
        """
        result_df = df.copy()
        data_type = metadata.data_type
        format_str = metadata.format
        
        try:
            # Conversão de acordo com o tipo
            if data_type == 'int':
                result_df[column_name] = pd.to_numeric(result_df[column_name], errors='coerce').astype('Int64')
                logger.info(f"Coluna {column_name} convertida para inteiro")
                
            elif data_type == 'float':
                result_df[column_name] = pd.to_numeric(result_df[column_name], errors='coerce')
                logger.info(f"Coluna {column_name} convertida para float")
                
            elif data_type == 'date':
                result_df[column_name] = pd.to_datetime(result_df[column_name], format=format_str, errors='coerce')
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
                
                result_df[column_name] = result_df[column_name].apply(to_bool)
                logger.info(f"Coluna {column_name} convertida para booleano")
                
        except Exception as e:
            logger.warning(f"Erro ao converter coluna {column_name} para {data_type}: {str(e)}")
        
        return result_df

    def read_data(self, query: Optional[str] = None) -> pd.DataFrame:
        """
        Lê dados do CSV ou diretório de CSVs, opcionalmente aplicando uma consulta SQL.
        
        Args:
            query: Consulta SQL opcional para filtrar ou transformar os dados.
            
        Returns:
            pd.DataFrame: DataFrame com os dados resultantes.
        """
        if not self._connected:
            raise DataConnectionException("Não conectado à fonte de dados. Chame connect() primeiro.")
            
        try:
            # Caso mais simples: sem query retorna todos os dados (já concatenados)
            if not query:
                if self.is_directory and self.config.params.get('return_dict', False):
                    # Retorna um dicionário de DataFrames para processamento avançado
                    return self.dataframes
                return self.data.copy() if self.data is not None else pd.DataFrame()
            
            # Adapta a query com metadados se necessário
            if hasattr(self.config, 'metadata') and self.config.metadata:
                query = self._adapt_query_with_metadata(query)
            
            # Se for um diretório, precisamos de lógica especial para consultas
            if self.is_directory and self.dataframes:
                return self._execute_query_on_directory(query)
            
            # Comportamento padrão para um único DataFrame
            import sqlite3
            
            # Criamos uma conexão SQLite em memória
            conn = sqlite3.connect(':memory:')
            
            # Registramos o DataFrame como uma tabela temporária
            table_name = f"csv_data_{self.config.source_id}"
            if self.data is not None:
                self.data.to_sql(table_name, conn, if_exists='replace', index=False)
            
                # Substituímos referências à tabela na query
                modified_query = query.replace("FROM csv", f"FROM {table_name}")
                
                # Executamos a query
                result = pd.read_sql_query(modified_query, conn)
                conn.close()
                
                return result
            else:
                return pd.DataFrame()
            
        except Exception as e:
            error_msg = f"Erro ao ler dados do CSV: {str(e)}"
            logger.error(error_msg)
            raise DataReadException(error_msg) from e

    def _execute_query_on_directory(self, query: str) -> pd.DataFrame:
        """
        Executa uma consulta SQL em um diretório de arquivos CSV.
        
        Args:
            query: Consulta SQL a ser executada.
            
        Returns:
            pd.DataFrame: Resultado da consulta.
        """
        import sqlite3
        
        # Criamos uma conexão SQLite em memória
        conn = sqlite3.connect(':memory:')
        
        # Registramos cada DataFrame como uma tabela temporária separada
        for file_name, df in self.dataframes.items():
            # Remove a extensão e caracteres especiais para criar nomes de tabela válidos
            table_name = os.path.splitext(file_name)[0]
            table_name = ''.join(c if c.isalnum() else '_' for c in table_name)
            df.to_sql(table_name, conn, if_exists='replace', index=False)
            logger.info(f"Registrado arquivo {file_name} como tabela {table_name}")
        
        # Registramos também o DataFrame concatenado para consultas simples
        if self.data is not None:
            combined_table = f"csv_data_{self.config.source_id}"
            self.data.to_sql(combined_table, conn, if_exists='replace', index=False)
            
            # Substituímos referências genéricas à tabela na query
            modified_query = query.replace("FROM csv", f"FROM {combined_table}")
        else:
            modified_query = query
        
        try:
            # Executamos a query
            logger.info(f"Executando query: {modified_query}")
            result = pd.read_sql_query(modified_query, conn)
            return result
        except Exception as e:
            logger.error(f"Erro ao executar query: {str(e)}")
            # Tenta inferir os nomes das tabelas na query
            error_msg = f"Erro ao executar query. Certifique-se de usar os nomes corretos das tabelas: {', '.join(self.dataframes.keys())}"
            raise DataReadException(error_msg) from e
        finally:
            conn.close()

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

    def close(self) -> None:
        """
        Libera recursos. Para CSV, apenas limpa a referência aos dados.
        """
        self.data = None
        self.dataframes = {}
        self.csv_files = []
        self._connected = False
        logger.info(f"Conexão CSV fechada: {self.config.params.get('path')}")

    def is_connected(self) -> bool:
        """
        Verifica se o conector está ativo.
        
        Returns:
            bool: True se conectado, False caso contrário.
        """
        if self.is_directory:
            return self._connected and bool(self.dataframes)
        else:
            return self._connected and self.data is not None

    def get_available_tables(self) -> List[str]:
        """
        Retorna uma lista de tabelas disponíveis (nomes de arquivos) quando estiver no modo diretório.
        
        Returns:
            List[str]: Lista de nomes de arquivos/tabelas disponíveis
        """
        if not self.is_directory:
            return []
        
        return list(self.dataframes.keys())


class PostgresConnector(DataConnector):
    """
    Conector para bancos de dados PostgreSQL.
    
    Attributes:
        config (DataSourceConfig): Configuração da fonte de dados.
        connection: Conexão com o banco de dados.
    """
    
    def __init__(self, config: DataSourceConfig):
        """
        Inicializa um novo conector PostgreSQL.
        
        Args:
            config: Configuração da fonte de dados.
        """
        self.config = config
        self.connection = None
        
        # Validação de parâmetros obrigatórios
        required_params = ['host', 'database', 'username', 'password']
        missing_params = [param for param in required_params if param not in self.config.params]
        
        if missing_params:
            raise ConfigurationException(
                f"Parâmetros obrigatórios ausentes para PostgreSQL: {', '.join(missing_params)}"
            )
    
    def connect(self) -> None:
        """
        Estabelece conexão com o banco PostgreSQL.
        """
        try:
            import psycopg2
            
            host = self.config.params['host']
            database = self.config.params['database']
            username = self.config.params['username']
            password = self.config.params['password']
            port = self.config.params.get('port', 5432)
            
            logger.info(f"Conectando ao PostgreSQL: {host}/{database}")
            
            self.connection = psycopg2.connect(
                host=host,
                port=port,
                database=database,
                user=username,
                password=password
            )
            
            logger.info(f"Conectado com sucesso ao PostgreSQL: {host}/{database}")
            
        except ImportError:
            error_msg = "Módulo psycopg2 não encontrado. Instale com: pip install psycopg2-binary"
            logger.error(error_msg)
            raise DataConnectionException(error_msg)
        except Exception as e:
            error_msg = f"Erro ao conectar com PostgreSQL: {str(e)}"
            logger.error(error_msg)
            raise DataConnectionException(error_msg) from e
    
    def read_data(self, query: str) -> pd.DataFrame:
        """
        Executa uma consulta SQL no banco PostgreSQL.
        
        Args:
            query: Consulta SQL a ser executada.
            
        Returns:
            pd.DataFrame: DataFrame com os resultados da consulta.
        """
        if not self.is_connected():
            raise DataConnectionException("Não conectado ao banco de dados. Chame connect() primeiro.")
            
        if not query:
            raise DataReadException("Query SQL é obrigatória para conectores PostgreSQL")
            
        try:
            return pd.read_sql_query(query, self.connection)
        except Exception as e:
            error_msg = f"Erro ao executar query no PostgreSQL: {str(e)}"
            logger.error(error_msg)
            raise DataReadException(error_msg) from e
    
    def close(self) -> None:
        """
        Fecha a conexão com o banco de dados.
        """
        if self.connection:
            self.connection.close()
            self.connection = None
            logger.info(f"Conexão PostgreSQL fechada: {self.config.params.get('host')}/{self.config.params.get('database')}")
    
    def is_connected(self) -> bool:
        """
        Verifica se a conexão está ativa.
        
        Returns:
            bool: True se conectado, False caso contrário.
        """
        if not self.connection:
            return False
            
        try:
            # Verifica se a conexão está ativa com uma consulta simples
            cursor = self.connection.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
            return True
        except Exception:
            return False


class DuckDBCsvConnector(DataConnector):
    """
    Conector DuckDB com suporte a metadados.
    
    Este conector utiliza DuckDB para processar consultas SQL em arquivos CSV
    de forma eficiente, com suporte adicional para metadados de colunas.
    Suporta a leitura de um diretório contendo múltiplos arquivos CSV.
    
    Attributes:
        config: Configuração do conector.
        connection: Conexão com o DuckDB.
        table_name: Nome da tabela no DuckDB.
        column_mapping: Mapeamento entre aliases e nomes reais de colunas.
        is_directory: Flag indicando se o caminho é um diretório.
        csv_files: Lista de arquivos CSV no diretório.
        tables: Dicionário de nomes de tabelas registradas.
    """
    
    def __init__(self, config: Union[DataSourceConfig]):
        """
        Inicializa o conector.
        
        Args:
            config: Configuração do conector.
        """
        self.config = config
        self.connection = None
        self.table_name = f"csv_data_{self.config.source_id}"
        self.column_mapping = {}
        self.is_directory = False
        self.csv_files = []
        self.tables = {}
        
        # Validação de parâmetros obrigatórios
        if 'path' not in self.config.params:
            raise ConfigurationException("Parâmetro 'path' é obrigatório para fontes CSV")
    
    def connect(self) -> None:
        """
        Estabelece conexão com o DuckDB e registra o arquivo CSV ou diretório como tabelas.
        """
        try:
            import duckdb
            
            # Inicializa a conexão DuckDB
            self.connection = duckdb.connect(database=':memory:')
            
            path = self.config.params['path']
            
            # Verifica se o caminho é um diretório
            if os.path.isdir(path):
                self.is_directory = True
                pattern = self.config.params.get('pattern', '*.csv')
                logger.info(f"Conectando ao diretório de CSVs via DuckDB: {path} com padrão {pattern}")
                
                # Lista todos os arquivos CSV no diretório
                self.csv_files = glob.glob(os.path.join(path, pattern))
                
                if not self.csv_files:
                    logger.warning(f"Nenhum arquivo CSV encontrado no diretório: {path}")
                    return
                
                # Determina parâmetros para leitura dos CSVs
                delim = self.config.params.get('delim', 
                        self.config.params.get('sep', 
                        self.config.params.get('delimiter', ',')))
                
                has_header = self.config.params.get('header', True)
                auto_detect = self.config.params.get('auto_detect', True)
                
                # Registra cada arquivo CSV como uma view/tabela no DuckDB
                for csv_file in self.csv_files:
                    try:
                        file_name = os.path.basename(csv_file)
                        # Remove a extensão e caracteres especiais para criar nomes de tabela válidos
                        table_name = os.path.splitext(file_name)[0]
                        table_name = ''.join(c if c.isalnum() else '_' for c in table_name)
                        
                        # Constrói a query para criar a view
                        query_parts = [f"CREATE VIEW {table_name} AS SELECT * FROM read_csv('{csv_file}'"]
                        params = []
                        
                        params.append(f"delim='{delim}'")
                        params.append(f"header={str(has_header).lower()}")
                        params.append(f"auto_detect={str(auto_detect).lower()}")
                        
                        if params:
                            query_parts.append(", " + ", ".join(params))
                        
                        query_parts.append(")")
                        create_query = "".join(query_parts)
                        
                        logger.info(f"Registrando arquivo {file_name} como tabela {table_name}")
                        logger.debug(f"Query: {create_query}")
                        
                        self.connection.execute(create_query)
                        self.tables[file_name] = table_name
                        
                    except Exception as e:
                        logger.error(f"Erro ao registrar arquivo CSV {file_name}: {str(e)}")
                
                # Cria uma view combinada se solicitado
                if self.config.params.get('create_combined_view', True) and self.tables:
                    try:
                        # Seleciona o primeiro arquivo para obter o esquema
                        first_table = next(iter(self.tables.values()))
                        schema_query = f"SELECT * FROM {first_table} LIMIT 0"
                        schema_df = self.connection.execute(schema_query).fetchdf()
                        
                        # Cria uma query UNION ALL para todas as tabelas
                        union_parts = []
                        for table_name in self.tables.values():
                            # Verifica se a tabela tem as mesmas colunas
                            try:
                                columns_query = f"SELECT * FROM {table_name} LIMIT 0"
                                table_columns = self.connection.execute(columns_query).fetchdf().columns
                                
                                # Adiciona apenas tabelas com estrutura compatível
                                if set(schema_df.columns) == set(table_columns):
                                    union_parts.append(f"SELECT * FROM {table_name}")
                                else:
                                    logger.warning(f"Tabela {table_name} ignorada na visão combinada devido a diferenças de esquema")
                            except:
                                logger.warning(f"Erro ao verificar esquema da tabela {table_name}")
                        
                        if union_parts:
                            # Cria a visão combinada
                            combined_query = f"CREATE VIEW {self.table_name} AS {' UNION ALL '.join(union_parts)}"
                            self.connection.execute(combined_query)
                            logger.info(f"Visão combinada criada: {self.table_name}")
                        
                    except Exception as e:
                        logger.warning(f"Não foi possível criar a visão combinada: {str(e)}")
                
            else:
                # Comportamento original para um único arquivo
                if not os.path.exists(path):
                    # Tenta encontrar o arquivo no diretório atual
                    current_dir = os.getcwd()
                    base_filename = os.path.basename(path)
                    alternative_path = os.path.join(current_dir, base_filename)
                    
                    if os.path.exists(alternative_path):
                        logger.info(f"Arquivo não encontrado em {path}, usando alternativa: {alternative_path}")
                        path = alternative_path
                    else:
                        logger.warning(f"Arquivo CSV não encontrado: {path}")
                        return
                
                logger.info(f"Conectando ao CSV via DuckDB: {path}")
                
                # Determina os parâmetros
                delim = self.config.params.get('delim', 
                        self.config.params.get('sep', 
                        self.config.params.get('delimiter', ',')))
                
                has_header = self.config.params.get('header', True)
                auto_detect = self.config.params.get('auto_detect', True)
                
                # Constrói a query para criar a view
                query_parts = [f"CREATE VIEW {self.table_name} AS SELECT * FROM read_csv('{path}'"]
                params = []
                
                params.append(f"delim='{delim}'")
                params.append(f"header={str(has_header).lower()}")
                params.append(f"auto_detect={str(auto_detect).lower()}")
                
                if params:
                    query_parts.append(", " + ", ".join(params))
                
                query_parts.append(")")
                create_query = "".join(query_parts)
                
                logger.info(f"Query para criação da view DuckDB: {create_query}")
                self.connection.execute(create_query)
                
                # Registra o nome da tabela
                self.tables[os.path.basename(path)] = self.table_name
            
            # Obtém as colunas para mapeamento
            self._create_column_mapping()
            
            # Verifica a estrutura das tabelas registradas
            self._log_tables_schema()
            
        except ImportError:
            error_msg = "Módulo duckdb não encontrado. Instale com: pip install duckdb"
            logger.error(error_msg)
            raise DataConnectionException(error_msg)
        except Exception as e:
            error_msg = f"Erro ao conectar com CSV via DuckDB: {str(e)}"
            logger.error(error_msg)
            raise DataConnectionException(error_msg) from e
    
    def _create_column_mapping(self) -> None:
        """
        Cria um mapeamento entre aliases e nomes reais de colunas.
        """
        self.column_mapping = {}
        
        # Se não houver tabelas registradas, não há o que mapear
        if not self.tables:
            return
        
        # Usa a primeira tabela para obter as colunas
        try:
            first_table = next(iter(self.tables.values()))
            query = f"SELECT * FROM {first_table} LIMIT 0"
            columns_df = self.connection.execute(query).fetchdf()
            columns = columns_df.columns
            
            # Se temos metadados de colunas, usamos os aliases definidos
            if hasattr(self.config, 'metadata') and self.config.metadata:
                for col_name, metadata in self.config.metadata.columns.items():
                    if col_name in columns:
                        for alias in metadata.alias:
                            self.column_mapping[alias.lower()] = col_name
                
                logger.info(f"Mapeamento de colunas criado a partir de metadados: {self.column_mapping}")
            else:
                # Caso contrário, usamos a abordagem heurística
                lower_cols = [col.lower() for col in columns]
                
                # Mapeia nomes genéricos para colunas reais
                generic_mappings = {
                    'date': ['date', 'data', 'dt', 'dia', 'mes', 'ano', 'data_venda', 'data_compra'],
                    'revenue': ['revenue', 'receita', 'valor', 'venda', 'montante', 'faturamento'],
                    'profit': ['profit', 'lucro', 'margem', 'ganho', 'resultado'],
                    'quantity': ['quantity', 'quantidade', 'qtde', 'qtd', 'volume', 'unidades'],
                    'id': ['id', 'codigo', 'code', 'identificador', 'chave'],
                    'product': ['product', 'produto', 'item', 'mercadoria'],
                    'customer': ['customer', 'cliente', 'comprador', 'consumidor']
                }
                
                # Cria o mapeamento
                for generic, options in generic_mappings.items():
                    for option in options:
                        for i, col_lower in enumerate(lower_cols):
                            if option in col_lower:
                                self.column_mapping[generic] = columns[i]
                                break
                        if generic in self.column_mapping:
                            break
                
                logger.info(f"Mapeamento de colunas criado por heurística: {self.column_mapping}")
        except Exception as e:
            logger.warning(f"Não foi possível criar o mapeamento de colunas: {str(e)}")
    
    def _log_tables_schema(self) -> None:
        """
        Registra informações sobre o esquema das tabelas para debug.
        """
        for file_name, table_name in self.tables.items():
            try:
                schema_info = self.connection.execute(f"DESCRIBE {table_name}").fetchdf()
                logger.info(f"Esquema da tabela {table_name} ({file_name}):")
                for _, row in schema_info.iterrows():
                    logger.info(f"  {row['column_name']} - {row['column_type']}")
            except Exception as e:
                logger.warning(f"Não foi possível obter o esquema da tabela {table_name}: {str(e)}")
    
    def read_data(self, query: Optional[str] = None) -> pd.DataFrame:
        """
        Lê dados do CSV ou diretório de CSVs, opcionalmente aplicando uma consulta SQL.
        
        Args:
            query: Consulta SQL opcional.
            
        Returns:
            pd.DataFrame: DataFrame com os resultados.
        """
        if not self.is_connected():
            raise DataConnectionException("Não conectado à fonte de dados. Chame connect() primeiro.")
            
        try:
            # Se não houver query específica, seleciona todos os dados da tabela principal
            if not query:
                if self.is_directory and self.config.params.get('return_dict', False):
                    # Retorna um dicionário de DataFrames para cada arquivo
                    result = {}
                    for file_name, table_name in self.tables.items():
                        try:
                            df = self.connection.execute(f"SELECT * FROM {table_name}").fetchdf()
                            result[file_name] = df
                        except Exception as e:
                            logger.warning(f"Erro ao ler tabela {table_name}: {str(e)}")
                    return result
                
                # Usa a tabela combinada ou a única tabela disponível
                table_to_query = self.table_name if self.table_name in self._get_all_tables() else next(iter(self.tables.values()), None)
                
                if table_to_query:
                    query = f"SELECT * FROM {table_to_query}"
                else:
                    return pd.DataFrame()
            else:
                # Adapta a query usando metadados e substituições de tabela
                query = self._adapt_query(query)
            
            logger.info(f"Executando query: {query}")
            
            # Executa a query
            try:
                result = self.connection.execute(query).fetchdf()
                return result
            except Exception as query_error:
                logger.warning(f"Erro na query: {str(query_error)}. Mostrando tabelas disponíveis.")
                
                # Lista as tabelas disponíveis para ajudar o usuário
                available_tables = self._get_all_tables()
                error_msg = (f"Erro ao executar query: {str(query_error)}. "
                            f"Tabelas disponíveis: {', '.join(available_tables)}")
                raise DataReadException(error_msg) from query_error
            
        except Exception as e:
            if isinstance(e, DataReadException):
                raise e
            
            error_msg = f"Erro ao ler dados do CSV via DuckDB: {str(e)}"
            logger.error(error_msg)
            
            # Tenta fornecer um DataFrame vazio em vez de falhar
            try:
                return pd.DataFrame()
            except:
                raise DataReadException(error_msg) from e
    
    def _get_all_tables(self) -> List[str]:
        """
        Retorna todas as tabelas e views disponíveis no DuckDB.
        
        Returns:
            List[str]: Lista de nomes de tabelas/views
        """
        try:
            tables_df = self.connection.execute("SHOW TABLES").fetchdf()
            if 'name' in tables_df.columns:
                return tables_df['name'].tolist()
            return []
        except Exception as e:
            logger.warning(f"Erro ao listar tabelas: {str(e)}")
            return list(self.tables.values())

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
    
    def close(self) -> None:
        """
        Fecha a conexão com o DuckDB.
        """
        if self.connection:
            try:
                # Tenta remover a view/tabela antes de fechar
                try:
                    self.connection.execute(f"DROP VIEW IF EXISTS {self.table_name}")
                except Exception as drop_error:
                    logger.warning(f"Não foi possível remover a view: {str(drop_error)}")
                
                # Fecha a conexão
                self.connection.close()
            except Exception as e:
                logger.warning(f"Erro ao fechar conexão DuckDB: {str(e)}")
            finally:
                self.connection = None
                logger.info(f"Conexão DuckDB fechada para CSV: {self.config.params.get('path')}")
    
    def is_connected(self) -> bool:
        """
        Verifica se o conector está ativo.
        
        Returns:
            bool: True se conectado, False caso contrário.
        """
        if not self.connection:
            return False
            
        try:
            # Verifica se a conexão está ativa
            self.connection.execute("SELECT 1")
            return True
        except Exception:
            return False
    
    def get_schema(self) -> pd.DataFrame:
        """
        Retorna o esquema (estrutura) do arquivo CSV.
        
        Returns:
            pd.DataFrame: DataFrame com informações de esquema.
        """
        if not self.is_connected():
            raise DataConnectionException("Não conectado à fonte de dados. Chame connect() primeiro.")
            
        try:
            # Obtém informações sobre o esquema das colunas
            query = f"DESCRIBE {self.table_name}"
            return self.connection.execute(query).fetchdf()
        except Exception as e:
            logger.warning(f"Erro ao obter esquema: {str(e)}")
            
            # Alternativa: criar schema baseado em uma consulta simples
            try:
                query = f"SELECT * FROM {self.table_name} LIMIT 1"
                sample = self.connection.execute(query).fetchdf()
                
                schema_data = {
                    'column_name': sample.columns,
                    'column_type': [str(sample[col].dtype) for col in sample.columns]
                }
                return pd.DataFrame(schema_data)
            except Exception as alt_error:
                error_msg = f"Erro ao obter esquema alternativo: {str(alt_error)}"
                logger.error(error_msg)
                raise DataReadException(error_msg) from e
    
    def sample_data(self, num_rows: int = 5) -> pd.DataFrame:
        """
        Retorna uma amostra dos dados.
        
        Args:
            num_rows: Número de linhas a retornar.
            
        Returns:
            pd.DataFrame: DataFrame com a amostra.
        """
        if not self.is_connected():
            raise DataConnectionException("Não conectado à fonte de dados. Chame connect() primeiro.")
            
        try:
            query = f"SELECT * FROM {self.table_name} LIMIT {num_rows}"
            return self.connection.execute(query).fetchdf()
        except Exception as e:
            error_msg = f"Erro ao obter amostra de dados: {str(e)}"
            logger.error(error_msg)
            raise DataReadException(error_msg) from e


class DataConnectorFactory:
    """
    Factory de conectores com suporte a metadados.
    
    Estende a factory padrão para criar conectores que reconhecem e
    utilizam metadados de colunas.
    """
    _connectors = {
        'csv': CsvConnector,
        'postgres': PostgresConnector,
        'duckdb_csv': DuckDBCsvConnector
    }
    
    @classmethod
    def register_connector(cls, source_type: str, connector_class) -> None:
        """
        Registra um novo tipo de conector na factory.
        
        Args:
            source_type: Nome do tipo de fonte de dados.
            connector_class: Classe do conector a ser registrada.
        """
        if not issubclass(connector_class, DataConnector):
            raise TypeError(f"A classe deve implementar a interface DataConnector")
            
        cls._connectors[source_type] = connector_class
        logger.info(f"Conector registrado para tipo: {source_type}")
    
    @classmethod
    def create_connector(cls, config: Union[Dict, DataSourceConfig]) -> DataConnector:
        """
        Cria um conector com suporte a metadados.
        
        Args:
            config: Configuração da fonte de dados.
            
        Returns:
            DataConnector: Conector criado.
        """
        # Converte o config para MetadataEnabledDataSourceConfig se necessário
        config = DataSourceConfig.from_dict(config)
        
        # Cria o conector apropriado com base no tipo
        source_type = config.source_type
        
        if source_type not in cls._connectors:
            raise ConfigurationException(f"Tipo de conector não suportado: {source_type}")
            
        connector_class = cls._connectors[source_type]
        
        # Se o conector é o CSV, cria uma versão com metadados
        if source_type == 'csv' and connector_class == CsvConnector:
            return CsvConnector(config)
        
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
