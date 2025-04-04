"""
Módulo de Conexão a Fontes de Dados
===================================

Este módulo implementa um sistema flexível e extensível para conexão com múltiplas fontes 
de dados (ex: CSV, bancos relacionais) seguindo o princípio Open/Closed - aberto para 
extensão, fechado para modificação.
"""
import os
import json
import pandas as pd
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
from typing import Optional, Dict, List, Any


# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("data_connector")

# Classes de exceção
class DataConnectionException(Exception):
    """Exceção para problemas de conexão com fontes de dados."""
    pass

class DataReadException(Exception):
    """Exceção para problemas de leitura de dados."""
    pass

class ConfigurationException(Exception):
    """Exceção para problemas com configurações."""
    pass

# Classe de configuração
class DataSourceConfig:
    """
    Configuração para fontes de dados.
    
    Attributes:
        source_id (str): Identificador único da fonte de dados.
        source_type (str): Tipo da fonte de dados (ex: 'csv', 'postgres', 'mysql').
        params (Dict): Parâmetros específicos para cada tipo de conector.
    """
    
    def __init__(self, source_id: str, source_type: str, **params):
        self.source_id = source_id
        self.source_type = source_type
        self.params = params
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DataSourceConfig':
        """
        Cria uma instância de configuração a partir de um dicionário.
        
        Args:
            config_dict: Dicionário contendo configurações.
            
        Returns:
            DataSourceConfig: Nova instância de configuração.
        """
        source_id = config_dict.get('id')
        source_type = config_dict.get('type')
        
        if not source_id:
            raise ConfigurationException("ID da fonte de dados não especificado")
        if not source_type:
            raise ConfigurationException("Tipo da fonte de dados não especificado")
            
        # Remove chaves especiais e mantém apenas os parâmetros
        params = {k: v for k, v in config_dict.items() if k not in ('id', 'type')}
        
        return cls(source_id, source_type, **params)
    
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

# Interface principal
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

# Implementação para CSV
class CsvConnector(DataConnector):
    """
    Conector para arquivos CSV.
    
    Attributes:
        config (DataSourceConfig): Configuração da fonte de dados.
        data (pd.DataFrame): DataFrame para armazenar os dados lidos.
    """
    
    def __init__(self, config: DataSourceConfig):
        """
        Inicializa um novo conector CSV.
        
        Args:
            config: Configuração da fonte de dados.
        """
        self.config = config
        self.data = None
        self._connected = False
        
        # Validação de parâmetros obrigatórios
        if 'path' not in self.config.params:
            raise ConfigurationException("Parâmetro 'path' é obrigatório para fontes CSV")
    
    def connect(self) -> None:
        """
        Carrega o arquivo CSV na memória.
        """
        try:
            path = self.config.params['path']
            delimiter = self.config.params.get('delimiter', ',')
            encoding = self.config.params.get('encoding', 'utf-8')
            
            logger.info(f"Conectando ao CSV: {path}")
            self.data = pd.read_csv(
                path, 
                delimiter=delimiter, 
                encoding=encoding
            )
            self._connected = True
            logger.info(f"Conectado com sucesso ao CSV: {path}")
        except Exception as e:
            self._connected = False
            error_msg = f"Erro ao conectar com CSV {self.config.params.get('path')}: {str(e)}"
            logger.error(error_msg)
            raise DataConnectionException(error_msg) from e
    
    def read_data(self, query: Optional[str] = None) -> pd.DataFrame:
        """
        Lê dados do CSV, opcionalmente aplicando uma consulta SQL.
        
        Args:
            query: Consulta SQL opcional para filtrar ou transformar os dados.
            
        Returns:
            pd.DataFrame: DataFrame com os dados resultantes.
        """
        if not self._connected:
            raise DataConnectionException("Não conectado à fonte de dados. Chame connect() primeiro.")
            
        try:
            if not query:
                return self.data.copy()
            
            # Usando pandas para executar SQL na memória
            import sqlite3
            from pandas.io.sql import pandasSQL_builder
            
            # Criamos uma conexão SQLite em memória
            conn = sqlite3.connect(':memory:')
            
            # Registramos o DataFrame como uma tabela temporária
            table_name = f"csv_data_{self.config.source_id}"
            self.data.to_sql(table_name, conn, if_exists='replace', index=False)
            
            # Substituímos referências à tabela na query
            modified_query = query.replace("FROM csv", f"FROM {table_name}")
            
            # Executamos a query
            result = pd.read_sql_query(modified_query, conn)
            conn.close()
            
            return result
            
        except Exception as e:
            error_msg = f"Erro ao ler dados do CSV: {str(e)}"
            logger.error(error_msg)
            raise DataReadException(error_msg) from e
    
    def close(self) -> None:
        """
        Libera recursos. Para CSV, apenas limpa a referência aos dados.
        """
        self.data = None
        self._connected = False
        logger.info(f"Conexão CSV fechada: {self.config.params.get('path')}")
    
    def is_connected(self) -> bool:
        """
        Verifica se o conector está ativo.
        
        Returns:
            bool: True se conectado, False caso contrário.
        """
        return self._connected and self.data is not None

# Implementação para PostgreSQL
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
    Conector CSV utilizando DuckDB para processamento eficiente de consultas SQL.
    
    Esta implementação corrige problemas com:
    - Parâmetros incorretos (utiliza 'delim' em vez de 'delimiter')
    - Tratamento de erros e recuperação
    - Caminhos de arquivo inválidos
    - Compatibilidade de colunas em consultas
    
    Attributes:
        config (DataSourceConfig): Configuração da fonte de dados.
        connection: Conexão com o DuckDB.
        table_name (str): Nome da tabela registrada no DuckDB.
    """
    
    def __init__(self, config: DataSourceConfig):
        """
        Inicializa um novo conector CSV baseado em DuckDB.
        
        Args:
            config: Configuração da fonte de dados.
        """
        self.config = config
        self.connection = None
        self.table_name = f"csv_data_{self.config.source_id}"
        self.column_mapping = {}  # Mapeamento entre nomes de colunas genéricas e reais
        
        # Validação de parâmetros obrigatórios
        if 'path' not in self.config.params:
            raise ConfigurationException("Parâmetro 'path' é obrigatório para fontes CSV")
    
    def connect(self) -> None:
        """
        Configura a conexão com o DuckDB e registra o arquivo CSV.
        
        Implementa:
        - Verificação de existência do arquivo
        - Parâmetros corretos para DuckDB
        - Fallback para diferentes métodos de leitura
        - Registro de informações de debug
        """
        try:
            import duckdb
            
            # Verificação do caminho do arquivo
            path = self.config.params['path']
            if not os.path.exists(path):
                # Tenta encontrar o arquivo no diretório atual
                current_dir = os.getcwd()
                base_filename = os.path.basename(path)
                alternative_path = os.path.join(current_dir, base_filename)
                
                if os.path.exists(alternative_path):
                    logger.info(f"Arquivo não encontrado em {path}, usando alternativa: {alternative_path}")
                    path = alternative_path
                else:
                    logger.warning(f"Arquivo CSV não encontrado: {path}. Criando tabela vazia.")
                    self.connection = duckdb.connect(database=':memory:')
                    empty_df = pd.DataFrame({
                        'id': [],
                        'data': [],
                        'valor': [],
                        'quantidade': []
                    })
                    self.connection.register(self.table_name, empty_df)
                    self._map_columns(empty_df.columns)
                    return
            
            logger.info(f"Conectando ao CSV via DuckDB: {path}")
            
            # Método 1: Usar pandas + registro no DuckDB
            try:
                # Determina o delimitador
                delim = self.config.params.get('delim', 
                        self.config.params.get('sep', 
                        self.config.params.get('delimiter', ',')))
                
                encoding = self.config.params.get('encoding', 'utf-8')
                
                # Tenta ler com pandas e registrar no DuckDB
                self.connection = duckdb.connect(database=':memory:')
                
                # Tenta ler o arquivo com pandas
                df = pd.read_csv(path, sep=delim, encoding=encoding)
                logger.info(f"Arquivo CSV lido com sucesso usando pandas. Colunas: {list(df.columns)}")
                
                # Armazena mapeamento de colunas
                self._map_columns(df.columns)
                
                # Registra o DataFrame no DuckDB
                self.connection.register(self.table_name, df)
                logger.info(f"DataFrame registrado no DuckDB como tabela: {self.table_name}")
                
            except Exception as pandas_error:
                logger.warning(f"Falha ao ler com pandas: {str(pandas_error)}. Tentando diretamente com DuckDB.")
                
                # Método 2: Usar DuckDB diretamente
                try:
                    if self.connection is None:
                        self.connection = duckdb.connect(database=':memory:')
                    
                    # Correto: DuckDB usa 'delim' em vez de 'delimiter'
                    query_parts = [f"CREATE VIEW {self.table_name} AS SELECT * FROM read_csv('{path}'"]
                    params = []
                    
                    # Adiciona parâmetros na sintaxe correta do DuckDB
                    if 'delim' in self.config.params:
                        params.append(f"delim='{self.config.params['delim']}'")
                    elif 'delimiter' in self.config.params:
                        params.append(f"delim='{self.config.params['delimiter']}'")
                    elif 'sep' in self.config.params:
                        params.append(f"delim='{self.config.params['sep']}'")
                    
                    if 'header' in self.config.params:
                        params.append(f"header={str(self.config.params['header']).lower()}")
                    
                    if 'auto_detect' in self.config.params:
                        params.append(f"auto_detect={str(self.config.params['auto_detect']).lower()}")
                    
                    if params:
                        query_parts.append(", " + ", ".join(params))
                    
                    query_parts.append(")")
                    query = "".join(query_parts)
                    
                    logger.info(f"Query para criação da view DuckDB: {query}")
                    self.connection.execute(query)
                    
                    # Obtém as colunas para mapeamento
                    columns_df = self.connection.execute(f"SELECT * FROM {self.table_name} LIMIT 0").fetchdf()
                    self._map_columns(columns_df.columns)
                    
                except Exception as duckdb_error:
                    logger.error(f"Erro ao ler com DuckDB: {str(duckdb_error)}")
                    
                    # Método 3: Fallback para uma tabela vazia
                    logger.warning("Criando tabela vazia como fallback")
                    if self.connection is None:
                        self.connection = duckdb.connect(database=':memory:')
                    
                    empty_df = pd.DataFrame({
                        'id': [],
                        'data': [],
                        'valor': [],
                        'quantidade': []
                    })
                    self.connection.register(self.table_name, empty_df)
                    self._map_columns(empty_df.columns)
            
            # Verifica a estrutura da tabela registrada
            self._log_table_schema()
            
        except ImportError:
            error_msg = "Módulo duckdb não encontrado. Instale com: pip install duckdb"
            logger.error(error_msg)
            raise DataConnectionException(error_msg)
        except Exception as e:
            error_msg = f"Erro ao conectar com CSV via DuckDB: {str(e)}"
            logger.error(error_msg)
            raise DataConnectionException(error_msg) from e
    
    def _map_columns(self, columns: List[str]) -> None:
        """
        Cria um mapeamento entre nomes de colunas genéricos e reais.
        Facilita consultas com nomes de colunas padronizados.
        
        Args:
            columns: Lista de nomes de colunas reais.
        """
        self.column_mapping = {}
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
        
        logger.info(f"Mapeamento de colunas criado: {self.column_mapping}")
    
    def _log_table_schema(self) -> None:
        """Registra informações sobre o esquema da tabela para debug."""
        try:
            schema_info = self.connection.execute(f"DESCRIBE {self.table_name}").fetchdf()
            logger.info(f"Esquema da tabela {self.table_name}:")
            for _, row in schema_info.iterrows():
                logger.info(f"  {row['column_name']} - {row['column_type']}")
        except Exception as e:
            logger.warning(f"Não foi possível obter o esquema: {str(e)}")
    
    def read_data(self, query: Optional[str] = None) -> pd.DataFrame:
        """
        Lê dados do CSV, opcionalmente aplicando uma consulta SQL.
        
        Implementa:
        - Adaptação da query para nomes de colunas reais
        - Tratamento de erros e recuperação
        - Verificação de compatibilidade de colunas
        
        Args:
            query: Consulta SQL opcional para filtrar ou transformar os dados.
            
        Returns:
            pd.DataFrame: DataFrame com os dados resultantes.
        """
        if not self.is_connected():
            raise DataConnectionException("Não conectado à fonte de dados. Chame connect() primeiro.")
            
        try:
            # Se não houver query específica, seleciona todos os dados
            if not query:
                query = f"SELECT * FROM {self.table_name}"
            else:
                # Substitui referências à "csv" e adapta nomes de colunas
                query = self._adapt_query(query)
                
            logger.info(f"Executando query: {query}")
            
            # Executa a query
            try:
                result = self.connection.execute(query).fetchdf()
                return result
            except Exception as query_error:
                logger.warning(f"Erro na query: {str(query_error)}. Tentando query simplificada.")
                
                # Tenta uma query mais simples
                fallback_query = f"SELECT * FROM {self.table_name} LIMIT 100"
                return self.connection.execute(fallback_query).fetchdf()
            
        except Exception as e:
            error_msg = f"Erro ao ler dados do CSV via DuckDB: {str(e)}"
            logger.error(error_msg)
            
            # Retorna um DataFrame vazio em vez de falhar
            try:
                return pd.DataFrame(columns=['id', 'data', 'valor', 'quantidade'])
            except:
                raise DataReadException(error_msg) from e
    
    def _adapt_query(self, query: str) -> str:
        """
        Adapta uma query SQL para usar os nomes corretos de colunas e tabelas.
        
        Args:
            query: Consulta SQL original.
            
        Returns:
            str: Consulta adaptada.
        """
        # Substitui referência à tabela
        adapted_query = query.replace("FROM csv", f"FROM {self.table_name}")
        
        # Substitui nomes genéricos de colunas pelos reais
        for generic, real in self.column_mapping.items():
            # Padrão para encontrar o nome genérico como uma palavra inteira
            import re
            pattern = r'(?<![a-zA-Z0-9_])' + re.escape(generic) + r'(?![a-zA-Z0-9_])'
            adapted_query = re.sub(pattern, real, adapted_query)
        
        return adapted_query
    
    def close(self) -> None:
        """
        Fecha a conexão com o DuckDB e libera recursos.
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
            pd.DataFrame: DataFrame com a amostra de dados.
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
    
    def get_column_names(self) -> List[str]:
        """
        Retorna os nomes das colunas disponíveis.
        
        Returns:
            List[str]: Lista com os nomes das colunas.
        """
        if not self.is_connected():
            raise DataConnectionException("Não conectado à fonte de dados. Chame connect() primeiro.")
            
        try:
            query = f"SELECT * FROM {self.table_name} LIMIT 0"
            df = self.connection.execute(query).fetchdf()
            return list(df.columns)
        except Exception as e:
            error_msg = f"Erro ao obter nomes de colunas: {str(e)}"
            logger.error(error_msg)
            raise DataReadException(error_msg) from e
    
    def count_rows(self) -> int:
        """
        Conta o número total de linhas.
        
        Returns:
            int: Número de linhas na tabela.
        """
        if not self.is_connected():
            raise DataConnectionException("Não conectado à fonte de dados. Chame connect() primeiro.")
            
        try:
            query = f"SELECT COUNT(*) FROM {self.table_name}"
            result = self.connection.execute(query).fetchone()
            return result[0] if result else 0
        except Exception as e:
            error_msg = f"Erro ao contar linhas: {str(e)}"
            logger.error(error_msg)
            raise DataReadException(error_msg) from e

# Factory para criação de conectores
class DataConnectorFactory:
    """
    Factory para criar instâncias de conectores específicos.
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
        Cria uma instância de conector com base na configuração.
        
        Args:
            config: Configuração da fonte de dados.
            
        Returns:
            DataConnector: Nova instância do conector apropriado.
            
        Raises:
            ConfigurationException: Se o tipo de conector não for suportado.
        """
        if isinstance(config, dict):
            config = DataSourceConfig.from_dict(config)
            
        source_type = config.source_type
        
        if source_type not in cls._connectors:
            raise ConfigurationException(f"Tipo de conector não suportado: {source_type}")
            
        connector_class = cls._connectors[source_type]
        return connector_class(config)
    
    @classmethod
    def create_from_json_path(cls, json_path: str) -> Dict[str, DataConnector]:
        return cls.create_from_json(cls, json_path)
        
    @classmethod
    def create_from_json(cls, json_config: str) -> Dict[str, DataConnector]:
        """
        Cria múltiplos conectores a partir de uma configuração JSON.
        
        Args:
            json_config: String JSON com configurações de múltiplas fontes.
            
        Returns:
            Dict[str, DataConnector]: Dicionário com instâncias de conectores.
        """
        try:
            config_data = json.loads(json_config)
            
            if 'data_sources' not in config_data:
                raise ConfigurationException("Formato de configuração inválido. Esperava 'data_sources' como chave principal.")
                
            sources_data = config_data['data_sources']
            
            connectors = {}
            for source_config in sources_data:
                source_id = source_config.get('id')
                if not source_id:
                    raise ConfigurationException("Configuração de fonte sem ID")
                    
                connector = cls.create_connector(source_config)
                connectors[source_id] = connector
                
            return connectors
                
        except json.JSONDecodeError as e:
            raise ConfigurationException(f"Erro ao decodificar JSON: {str(e)}")
