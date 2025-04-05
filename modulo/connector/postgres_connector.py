import pandas as pd
import logging

from connector.data_connector import DataConnector
from connector.datasource_config import DataSourceConfig
from modulo.connector.exceptions import ConfigurationException, DataConnectionException, DataReadException

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("connector")



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
