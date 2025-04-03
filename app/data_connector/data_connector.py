"""
Conectores de Dados para GenBI - Módulo para conexão com diferentes fontes de dados
"""

from abc import ABC, abstractmethod
import logging
from typing import Any, Dict, List, Optional, Union, Tuple
import pandas as pd

logger = logging.getLogger("GenBI.DataConnector")

class DataConnector(ABC):
    """Interface base para todos os conectores de dados"""
    
    @abstractmethod
    def connect(self) -> bool:
        """Estabelece conexão com a fonte de dados"""
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Encerra a conexão com a fonte de dados"""
        pass
    
    @abstractmethod
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Executa uma consulta e retorna os resultados como DataFrame"""
        pass
    
    @abstractmethod
    def get_table_schema(self, table_name: str, schema: Optional[str] = None) -> Dict[str, Any]:
        """Retorna o schema de uma tabela"""
        pass
    
    @abstractmethod
    def list_tables(self, schema: Optional[str] = None) -> List[str]:
        """Lista todas as tabelas disponíveis"""
        pass
    
    @abstractmethod
    def test_connection(self) -> bool:
        """Testa se a conexão está funcionando"""
        pass

class SQLiteConnector(DataConnector):
    """Conector para SQLite"""
    
    def __init__(self, database_path: str):
        """
        Inicializa o conector SQLite
        
        Args:
            database_path: Caminho para o arquivo de banco de dados
        """
        self.database_path = database_path
        self.connection = None
        
    def connect(self) -> bool:
        """Estabelece conexão com o banco de dados SQLite"""
        try:
            import sqlite3
            self.connection = sqlite3.connect(self.database_path)
            return True
        except Exception as e:
            logger.error(f"Erro ao conectar ao SQLite: {str(e)}")
            return False
    
    def disconnect(self) -> None:
        """Encerra a conexão com o banco de dados"""
        if self.connection:
            self.connection.close()
    
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Executa uma consulta SQL e retorna os resultados como DataFrame"""
        if not self.connection:
            self.connect()
        
        try:
            if params:
                return pd.read_sql_query(query, self.connection, params=params)
            else:
                return pd.read_sql_query(query, self.connection)
        except Exception as e:
            logger.error(f"Erro ao executar consulta SQLite: {str(e)}")
            raise
    
    def get_table_schema(self, table_name: str, schema: Optional[str] = None) -> Dict[str, Any]:
        """Retorna o schema de uma tabela SQLite"""
        if not self.connection:
            self.connect()
        
        try:
            # Obter informações da tabela
            pragma_query = f"PRAGMA table_info({table_name})"
            columns_df = pd.read_sql_query(pragma_query, self.connection)
            
            # Converter para o formato esperado
            columns = {}
            for _, row in columns_df.iterrows():
                columns[row['name']] = {
                    'type': row['type'],
                    'nullable': row['notnull'] == 0,
                    'primary_key': row['pk'] == 1,
                    'default': row['dflt_value']
                }
            
            return {
                'name': table_name,
                'schema': 'main',  # SQLite usa 'main' como schema padrão
                'columns': columns
            }
        except Exception as e:
            logger.error(f"Erro ao obter schema da tabela {table_name}: {str(e)}")
            raise
    
    def list_tables(self, schema: Optional[str] = None) -> List[str]:
        """Lista todas as tabelas disponíveis no SQLite"""
        if not self.connection:
            self.connect()
        
        try:
            query = "SELECT name FROM sqlite_master WHERE type='table'"
            tables_df = pd.read_sql_query(query, self.connection)
            return tables_df['name'].tolist()
        except Exception as e:
            logger.error(f"Erro ao listar tabelas SQLite: {str(e)}")
            raise
    
    def test_connection(self) -> bool:
        """Testa se a conexão SQLite está funcionando"""
        try:
            if not self.connection:
                return self.connect()
            
            # Executa uma consulta simples para verificar a conexão
            cursor = self.connection.cursor()
            cursor.execute("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"Erro ao testar conexão SQLite: {str(e)}")
            return False