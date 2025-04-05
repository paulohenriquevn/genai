"""
Executor de consultas SQL com suporte a diferentes dialetos e otimizações.
"""

import re
import logging
import pandas as pd
from typing import Dict, Callable, List, Any, Optional

from core.exceptions import QueryExecutionError
from core.engine.dataset import Dataset

# Configura o logger
logger = logging.getLogger("core_integration")


class SQLExecutor:
    """
    Executor de consultas SQL com suporte a diferentes dialetos e funções personalizadas.
    
    Fornece:
    - Adaptação de consultas SQL para compatibilidade com DuckDB
    - Registro de funções SQL personalizadas
    - Verificação de existência de tabelas
    - Fallback para execução com pandas
    """
    
    def __init__(self, datasets: Dict[str, Dataset]):
        """
        Inicializa o executor SQL.
        
        Args:
            datasets: Dicionário de datasets disponíveis (nome -> objeto Dataset)
        """
        self.datasets = datasets

    def create_sql_executor(self) -> Callable:
        """
        Cria uma função para executar consultas SQL em datasets.
        
        Returns:
            Função que executa SQL em DataFrames com suporte a funções SQL compatíveis
        """
        # Integração com DuckDB para execução SQL mais robusta
        try:
            import duckdb
            
            def adapt_sql_query(sql_query: str) -> str:
                """
                Adapta uma consulta SQL para compatibilidade com DuckDB.
                
                Args:
                    sql_query: Consulta SQL original
                    
                Returns:
                    Consulta SQL adaptada para DuckDB
                """
                # Verificação de tabelas existentes
                table_names = list(self.datasets.keys())
                
                # Verifica se a consulta referencia tabelas inexistentes
                for table in re.findall(r'FROM\s+(\w+)', sql_query, re.IGNORECASE):
                    if table not in table_names:
                        logger.warning(f"Tabela '{table}' não encontrada nos datasets carregados")
                
                # Substitui funções de data incompatíveis
                # DATE_FORMAT(campo, '%Y-%m-%d') -> strftime('%Y-%m-%d', campo)
                sql_query = re.sub(
                    r'DATE_FORMAT\s*\(\s*([^,]+)\s*,\s*[\'"]([^\'"]+)[\'"]\s*\)',
                    r"strftime('\2', \1)",
                    sql_query
                )
                
                # TO_DATE(string) -> DATE(string)
                sql_query = re.sub(
                    r'TO_DATE\s*\(\s*([^)]+)\s*\)',
                    r'DATE(\1)',
                    sql_query
                )
                
                # Funções de string
                # CONCAT(a, b) -> a || b
                sql_query = re.sub(
                    r'CONCAT\s*\(\s*([^,]+)\s*,\s*([^)]+)\s*\)',
                    r'(\1 || \2)',
                    sql_query
                )
                
                # SUBSTRING(x, start, len) -> SUBSTR(x, start, len)
                sql_query = re.sub(
                    r'SUBSTRING\s*\(',
                    r'SUBSTR(',
                    sql_query
                )
                
                # Funções de agregação
                # GROUP_CONCAT -> STRING_AGG
                sql_query = re.sub(
                    r'GROUP_CONCAT\s*\(',
                    r'STRING_AGG(',
                    sql_query
                )
                
                logger.debug(f"Consulta SQL adaptada: {sql_query}")
                return sql_query
            
            def check_table_existence(sql_query: str) -> None:
                """Verifica se as tabelas referenciadas existem."""
                table_refs = re.findall(r'FROM\s+(\w+)', sql_query, re.IGNORECASE)
                table_refs.extend(re.findall(r'JOIN\s+(\w+)', sql_query, re.IGNORECASE))
                
                for table in table_refs:
                    if table not in self.datasets:
                        raise ValueError(f"Tabela '{table}' não encontrada nos datasets carregados. " + 
                                    f"Datasets disponíveis: {', '.join(self.datasets.keys())}")
            
            def register_custom_sql_functions(con: duckdb.DuckDBPyConnection) -> None:
                """
                Registra funções SQL personalizadas no DuckDB para ampliar a compatibilidade
                com outros dialetos SQL, usando abordagem simplificada.
                
                Args:
                    con: Conexão DuckDB
                """
                try:
                    # Função utilitária para criar SQL functions de forma segura
                    def safe_create_function(sql):
                        try:
                            con.execute(sql)
                        except Exception as e:
                            logger.warning(f"Erro ao criar função SQL: {str(e)}")
                    
                    # GROUP_CONCAT para compatibilidade com MySQL
                    safe_create_function("CREATE OR REPLACE MACRO GROUP_CONCAT(x) AS STRING_AGG(x, ',')")
                    
                    # DATE_FORMAT simplificada (casos mais comuns)
                    safe_create_function("""
                    CREATE OR REPLACE MACRO DATE_FORMAT(d, f) AS
                    CASE 
                        WHEN f = '%Y-%m-%d' THEN strftime('%Y-%m-%d', d)
                        WHEN f = '%Y-%m' THEN strftime('%Y-%m', d)
                        WHEN f = '%Y' THEN strftime('%Y', d)
                        ELSE strftime('%Y-%m-%d', d)
                    END
                    """)
                    
                    # TO_DATE para converter para data
                    safe_create_function("CREATE OR REPLACE MACRO TO_DATE(d) AS TRY_CAST(d AS DATE)")
                    
                    # String concatenation helpers
                    safe_create_function("CREATE OR REPLACE MACRO CONCAT(a, b) AS a || b")
                    
                    # Concat with separator (simplified version)
                    safe_create_function("""
                    CREATE OR REPLACE MACRO CONCAT_WS(sep, a, b) AS
                    CASE 
                        WHEN a IS NULL AND b IS NULL THEN NULL
                        WHEN a IS NULL THEN b
                        WHEN b IS NULL THEN a
                        ELSE a || sep || b
                    END
                    """)
                    
                    # Register extract functions for date parts
                    safe_create_function("""
                    CREATE OR REPLACE MACRO YEAR(d) AS EXTRACT(YEAR FROM d)
                    """)
                    
                    safe_create_function("""
                    CREATE OR REPLACE MACRO MONTH(d) AS EXTRACT(MONTH FROM d)
                    """)
                    
                    safe_create_function("""
                    CREATE OR REPLACE MACRO DAY(d) AS EXTRACT(DAY FROM d)
                    """)
                    
                    logger.info("Funções SQL personalizadas registradas com sucesso")
                    
                except Exception as e:
                    logger.warning(f"Erro ao registrar funções SQL personalizadas: {str(e)}")
            
            def execute_sql(sql_query: str) -> pd.DataFrame:
                """Executa uma consulta SQL usando DuckDB com adaptações de compatibilidade."""
                try:
                    # Verifica se tabelas existem antes de executar
                    check_table_existence(sql_query)
                    
                    # Adapta a consulta para compatibilidade com DuckDB
                    adapted_query = adapt_sql_query(sql_query)
                    
                    # Estabelece conexão com todos os dataframes
                    con = duckdb.connect(database=':memory:')
                    
                    # Registra funções SQL personalizadas
                    register_custom_sql_functions(con)
                    
                    # Registra todos os datasets
                    for name, dataset in self.datasets.items():
                        # Registra o dataframe
                        con.register(name, dataset.dataframe)
                        
                        # Cria visualizações otimizadas para funções de data
                        con.execute(f'''
                        CREATE OR REPLACE VIEW {name}_date_view AS 
                        SELECT * FROM {name}
                        ''')
                    
                    # Executa a consulta
                    result = con.execute(adapted_query).fetchdf()
                    
                    # Registra a consulta SQL para debugging
                    sql_logger = logging.getLogger("sql_logger")
                    sql_logger.info(f"Consulta SQL executada: {adapted_query}")
                    
                    return result
                except Exception as e:
                    logger.error(f"Erro SQL: {str(e)}")
                    raise QueryExecutionError(f"Erro ao executar SQL: {str(e)}")
        
        except ImportError:
            # Fallback para pandas se DuckDB não estiver disponível
            logger.warning("DuckDB não encontrado, usando pandas para consultas SQL")
            
            def execute_sql(sql_query: str) -> pd.DataFrame:
                """Executa uma consulta SQL básica usando pandas."""
                try:
                    # Para o modo pandas, suporta apenas SELECT * FROM dataset
                    import re
                    match = re.search(r'FROM\s+(\w+)', sql_query, re.IGNORECASE)
                    
                    if not match:
                        raise ValueError("Consulta SQL inválida. Formato esperado: SELECT * FROM dataset")
                    
                    dataset_name = match.group(1)
                    
                    if dataset_name not in self.datasets:
                        raise ValueError(f"Dataset '{dataset_name}' não encontrado")
                    
                    # Registra a consulta SQL para debugging
                    sql_logger = logging.getLogger("sql_logger")
                    sql_logger.info(f"Consulta SQL simulada: {sql_query}")
                    
                    # Retorna o dataset inteiro (limitação do modo pandas)
                    return self.datasets[dataset_name].dataframe
                except Exception as e:
                    logger.error(f"Erro SQL: {str(e)}")
                    raise QueryExecutionError(f"Erro ao executar SQL: {str(e)}")
        
        return execute_sql