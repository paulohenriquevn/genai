"""
Conectores de Dados para GenBI - Módulo para conexão com diferentes fontes de dados
"""

from abc import ABC, abstractmethod
import logging
import os
import re
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

class CSVConnector(DataConnector):
    """Conector para arquivos CSV"""
    
    def __init__(self, csv_dir: str = "uploads/csv"):
        """
        Inicializa o conector CSV
        
        Args:
            csv_dir: Diretório para armazenar os arquivos CSV
        """
        self.csv_dir = csv_dir
        self.loaded_files = {}  # Mapeia nomes de arquivos para DataFrames
        self.file_schemas = {}  # Armazena o schema de cada arquivo
        
        # Criar diretório de uploads se não existir
        os.makedirs(csv_dir, exist_ok=True)
    
    def connect(self) -> bool:
        """Estabelece 'conexão' com os arquivos CSV (carrega arquivos existentes)"""
        try:
            # Listar e carregar todos os arquivos CSV no diretório
            file_count = 0
            
            for filename in os.listdir(self.csv_dir):
                if filename.lower().endswith('.csv'):
                    file_path = os.path.join(self.csv_dir, filename)
                    try:
                        # Carregar arquivo para o dicionário
                        self.loaded_files[filename] = pd.read_csv(file_path)
                        # Inferir e armazenar schema
                        self._infer_schema(filename)
                        file_count += 1
                    except Exception as e:
                        logger.warning(f"Erro ao carregar arquivo CSV {filename}: {str(e)}")
            
            logger.info(f"Carregados {file_count} arquivos CSV")
            return True
        except Exception as e:
            logger.error(f"Erro ao conectar com arquivos CSV: {str(e)}")
            return False
    
    def disconnect(self) -> None:
        """'Desconecta' do CSV (limpa os dados carregados)"""
        self.loaded_files = {}
        self.file_schemas = {}
    
    def _infer_schema(self, filename: str) -> None:
        """Infere o schema de um arquivo CSV carregado"""
        if filename not in self.loaded_files:
            return
        
        df = self.loaded_files[filename]
        columns = {}
        
        for col_name in df.columns:
            dtype = df[col_name].dtype
            
            # Mapear tipo pandas para tipo mais genérico
            if pd.api.types.is_integer_dtype(dtype):
                col_type = "integer"
            elif pd.api.types.is_float_dtype(dtype):
                col_type = "number"
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                col_type = "datetime"
            elif pd.api.types.is_bool_dtype(dtype):
                col_type = "boolean"
            else:
                col_type = "string"
            
            # Verificar se parece ser categórico (menos de 20 valores únicos e menos de 20% do total)
            if col_type in ["string", "integer"]:
                nunique = df[col_name].nunique()
                if nunique < 20 and nunique < 0.2 * len(df):
                    semantic_type = "category"
                elif col_type == "string" and any(date_pattern in col_name.lower() for date_pattern in ["date", "dt", "time"]):
                    semantic_type = "date"
                else:
                    semantic_type = None
            elif col_type == "number":
                # Tentar determinar se é preço, porcentagem, etc.
                if any(price_pattern in col_name.lower() for price_pattern in ["price", "cost", "value", "amount", "revenue"]):
                    semantic_type = "price"
                elif col_name.lower().endswith(('%', 'percent', 'percentage')):
                    semantic_type = "percentage"
                else:
                    semantic_type = "measure"
            else:
                semantic_type = None
            
            columns[col_name] = {
                'type': col_type,
                'nullable': df[col_name].isna().any(),
                'semantic_type': semantic_type
            }
        
        self.file_schemas[filename] = {
            'name': os.path.splitext(filename)[0],
            'columns': columns,
            'row_count': len(df)
        }
    
    def upload_csv(self, file_path: str, filename: Optional[str] = None) -> str:
        """
        Carrega um arquivo CSV para o diretório de uploads
        
        Args:
            file_path: Caminho para o arquivo CSV temporário
            filename: Nome de arquivo opcional (se não fornecido, usa o original)
            
        Returns:
            str: Nome do arquivo armazenado
        """
        if not filename:
            filename = os.path.basename(file_path)
        
        # Garantir que tenha extensão .csv
        if not filename.lower().endswith('.csv'):
            filename += '.csv'
        
        # Tornar nome de arquivo seguro (remover caracteres especiais)
        safe_filename = re.sub(r'[^\w\.-]', '_', filename)
        
        # Caminho completo para destino
        destination = os.path.join(self.csv_dir, safe_filename)
        
        try:
            # Ler o arquivo com pandas para validar
            df = pd.read_csv(file_path)
            
            # Salvar no diretório de uploads
            df.to_csv(destination, index=False)
            
            # Adicionar ao dicionário de arquivos carregados
            self.loaded_files[safe_filename] = df
            
            # Inferir schema
            self._infer_schema(safe_filename)
            
            return safe_filename
        except Exception as e:
            logger.error(f"Erro ao carregar arquivo CSV: {str(e)}")
            raise
    
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        'Executa' uma consulta em um arquivo CSV
        
        A consulta deve ter um formato especial:
        SELECT [colunas] FROM [arquivo.csv] WHERE [condições] GROUP BY [colunas] ORDER BY [colunas]
        
        Args:
            query: Consulta SQL-like
            params: Parâmetros para a consulta (ignorados para CSV)
            
        Returns:
            pd.DataFrame: Resultados da consulta
        """
        # Análise simplificada da "consulta"
        query = query.strip()
        
        # Verificar se é uma consulta virtual para CSV
        if "FROM" not in query:
            raise ValueError("Consulta CSV deve especificar FROM [arquivo.csv]")
        
        # Verificar se o usuário está especificando um arquivo CSV
        from_match = re.search(r'FROM\s+([^\s,;]+\.csv)', query, re.IGNORECASE)
        if not from_match:
            # Tente encontrar qualquer identificador após FROM
            from_match = re.search(r'FROM\s+([^\s,;]+)', query, re.IGNORECASE)
            if not from_match:
                raise ValueError("Não foi possível identificar o arquivo CSV na consulta")
            
            # Verifique se o identificador corresponde a um arquivo (com ou sem extensão .csv)
            csv_file = from_match.group(1)
            if not csv_file.lower().endswith('.csv'):
                csv_file += '.csv'
        else:
            csv_file = from_match.group(1)
        
        # Verificar se o arquivo existe
        if csv_file not in self.loaded_files:
            available_files = list(self.loaded_files.keys())
            raise ValueError(f"Arquivo CSV '{csv_file}' não encontrado. Disponíveis: {', '.join(available_files)}")
        
        # Obter DataFrame
        df = self.loaded_files[csv_file]
        
        # Processar cláusula SELECT
        select_all = "SELECT *" in query or "SELECT" not in query
        selected_columns = []
        column_transforms = {}  # Armazenar transformações de colunas (como funções de data)
        
        if not select_all:
            select_match = re.search(r'SELECT\s+(.+?)\s+FROM', query, re.IGNORECASE)
            if select_match:
                columns_str = select_match.group(1).strip()
                selected_columns = [col.strip() for col in columns_str.split(',')]
                
                # Mapear e processar funções especiais no SELECT
                for i, col in enumerate(selected_columns):
                    # Verificar se é uma expressão com função
                    if "(" in col and ")" in col:
                        # Verificar se é uma função de data (strftime)
                        if "strftime" in col.lower():
                            # Extrair partes da função
                            format_match = re.search(r"strftime\(['\"](.+?)['\"],\s*(.+?)\)(?:\s+AS\s+([^\s,]+))?", col, re.IGNORECASE)
                            if format_match:
                                date_format = format_match.group(1)
                                date_col = format_match.group(2).strip()
                                date_alias = format_match.group(3) if format_match.group(3) else "month"
                                
                                # Armazenar a transformação para aplicar ao dataframe posteriormente
                                column_transforms[date_alias] = {
                                    'type': 'date_format',
                                    'format': date_format,
                                    'source_col': date_col
                                }
                
                # Verificar se todas as colunas existem e tentar mapeamento automático
                column_mapping = {}
                for col in selected_columns:
                    # Ignorar expressões com funções e aliases como "strftime(...) AS month"
                    if "(" in col and ")" in col:
                        continue
                        
                    # Verificar se é um alias de coluna
                    col_parts = col.split(' AS ')
                    col_name = col_parts[0].strip()
                    
                    if col_name != '*' and col_name not in df.columns:
                        # Tentar mapear nomes de colunas comuns
                        if col_name.lower() == 'categoria' and 'category' in df.columns:
                            column_mapping[col_name] = 'category'
                        elif col_name.lower() == 'receita_total' and 'total_amount' in df.columns:
                            column_mapping[col_name] = 'total_amount'
                        elif col_name.lower() == 'quantidade_vendida' and 'quantity' in df.columns:
                            column_mapping[col_name] = 'quantity'
                        elif col_name not in column_transforms:  # Ignorar colunas transformadas
                            # Se é uma coluna com alias, verificar se o alias já existe no dataframe
                            if len(col_parts) > 1 and col_parts[1].strip() in df.columns:
                                continue
                            else:
                                logger.warning(f"Coluna '{col_name}' não encontrada no arquivo {csv_file}")
                
                # Substituir colunas na consulta original se necessário
                for old_col, new_col in column_mapping.items():
                    logger.info(f"Mapeando coluna '{old_col}' para '{new_col}'")
                    # Substituir as referências à coluna na consulta completa
                    query = re.sub(r'\b' + re.escape(old_col) + r'\b', new_col, query)
                
                # Aplicar transformações de data ao dataframe
                for alias, transform in column_transforms.items():
                    if transform['type'] == 'date_format':
                        try:
                            source_col = transform['source_col']
                            # Se a coluna fonte precisa ser mapeada
                            if source_col in column_mapping:
                                source_col = column_mapping[source_col]
                                
                            logger.info(f"Aplicando transformação de data à coluna {source_col} como {alias}")
                            if source_col in df.columns:
                                # Criar nova coluna com o formato de data desejado
                                if transform['format'] == '%Y-%m':
                                    df[alias] = pd.to_datetime(df[source_col]).dt.strftime('%Y-%m')
                                else:
                                    df[alias] = pd.to_datetime(df[source_col]).dt.strftime(transform['format'])
                        except Exception as e:
                            logger.warning(f"Erro ao transformar coluna de data: {str(e)}")
                
                # Precisamos reprocessar a consulta após as substituições
                if column_mapping:
                    # Re-extrair partes da consulta modificada
                    select_match = re.search(r'SELECT\s+(.+?)\s+FROM', query, re.IGNORECASE)
                    if select_match:
                        columns_str = select_match.group(1).strip()
                        selected_columns = [col.strip() for col in columns_str.split(',')]
        
        # Processar cláusula WHERE
        filtered_df = df.copy()
        where_match = re.search(r'WHERE\s+(.+?)(?:\s+GROUP BY|\s+ORDER BY|$)', query, re.IGNORECASE)
        
        if where_match:
            # Implementação muito simplificada - apenas para exemplos básicos
            # Em produção, usaria parser de expressão mais robusto
            condition = where_match.group(1).strip()
            try:
                filtered_df = filtered_df.query(condition)
            except Exception as e:
                logger.warning(f"Erro ao aplicar filtro WHERE: {str(e)}")
                logger.warning("Condição WHERE ignorada")
        
        # Definir dicionário de mapeamento se não foi criado anteriormente
        if 'column_mapping' not in locals():
            column_mapping = {}
            # Verificar cada coluna se existe no DataFrame
            for col in df.columns:
                if col.lower() == 'category' and 'categoria' not in df.columns:
                    column_mapping['categoria'] = col
                elif col.lower() == 'total_amount' and 'receita_total' not in df.columns:
                    column_mapping['receita_total'] = col
                elif col.lower() == 'quantity' and 'quantidade_vendida' not in df.columns:
                    column_mapping['quantidade_vendida'] = col
        
        # Processar cláusula GROUP BY
        groupby_match = re.search(r'GROUP BY\s+(.+?)(?:\s+ORDER BY|$)', query, re.IGNORECASE)
        
        if groupby_match:
            group_cols_str = groupby_match.group(1)
            
            # Aplicar mapeamento de colunas ao GROUP BY
            for old_col, new_col in column_mapping.items():
                group_cols_str = re.sub(r'\b' + re.escape(old_col) + r'\b', new_col, group_cols_str)
                
            group_cols = [col.strip() for col in group_cols_str.split(',')]
            
            # Verificar se todas as colunas existem e tratar funções especiais
            final_group_cols = []
            for col in group_cols:
                # Verificar se é uma expressão com função (como strftime)
                if "(" in col and ")" in col:
                    # Para GROUP BY com funções de data, adicionar a coluna como uma nova coluna calculada
                    if "strftime" in col.lower():
                        # Extrair o formato e coluna usados no strftime
                        format_match = re.search(r"strftime\(['\"](.+?)['\"],\s*(.+?)\)", col, re.IGNORECASE)
                        if format_match:
                            date_format = format_match.group(1)
                            date_col = format_match.group(2).strip()
                            
                            # Converter para formato pandas
                            if date_format == '%Y-%m':
                                # Criar uma coluna temporária com o ano-mês
                                col_alias = "month"
                                try:
                                    # Converter para datetime e extrair ano-mês
                                    df[col_alias] = pd.to_datetime(df[date_col]).dt.strftime('%Y-%m')
                                    final_group_cols.append(col_alias)
                                    continue
                                except Exception as e:
                                    logger.warning(f"Erro ao criar coluna para agrupamento por data: {str(e)}")
                
                # Verificar se a coluna existe diretamente
                if col in df.columns:
                    final_group_cols.append(col)
                else:
                    logger.warning(f"Coluna '{col}' no GROUP BY não encontrada. Colunas disponíveis: {list(df.columns)}")
            
            # Se não temos colunas válidas para agrupar, usar todas as colunas
            if not final_group_cols and group_cols:
                logger.warning(f"Nenhuma coluna válida para agrupar. Tentando resolver expressões.")
                # Adicionar colunas originais que existem no dataframe
                for col in group_cols:
                    # Se parece ser uma coluna com alias (como em "strftime(...) AS month")
                    alias_match = re.search(r'\bAS\s+([^\s,]+)', col, re.IGNORECASE)
                    if alias_match:
                        alias = alias_match.group(1).strip()
                        # Adicionar o alias como uma coluna se ele estava no SELECT
                        if alias not in df.columns:
                            logger.warning(f"Criando coluna temporária '{alias}' para GROUP BY")
                            # Tentar criar a coluna com base na primeira coluna de date
                            date_cols = [c for c in df.columns if "date" in c.lower()]
                            if date_cols and "month" in alias.lower():
                                try:
                                    df[alias] = pd.to_datetime(df[date_cols[0]]).dt.strftime('%Y-%m')
                                    final_group_cols.append(alias)
                                except Exception as e:
                                    logger.warning(f"Erro ao criar coluna temporária: {str(e)}")
            
            # Usar as colunas finais resolvidas para agrupamento
            group_cols = final_group_cols if final_group_cols else group_cols
            
            # Verificar se há funções de agregação no SELECT
            agg_funcs = {}
            for col in selected_columns:
                if "(" in col and ")" in col:
                    # Extrair nome da função e coluna
                    func_match = re.match(r'(\w+)\(([^)]+)\)', col)
                    if func_match:
                        func_name = func_match.group(1).lower()
                        col_name = func_match.group(2).strip()
                        
                        # Aplicar mapeamento de colunas à coluna dentro da função
                        if col_name in column_mapping:
                            col_name = column_mapping[col_name]
                        
                        if func_name == 'sum':
                            agg_funcs[col_name] = 'sum'
                        elif func_name == 'avg':
                            agg_funcs[col_name] = 'mean'
                        elif func_name in ['count', 'min', 'max']:
                            agg_funcs[col_name] = func_name
            
            if agg_funcs:
                try:
                    filtered_df = filtered_df.groupby(group_cols).agg(agg_funcs).reset_index()
                except Exception as e:
                    logger.error(f"Erro ao aplicar GROUP BY: {str(e)}")
                    raise
            else:
                # Se não houver funções de agregação explícitas, agrupe apenas pelas colunas
                filtered_df = filtered_df.groupby(group_cols).first().reset_index()
        
        # Processar cláusula ORDER BY
        orderby_match = re.search(r'ORDER BY\s+(.+?)(?:\s+LIMIT|$)', query, re.IGNORECASE)
        
        if orderby_match:
            order_str = orderby_match.group(1)
            
            # Aplicar mapeamento de colunas ao ORDER BY
            for old_col, new_col in column_mapping.items():
                order_str = re.sub(r'\b' + re.escape(old_col) + r'\b', new_col, order_str)
                
            order_cols = []
            ascending = []
            
            order_parts = order_str.split(',')
            for part in order_parts:
                part = part.strip()
                if " DESC" in part.upper():
                    order_cols.append(part.replace(" DESC", "").replace(" desc", "").strip())
                    ascending.append(False)
                else:
                    order_col = part.replace(" ASC", "").replace(" asc", "").strip()
                    order_cols.append(order_col)
                    ascending.append(True)
            
            # Verificar se todas as colunas no ORDER BY existem
            valid_order_cols = []
            valid_ascending = []
            for i, col in enumerate(order_cols):
                if col in df.columns:
                    valid_order_cols.append(col)
                    valid_ascending.append(ascending[i])
                else:
                    logger.warning(f"Coluna '{col}' no ORDER BY não encontrada. Ignorando.")
            
            try:
                if valid_order_cols:  # Prosseguir apenas se houver colunas válidas
                    filtered_df = filtered_df.sort_values(by=valid_order_cols, ascending=valid_ascending)
            except Exception as e:
                logger.warning(f"Erro ao aplicar ORDER BY: {str(e)}")
        
        # Processar cláusula LIMIT
        limit_match = re.search(r'LIMIT\s+(\d+)', query, re.IGNORECASE)
        
        if limit_match:
            try:
                limit = int(limit_match.group(1))
                filtered_df = filtered_df.head(limit)
            except Exception as e:
                logger.warning(f"Erro ao aplicar LIMIT: {str(e)}")
        
        # Selecionar apenas as colunas necessárias se especificadas
        if selected_columns and not select_all:
            # Lidar com funções de agregação no SELECT e aplicar mapeamento de colunas
            result_columns = []
            final_columns = []  # Para nomes reais após mapeamento
            
            for col in selected_columns:
                original_col = col
                
                if "(" in col and ")" in col:
                    # Verificar se é função de data (strftime)
                    if "strftime" in col.lower():
                        # Extrair o alias da função strftime
                        alias_match = re.search(r'AS\s+([^\s,]+)', col, re.IGNORECASE)
                        alias = alias_match.group(1).strip() if alias_match else "month"
                        # A coluna transformada já foi adicionada ao DataFrame
                        if alias in filtered_df.columns:
                            result_columns.append(alias)
                            final_columns.append(alias)
                        continue
                        
                    # Tentar extrair nome da função e coluna dentro dela
                    func_match = re.match(r'(\w+)\(([^)]+)\)(?: AS (.+))?', col)
                    if func_match:
                        func_name = func_match.group(1).lower()
                        col_name = func_match.group(2).strip()
                        alias = func_match.group(3).strip() if func_match.group(3) else None
                        
                        # Verificar se a coluna dentro da função precisa de mapeamento
                        if col_name in column_mapping:
                            mapped_col = column_mapping[col_name]
                            # Reconstruir a expressão com a coluna mapeada
                            if alias:
                                mapped_expr = f"{func_name}({mapped_col}) AS {alias}"
                            else:
                                mapped_expr = f"{func_name}({mapped_col})"
                            result_columns.append(mapped_expr)
                        else:
                            result_columns.append(col)
                else:
                    # Coluna simples
                    if col in column_mapping:
                        result_columns.append(column_mapping[col])
                        final_columns.append(column_mapping[col])
                    else:
                        result_columns.append(col)
                        final_columns.append(col)
            
            # Verificar se a consulta usou nomes mapeados e quais nomes estão no DataFrame resultado
            available_columns = filtered_df.columns.tolist()
            logger.info(f"Colunas disponíveis após processamento: {available_columns}")
            logger.info(f"Tentando selecionar colunas: {result_columns}")
            
            # Primeiro procurar por funções de agregação e colunas transformadas
            # Isso pode ter criado novas colunas no dataframe filtrado
            final_usable_columns = []
            
            # Adicionar primeiro as colunas transformadas que sabemos que existem
            for alias in column_transforms.keys():
                if alias in filtered_df.columns:
                    final_usable_columns.append(alias)
            
            # Depois verificar aliases de funções de agregação
            for col in result_columns:
                # Se for uma expressão com AS, extrair o alias
                as_match = re.search(r'\bAS\s+([^\s,]+)', col, re.IGNORECASE)
                if as_match:
                    alias = as_match.group(1).strip()
                    if alias in available_columns and alias not in final_usable_columns:
                        final_usable_columns.append(alias)
                # Verificar se a coluna original existe
                elif col in available_columns and col not in final_usable_columns:
                    final_usable_columns.append(col)
            
            # Se temos colunas válidas, retornar apenas essas
            if final_usable_columns:
                try:
                    logger.info(f"Selecionando colunas finais: {final_usable_columns}")
                    return filtered_df[final_usable_columns]
                except Exception as e:
                    logger.warning(f"Erro ao selecionar colunas: {str(e)}")
                    return filtered_df
            
            # Verificar se é possível usar as colunas diretamente
            try:
                valid_cols = [col for col in result_columns if col in available_columns]
                if valid_cols:
                    return filtered_df[valid_cols]
            except Exception as e:
                logger.warning(f"Erro ao selecionar colunas válidas: {str(e)}")
            
            # Se todas as estratégias falharem e temos aliases de coluna transformada,
            # tentar usar diretamente essas colunas
            if column_transforms:
                try:
                    transform_cols = [alias for alias in column_transforms.keys() if alias in filtered_df.columns]
                    if transform_cols:
                        logger.info(f"Usando colunas transformadas: {transform_cols}")
                        # Se temos categoria, incluir ela também
                        if 'category' in filtered_df.columns:
                            transform_cols.append('category')
                        # Se temos colunas de agregação (SUM), incluir elas também
                        for col in filtered_df.columns:
                            if col.startswith('sum_') or col.startswith('avg_') or col.startswith('count_'):
                                transform_cols.append(col)
                        return filtered_df[transform_cols]
                except Exception as e:
                    logger.warning(f"Erro ao usar colunas transformadas: {str(e)}")
            
            return filtered_df
        
        return filtered_df
    
    def get_table_schema(self, table_name: str, schema: Optional[str] = None) -> Dict[str, Any]:
        """
        Retorna o schema de um arquivo CSV
        
        Args:
            table_name: Nome do arquivo CSV (com ou sem extensão)
            schema: Ignorado para CSV
            
        Returns:
            Dict: Schema do arquivo
        """
        # Normalizar nome do arquivo
        if not table_name.lower().endswith('.csv'):
            table_name += '.csv'
        
        # Verificar se temos o schema
        if table_name in self.file_schemas:
            return self.file_schemas[table_name]
        
        # Se não temos, tente carregar o arquivo e inferir o schema
        if table_name in self.loaded_files:
            self._infer_schema(table_name)
            if table_name in self.file_schemas:
                return self.file_schemas[table_name]
        
        raise ValueError(f"Arquivo CSV '{table_name}' não encontrado ou não carregado")
    
    def list_tables(self, schema: Optional[str] = None) -> List[str]:
        """
        Lista todos os arquivos CSV disponíveis
        
        Args:
            schema: Ignorado para CSV
            
        Returns:
            List[str]: Lista de nomes de arquivos CSV
        """
        return list(self.loaded_files.keys())
    
    def test_connection(self) -> bool:
        """Testa se o diretório CSV está acessível"""
        return os.path.isdir(self.csv_dir) and os.access(self.csv_dir, os.R_OK | os.W_OK)


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