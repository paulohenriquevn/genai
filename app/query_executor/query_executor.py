"""
Módulo de Execução de Consultas e Geração de Visualizações
"""

import logging
import pandas as pd
import re
from typing import Any, Dict, Optional
from datetime import datetime
import os

logger = logging.getLogger("GenBI.QueryExecutor")

class QueryResult:
    """Representa o resultado de uma consulta"""
    def __init__(self, data: pd.DataFrame, 
                 execution_time: float, 
                 from_cache: bool = False):
        """
        Inicializa um resultado de consulta
        
        Args:
            data: DataFrame com os resultados
            execution_time: Tempo de execução da consulta
            from_cache: Se o resultado veio do cache
        """
        self.data = data
        self.execution_time = execution_time
        self.from_cache = from_cache
        self.executed_at = datetime.now()
        
        # Metadados adicionais
        self.row_count = len(data)
        self.column_count = len(data.columns)

class QueryCache:
    """Sistema de cache para consultas"""
    def __init__(self, cache_dir: str = "cache", ttl: int = 3600):
        """
        Inicializa o sistema de cache
        
        Args:
            cache_dir: Diretório para armazenar cache
            ttl: Tempo de vida do cache em segundos
        """
        self.cache_dir = cache_dir
        self.ttl = ttl
        
        # Criar diretório de cache se não existir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Importar pickle para serialização
        import pickle
        import hashlib
        self.pickle = pickle
        self.hashlib = hashlib
    
    def _get_cache_path(self, key: str) -> str:
        """Gera o caminho do arquivo de cache baseado na chave"""
        # Gerar hash da chave para evitar problemas com caracteres especiais
        hashed_key = self.hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{hashed_key}.pkl")
    
    def get(self, key: str) -> Optional[Any]:
        """
        Recupera item do cache
        
        Args:
            key: Chave de identificação do cache
        
        Returns:
            Dados em cache ou None se não encontrado/expirado
        """
        cache_path = self._get_cache_path(key)
        
        # Verificar se o arquivo existe
        if not os.path.exists(cache_path):
            return None
            
        # Verificar se o cache expirou
        file_mod_time = os.path.getmtime(cache_path)
        if (time.time() - file_mod_time) > self.ttl:
            # Cache expirado, remover arquivo
            try:
                os.remove(cache_path)
            except OSError:
                pass
            return None
            
        try:
            # Carregar dados do cache
            with open(cache_path, 'rb') as f:
                return self.pickle.load(f)
        except Exception as e:
            logger.error(f"Erro ao carregar cache: {str(e)}")
            return None
    
    def set(self, key: str, value: Any):
        """
        Armazena item no cache
        
        Args:
            key: Chave de identificação do cache
            value: Valor a ser armazenado
        """
        cache_path = self._get_cache_path(key)
        
        try:
            # Salvar dados no cache
            with open(cache_path, 'wb') as f:
                self.pickle.dump(value, f)
        except Exception as e:
            logger.error(f"Erro ao salvar cache: {str(e)}")
    
    def invalidate(self):
        """
        Limpa todo o cache
        
        Returns:
            Número de entradas removidas
        """
        count = 0
        for filename in os.listdir(self.cache_dir):
            if filename.endswith(".pkl"):
                file_path = os.path.join(self.cache_dir, filename)
                try:
                    os.remove(file_path)
                    count += 1
                except OSError as e:
                    logger.error(f"Erro ao remover arquivo de cache {file_path}: {str(e)}")
        return count

class QueryExecutor:
    """Executa consultas em fontes de dados"""
    
    def __init__(self, data_connector, cache: Optional[QueryCache] = None, data_connectors=None):
        """
        Inicializa o executor de consultas
        
        Args:
            data_connector: Conector de dados padrão
            cache: Sistema de cache opcional
            data_connectors: Dicionário de conectores de dados por tipo
        """
        self.data_connector = data_connector  # Conector padrão
        self.data_connectors = data_connectors or {}  # Todos os conectores
        self.cache = cache
    
    def add_connector(self, name: str, connector):
        """Adiciona um novo conector de dados"""
        self.data_connectors[name] = connector
        
    def select_connector(self, query: str):
        """
        Seleciona o conector de dados apropriado baseado na consulta
        
        Args:
            query: Consulta a ser executada
            
        Returns:
            Conector de dados apropriado
        """
        query = query.strip().lower()
        
        # Verificar se parece ser uma consulta para CSV (contém .csv)
        if '.csv' in query and 'csv' in self.data_connectors:
            # Modificar a consulta para evitar JOINs em CSV se necessário
            if " join " in query:
                logger.warning("Consulta CSV contém JOINs, que não são suportados corretamente. Simplificando consulta...")
                # Extrair apenas a primeira tabela que parece ser um arquivo CSV
                csv_table_match = re.search(r'from\s+([^\s]+\.csv)', query)
                if csv_table_match:
                    csv_table = csv_table_match.group(1)
                    logger.info(f"Arquivo CSV identificado: {csv_table}")
            
            return self.data_connectors['csv']
        
        # Se a consulta começar com "csv:" usando o conector CSV explicitamente
        if query.startswith('csv:') and 'csv' in self.data_connectors:
            return self.data_connectors['csv']
            
        # Para outros casos, usar o conector padrão
        return self.data_connector
        
    def execute(self, query: str, 
                params: Optional[Dict[str, Any]] = None, 
                use_cache: bool = True,
                connector_name: Optional[str] = None) -> QueryResult:
        """
        Executa uma consulta
        
        Args:
            query: Consulta a ser executada
            params: Parâmetros para a consulta
            use_cache: Se deve usar o cache
            connector_name: Nome do conector a usar (opcional)
            
        Returns:
            Resultado da consulta
        """
        import time
        
        start_time = time.time()
        
        # Remover qualquer prefixo de conector (ex: "csv:", "sqlite:")
        original_query = query
        connector_prefix = None
        
        if ':' in query and ' ' not in query.split(':')[0]:
            parts = query.split(':', 1)
            if len(parts) > 1:
                connector_prefix = parts[0].lower()
                query = parts[1].strip()
        
        # Usar o conector especificado ou selecionar o apropriado
        if connector_name and connector_name in self.data_connectors:
            selected_connector = self.data_connectors[connector_name]
        elif connector_prefix and connector_prefix in self.data_connectors:
            selected_connector = self.data_connectors[connector_prefix]
        else:
            # Selecionar automaticamente baseado na consulta
            selected_connector = self.select_connector(original_query)
            
        # Se é um conector CSV, simplificar a consulta se necessário
        if selected_connector is self.data_connectors.get('csv') and " JOIN " in query.upper():
            logger.warning("Detectada consulta CSV com JOINs. Simplificando para consulta plana...")
            # Extrair a tabela CSV principal
            csv_table_match = re.search(r'FROM\s+([^\s,]+\.csv)', query, re.IGNORECASE)
            if csv_table_match:
                csv_table = csv_table_match.group(1)
                
                # Simplificar a consulta para usar apenas colunas diretas
                simplified_query = query
                
                # Remover aliases de tabela em referências a colunas
                for alias in ['p.', 'oi.', 'o.', 'c.']:
                    simplified_query = simplified_query.replace(alias, '')
                
                # Remover cláusulas JOIN completamente
                simplified_query = re.sub(r'JOIN\s+[^\s]+\s+[a-z]+\s+ON\s+[^WHERE|GROUP|ORDER|LIMIT|;]+', '', 
                                          simplified_query, flags=re.IGNORECASE)
                
                # Tentar outro padrão se o primeiro falhar
                if " JOIN " in simplified_query.upper():
                    simplified_query = re.sub(r'JOIN.+?(?=(WHERE|GROUP BY|ORDER BY|LIMIT|$))', '', 
                                             simplified_query, flags=re.IGNORECASE | re.DOTALL)
                
                # Remover alias de tabela da cláusula FROM
                simplified_query = re.sub(r'FROM\s+([^\s]+\.csv)\s+[a-z][a-z]?', 
                                         f'FROM {csv_table}', simplified_query, flags=re.IGNORECASE)
                
                # Simplificar ainda mais, focando apenas nas colunas e na tabela CSV
                select_match = re.search(r'SELECT\s+(.+?)\s+FROM', simplified_query, re.IGNORECASE | re.DOTALL)
                if select_match:
                    select_cols = select_match.group(1).strip()
                    # Remover prefixos de tabela das colunas selecionadas
                    for prefix in ['p.', 'oi.', 'o.', 'c.']:
                        select_cols = select_cols.replace(prefix, '')
                    
                    # Reconstruir a consulta simplificada
                    simplified_query = f"SELECT {select_cols} FROM {csv_table}"
                    
                    # Adicionar GROUP BY se existir na consulta original
                    group_match = re.search(r'GROUP BY\s+(.+?)(?:\s+ORDER BY|\s+LIMIT|$)', query, re.IGNORECASE | re.DOTALL)
                    if group_match:
                        group_cols = group_match.group(1).strip()
                        # Remover prefixos de tabela
                        for prefix in ['p.', 'oi.', 'o.', 'c.']:
                            group_cols = group_cols.replace(prefix, '')
                        simplified_query += f" GROUP BY {group_cols}"
                    
                    # Adicionar ORDER BY se existir
                    order_match = re.search(r'ORDER BY\s+(.+?)(?:\s+LIMIT|$)', query, re.IGNORECASE | re.DOTALL)
                    if order_match:
                        order_cols = order_match.group(1).strip()
                        # Remover prefixos de tabela
                        for prefix in ['p.', 'oi.', 'o.', 'c.']:
                            order_cols = order_cols.replace(prefix, '')
                        simplified_query += f" ORDER BY {order_cols}"
                    
                    # Adicionar LIMIT se existir
                    limit_match = re.search(r'LIMIT\s+(\d+)', query, re.IGNORECASE)
                    if limit_match:
                        limit = limit_match.group(1)
                        simplified_query += f" LIMIT {limit}"
                
                logger.info(f"Consulta simplificada: {simplified_query}")
                query = simplified_query
        
        # Chave de cache composta pelo conector + query
        cache_key = f"{id(selected_connector)}:{original_query}"
        
        # Verificar cache se habilitado
        if use_cache and self.cache:
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                return QueryResult(
                    data=cached_result, 
                    execution_time=time.time() - start_time, 
                    from_cache=True
                )
        
        # Executar consulta
        try:
            result_df = selected_connector.execute_query(query, params)
            
            # Calcular tempo de execução
            execution_time = time.time() - start_time
            
            # Criar objeto de resultado
            query_result = QueryResult(
                data=result_df, 
                execution_time=execution_time
            )
            
            # Armazenar em cache se habilitado
            if use_cache and self.cache:
                self.cache.set(cache_key, result_df)
            
            return query_result
        
        except Exception as e:
            logger.error(f"Erro ao executar consulta: {str(e)}")
            # Retornar DataFrame vazio em caso de erro
            return QueryResult(
                data=pd.DataFrame(), 
                execution_time=time.time() - start_time
            )

# Importações para visualizações
try:
    import plotly.graph_objs as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    logger.warning("Plotly não instalado. Algumas visualizações podem não funcionar.")
    PLOTLY_AVAILABLE = False

class VisualizationGenerator:
    """Gera visualizações a partir de resultados de consulta"""
    
    def __init__(self):
        """Inicializa o gerador de visualizações"""
        if not PLOTLY_AVAILABLE:
            logger.warning("Inicializando VisualizationGenerator sem suporte a Plotly.")
    
    def generate_visualization_code(self, 
                                   result: QueryResult, 
                                   viz_type: str, 
                                   options: Optional[Dict[str, Any]] = None) -> str:
        """
        Gera código HTML para visualização
        
        Args:
            result: Resultado da consulta
            viz_type: Tipo de visualização
            options: Opções de configuração
        
        Returns:
            Código HTML da visualização
        """
        if not PLOTLY_AVAILABLE:
            return "<h2>Visualização não disponível. Biblioteca Plotly não instalada.</h2>"
            
        try:
            # Validar dados
            if result.row_count == 0:
                return "<h2>Sem dados para visualizar</h2>"
            
            # Opções padrão
            options = options or {}
            title = options.get('title', 'Visualização de Dados')
            
            # Validar colunas
            if viz_type in ['bar', 'line', 'pie']:
                cols = list(result.data.columns)
                if not cols:
                    return "<h2>Dados sem colunas para visualizar</h2>"
            
            # Gerar visualização baseada no tipo
            if viz_type == 'bar':
                # Obter e validar colunas
                x_col = options.get('x_column', cols[0])
                y_col = options.get('y_column', cols[1] if len(cols) > 1 else cols[0])
                
                if x_col not in cols or y_col not in cols:
                    return f"<h2>Colunas especificadas ({x_col}, {y_col}) não encontradas nos dados</h2>"
                
                fig = px.bar(
                    result.data, 
                    x=x_col, 
                    y=y_col, 
                    title=title,
                    labels={x_col: options.get('x_label', x_col), 
                            y_col: options.get('y_label', y_col)},
                    color=options.get('color_column'),
                    barmode=options.get('barmode', 'group'),
                    template=options.get('template', 'plotly')
                )
                
                # Configurar layout
                fig.update_layout(
                    autosize=True,
                    height=options.get('height', 500),
                    margin=dict(l=50, r=50, b=100, t=100, pad=4)
                )
                
            elif viz_type == 'line':
                x_col = options.get('x_column', cols[0])
                y_col = options.get('y_column', cols[1] if len(cols) > 1 else cols[0])
                
                if x_col not in cols or y_col not in cols:
                    return f"<h2>Colunas especificadas ({x_col}, {y_col}) não encontradas nos dados</h2>"
                
                fig = px.line(
                    result.data, 
                    x=x_col, 
                    y=y_col, 
                    title=title,
                    labels={x_col: options.get('x_label', x_col), 
                            y_col: options.get('y_label', y_col)},
                    color=options.get('color_column'),
                    line_shape=options.get('line_shape', 'linear'),
                    template=options.get('template', 'plotly')
                )
                
                # Configurar layout
                fig.update_layout(
                    autosize=True,
                    height=options.get('height', 500),
                    margin=dict(l=50, r=50, b=100, t=100, pad=4)
                )
                
            elif viz_type == 'pie':
                names_col = options.get('names_column', cols[0])
                values_col = options.get('values_column', cols[1] if len(cols) > 1 else cols[0])
                
                if names_col not in cols or values_col not in cols:
                    return f"<h2>Colunas especificadas ({names_col}, {values_col}) não encontradas nos dados</h2>"
                
                fig = px.pie(
                    result.data, 
                    names=names_col, 
                    values=values_col, 
                    title=title,
                    hole=options.get('hole', 0),
                    template=options.get('template', 'plotly')
                )
                
                # Configurar layout
                fig.update_layout(
                    autosize=True,
                    height=options.get('height', 500),
                    margin=dict(l=50, r=50, b=100, t=100, pad=4)
                )
                
            elif viz_type == 'scatter':
                x_col = options.get('x_column', cols[0])
                y_col = options.get('y_column', cols[1] if len(cols) > 1 else cols[0])
                
                if x_col not in cols or y_col not in cols:
                    return f"<h2>Colunas especificadas ({x_col}, {y_col}) não encontradas nos dados</h2>"
                
                fig = px.scatter(
                    result.data, 
                    x=x_col, 
                    y=y_col, 
                    title=title,
                    labels={x_col: options.get('x_label', x_col), 
                            y_col: options.get('y_label', y_col)},
                    color=options.get('color_column'),
                    size=options.get('size_column'),
                    template=options.get('template', 'plotly')
                )
                
                # Configurar layout
                fig.update_layout(
                    autosize=True,
                    height=options.get('height', 500),
                    margin=dict(l=50, r=50, b=100, t=100, pad=4)
                )
                
            elif viz_type == 'table':
                fig = go.Figure(data=[go.Table(
                    header=dict(
                        values=list(result.data.columns),
                        fill_color=options.get('header_color', 'paleturquoise'),
                        align='left'
                    ),
                    cells=dict(
                        values=[result.data[col] for col in result.data.columns],
                        fill_color=options.get('cells_color', 'lavender'),
                        align='left'
                    )
                )])
                
                fig.update_layout(
                    title=title,
                    autosize=True,
                    height=options.get('height', 500),
                    margin=dict(l=50, r=50, b=100, t=100, pad=4)
                )
                
            else:
                # Tipo de visualização não suportado
                return f"<h2>Tipo de visualização não suportado: {viz_type}</h2>"
                
            # Adicionar marca d'água e rodapé
            if options.get('add_watermark', True):
                fig.add_annotation(
                    text="GenBI - Business Intelligence Generativo",
                    x=1.0,
                    y=-0.1,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=10, color="gray"),
                    opacity=0.7
                )
            
            # Converter para HTML
            return fig.to_html(
                full_html=False, 
                include_plotlyjs='cdn',
                config={
                    'displayModeBar': options.get('display_mode_bar', True),
                    'responsive': True
                }
            )
        
        except Exception as e:
            logger.error(f"Erro ao gerar visualização: {str(e)}")
            return f"<h2>Erro ao gerar visualização: {str(e)}</h2>"