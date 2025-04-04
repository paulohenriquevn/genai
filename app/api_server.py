"""
GenBI API Server - Módulo da API REST para o sistema GenBI
"""

from app.llm_integration.nl_processor import NLtoSQLProcessor
from fastapi import FastAPI, HTTPException, Query, Body, Depends, status, Request, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import tempfile
import json
import os
import re
import logging
import time
import random
import uuid
import shutil
from datetime import datetime, timedelta

# Configuração de logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("GenBI.API")

# Modelos de dados para a API
class QuestionRequest(BaseModel):
    """Modelo para solicitações de perguntas em linguagem natural"""
    question: str = Field(..., description="Pergunta em linguagem natural")
    use_cache: bool = Field(True, description="Se deve usar o cache para a consulta")
    explain_sql: bool = Field(False, description="Se deve explicar a consulta SQL gerada")
    connector: Optional[str] = Field(None, description="Conector de dados específico a utilizar (sqlite, csv)")
    
class CSVQuestionRequest(BaseModel):
    """Modelo para solicitações de perguntas em linguagem natural sobre CSV"""
    question: str = Field(..., description="Pergunta em linguagem natural")
    csv_filename: str = Field(..., description="Nome do arquivo CSV para consultar")
    use_cache: bool = Field(True, description="Se deve usar o cache para a consulta")
    explain_sql: bool = Field(False, description="Se deve explicar a consulta SQL gerada")
    
class SQLRequest(BaseModel):
    """Modelo para solicitações de consultas SQL diretas"""
    query: str = Field(..., description="Consulta SQL")
    params: Optional[Dict[str, Any]] = Field(None, description="Parâmetros para a consulta")
    use_cache: bool = Field(True, description="Se deve usar o cache para a consulta")
    connector: Optional[str] = Field(None, description="Conector de dados específico a utilizar (sqlite, csv)")
    
class VisualizationRequest(BaseModel):
    """Modelo para solicitações de visualizações"""
    query_id: str = Field(..., description="ID da consulta para visualizar")
    type: str = Field(..., description="Tipo de visualização (bar, line, pie, scatter, table)")
    options: Optional[Dict[str, Any]] = Field(None, description="Opções de configuração")
    
class ModelConfig(BaseModel):
    """Modelo para configuração de modelo de dados"""
    name: str
    description: Optional[str] = None
    columns: List[Dict[str, Any]]
    
class RelationshipConfig(BaseModel):
    """Modelo para configuração de relacionamento"""
    name: str
    type: str
    models: List[str]
    condition: str
    
class CatalogConfig(BaseModel):
    """Modelo para configuração do catálogo de dados"""
    models: List[ModelConfig]
    relationships: Optional[List[RelationshipConfig]] = None
    
class CSVUploadResponse(BaseModel):
    """Modelo para resposta de upload de CSV"""
    filename: str
    row_count: int
    columns: List[str]
    preview: List[Dict[str, Any]]
    schema: Dict[str, Any]

# Criação da aplicação FastAPI
app = FastAPI(
    title="GenBI API",
    description="API para Business Intelligence Generativa",
    version="1.0.0"
)

# Configuração de CORS
allowed_origins = os.environ.get("ALLOWED_ORIGINS", "http://localhost:8000,http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,  # Lista de domínios permitidos
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
)

# Sistema de armazenamento de resultados com expiração
class QueryResultsStorage:
    """Classe para armazenamento de resultados de consultas com expiração automática"""
    
    def __init__(self, ttl: int = 3600):
        """
        Inicializa o armazenamento
        
        Args:
            ttl: Tempo de vida dos resultados em segundos (padrão: 1 hora)
        """
        self.storage = {}
        self.ttl = ttl
        
    def set(self, key: str, value: Any):
        """Armazena um resultado com timestamp"""
        self.storage[key] = {
            'value': value,
            'timestamp': time.time()
        }
        
    def get(self, key: str) -> Optional[Any]:
        """Recupera um resultado se existir e não estiver expirado"""
        if key not in self.storage:
            return None
            
        item = self.storage[key]
        # Verificar se expirou
        if time.time() - item['timestamp'] > self.ttl:
            del self.storage[key]
            return None
            
        return item['value']
        
    def clean_expired(self) -> int:
        """Remove itens expirados e retorna o número de itens removidos"""
        now = time.time()
        expired_keys = [
            k for k, v in self.storage.items() 
            if now - v['timestamp'] > self.ttl
        ]
        
        for key in expired_keys:
            del self.storage[key]
            
        return len(expired_keys)
        
    def clear(self):
        """Limpa todo o armazenamento"""
        count = len(self.storage)
        self.storage = {}
        return count

# Inicializar armazenamento de resultados
query_results_storage = QueryResultsStorage(ttl=3600)  # 1 hora de TTL

# Definir middleware para limitar taxa de requisições
class RateLimiter:
    """Middleware para limitar taxa de requisições"""
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.clients = {}
        
    async def __call__(self, request: Request, call_next):
        # Obter IP do cliente
        client_ip = request.client.host
        
        # Verificar se o cliente já está no dicionário
        if client_ip not in self.clients:
            self.clients[client_ip] = {
                'count': 0,
                'reset_time': time.time() + 60  # 1 minuto
            }
        
        # Verificar se o tempo expirou e reiniciar contagem
        if time.time() > self.clients[client_ip]['reset_time']:
            self.clients[client_ip] = {
                'count': 0,
                'reset_time': time.time() + 60  # 1 minuto
            }
        
        # Incrementar contagem
        self.clients[client_ip]['count'] += 1
        
        # Verificar se excedeu limite
        if self.clients[client_ip]['count'] > self.requests_per_minute:
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Muitas requisições. Tente novamente mais tarde."
                }
            )
        
        # Limpar clientes expirados ocasionalmente
        if random.random() < 0.01:  # 1% de chance de limpar
            self._clean_expired_clients()
        
        # Continuar com a requisição
        return await call_next(request)
    
    def _clean_expired_clients(self):
        """Limpa clientes expirados"""
        now = time.time()
        expired_clients = [
            ip for ip, data in self.clients.items()
            if now > data['reset_time']
        ]
        
        for ip in expired_clients:
            del self.clients[ip]

# Classe para inicialização e gerenciamento do sistema GenBI
class GenBISystem:
    """Classe principal para gerenciar o sistema GenBI"""
    
    def __init__(self, config_path: str = "config/config.json"):
        """
        Inicializa o sistema GenBI
        
        Args:
            config_path: Caminho para o arquivo de configuração
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # Inicializar componentes
        self._init_components()
        
    def _load_config(self) -> Dict[str, Any]:
        """Carrega a configuração do sistema"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Arquivo de configuração {self.config_path} não encontrado. Usando padrões.")
            return {
                "data_sources": [
                    {
                        "type": "sqlite",
                        "config": {
                            "database": "database.db"
                        }
                    }
                ],
                "llm": {
                    "provider": "openai",
                    "config": {
                        "api_key": os.environ.get("OPENAI_API_KEY"),
                        "model": "gpt-4o"
                    }
                },
                "cache": {
                    "enabled": True,
                    "ttl": 3600  # 1 hora
                },
                "catalog": {
                    "models": [],
                    "relationships": []
                }
            }
    
    def _init_components(self):
        """Inicializa os componentes do sistema"""
        from app.data_connector.data_connector import SQLiteConnector, CSVConnector
        from app.llm_integration.llm_client import OpenAIClient
        from app.llm_integration.nl_processor import NLtoSQLProcessor
        from app.query_executor.query_executor import QueryExecutor, QueryCache
        
        # Inicializar cache se habilitado
        self.cache = None
        if self.config.get("cache", {}).get("enabled", False):
            cache_config = self.config["cache"]
            self.cache = QueryCache(
                cache_dir=cache_config.get("dir", "cache"),
                ttl=cache_config.get("ttl", 3600)
            )
        
        # Inicializar conectores de dados
        self.data_connectors = {}
        self.default_connector = None
        
        # Inicializar conector CSV
        csv_config = self.config.get("csv", {})
        csv_dir = csv_config.get("directory", "uploads/csv")
        self.data_connectors["csv"] = CSVConnector(csv_dir=csv_dir)
        self.data_connectors["csv"].connect()  # Carregar CSVs existentes
        
        # Inicializar conectores configurados
        if self.config.get("data_sources"):
            for ds_config in self.config["data_sources"]:
                if ds_config["type"] == "sqlite":
                    connector = SQLiteConnector(ds_config["config"]["database"])
                    self.data_connectors["sqlite"] = connector
                    # Define o primeiro conector encontrado como padrão, se ainda não definido
                    if self.default_connector is None:
                        self.default_connector = "sqlite"
                # Adicionar aqui suporte para outros tipos de conectores
                else:
                    logger.warning(f"Tipo de fonte de dados não totalmente suportado: {ds_config['type']}")
            
        # Se não tiver um conector padrão definido, usar CSV
        if self.default_connector is None:
            if "csv" in self.data_connectors and self.data_connectors["csv"].test_connection():
                self.default_connector = "csv"
            else:
                raise ValueError("Nenhuma fonte de dados válida configurada")
                
        # Registrar data_connector para compatibilidade com código existente
        self.data_connector = self.data_connectors[self.default_connector]
            
        # Inicializar cliente LLM
        llm_config = self.config.get("llm", {})
        if llm_config.get("provider") == "openai":
            self.llm_client = OpenAIClient(
                api_key=llm_config.get("config", {}).get("api_key"),
                model=llm_config.get("config", {}).get("model", "gpt-4o")
            )
        else:
            raise ValueError(f"Provedor LLM não suportado: {llm_config.get('provider')}")
        
        # Inicializar processador NL para SQL
        self.nl_processor = NLtoSQLProcessor(
            llm_client=self.llm_client,
            catalog_info=self.config.get("catalog", {})
        )
        
        # Inicializar executor de consultas com todos os conectores
        self.query_executor = QueryExecutor(
            data_connector=self.data_connector,  # Conector padrão
            cache=self.cache,
            data_connectors=self.data_connectors  # Todos os conectores
        )
        
    def update_catalog(self, catalog_config: Dict[str, Any]):
        """
        Atualiza a configuração do catálogo de dados
        
        Args:
            catalog_config: Nova configuração de catálogo
        """
        # Atualizar configuração
        self.config["catalog"] = catalog_config
        
        # Salvar configuração
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
            
        # Reinicializar processador NL para SQL
        self.nl_processor = NLtoSQLProcessor(
            llm_client=self.llm_client,
            catalog_info=catalog_config
        )
        
        # Invalidar cache
        if self.cache:
            self.cache.invalidate()
    
    def process_natural_language_query(self, question: str, 
                                      use_cache: bool = True,
                                      explain_sql: bool = False,
                                      connector: Optional[str] = None) -> Dict[str, Any]:
        """
        Processa uma pergunta em linguagem natural
        
        Args:
            question: Pergunta em linguagem natural
            use_cache: Se deve usar o cache
            explain_sql: Se deve explicar a consulta SQL
            connector: Nome do conector a utilizar (opcional)
            
        Returns:
            Dict: Resultados da consulta e metadados
        """
        # Gerar SQL a partir da pergunta
        sql = self.nl_processor.generate_sql(question)
        
        # Executar consulta com o conector especificado ou o padrão
        result = self.query_executor.execute(sql, use_cache=use_cache, connector_name=connector)
        
        # Gerar explicação se solicitado
        explanation = None
        if explain_sql:
            explanation = self.nl_processor.explain_sql(sql, question)
            
        # Gerar ID único para a consulta
        query_id = str(uuid.uuid4())
        
        # Armazenar resultados no sistema de armazenamento
        query_results_storage.set(query_id, result)
        
        # Construir resposta
        response = {
            "query_id": query_id,
            "question": question,
            "sql": sql,
            "data": result.data.to_dict(orient='records'),
            "metadata": {
                "execution_time": result.execution_time,
                "row_count": result.row_count,
                "column_count": result.column_count,
                "from_cache": result.from_cache,
                "executed_at": result.executed_at.isoformat()
            }
        }
        
        if explanation:
            response["explanation"] = explanation
            
        return response
        
    def process_csv_question(self, question: str, 
                             csv_filename: str,
                             use_cache: bool = True,
                             explain_sql: bool = False) -> Dict[str, Any]:
        """
        Processa uma pergunta sobre um arquivo CSV específico
        
        Args:
            question: Pergunta em linguagem natural
            csv_filename: Nome do arquivo CSV
            use_cache: Se deve usar o cache
            explain_sql: Se deve explicar a consulta SQL
            
        Returns:
            Dict: Resultados da consulta e metadados
        """
        # Verificar se o conector CSV está disponível
        if "csv" not in self.data_connectors:
            raise ValueError("Conector CSV não disponível no sistema")
            
        csv_connector = self.data_connectors["csv"]
        
        # Verificar se o arquivo CSV existe
        if not csv_filename.lower().endswith('.csv'):
            csv_filename += '.csv'
            
        available_files = csv_connector.list_tables()
        if csv_filename not in available_files:
            raise ValueError(f"Arquivo CSV '{csv_filename}' não encontrado. Disponíveis: {', '.join(available_files)}")
        
        # Obter schema do arquivo CSV para incluir no prompt
        try:
            csv_schema = csv_connector.get_table_schema(csv_filename)
        except Exception as e:
            logger.warning(f"Erro ao obter schema do CSV: {str(e)}")
            csv_schema = {"name": csv_filename, "columns": {}}
        
        # Criar cópia temporária do catalog_info para usar apenas para esta consulta
        temp_catalog_info = {"models": []}
        
        # Construir modelo temporário baseado na estrutura real do CSV
        csv_model = {
            "name": os.path.splitext(csv_filename)[0],  # Nome do arquivo sem extensão
            "description": f"Dados do arquivo CSV {csv_filename}",
            "columns": []
        }
        
        # Adicionar todas as colunas do CSV como colunas no modelo temporário
        for col_name, col_info in csv_schema.get("columns", {}).items():
            col_type = col_info.get("type", "string")
            semantic_type = col_info.get("semantic_type", "")
            
            # Mapear tipo pandas para tipo mais genérico para o modelo
            column_def = {
                "name": col_name,
                "type": col_type
            }
            
            if semantic_type:
                column_def["semanticType"] = semantic_type
                
            csv_model["columns"].append(column_def)
        
        # Adicionar modelo temporário ao catálogo temporário
        temp_catalog_info["models"].append(csv_model)
        
        # Criar um processador NL temporário com o schema real do CSV
        temp_nl_processor = NLtoSQLProcessor(
            llm_client=self.llm_client,
            catalog_info=temp_catalog_info
        )
        
        # Construir uma descrição explícita das colunas para incluir na pergunta
        column_names = ", ".join([col["name"] for col in csv_model["columns"]])
        enhanced_question = f"Com base no arquivo CSV '{csv_filename}' que possui as colunas: {column_names}, responda a seguinte pergunta: {question}"
        
        # Gerar SQL usando o processador temporário com o schema correto
        try:
            sql = temp_nl_processor.generate_sql(enhanced_question)
        except Exception as e:
            logger.warning(f"Erro ao gerar SQL via NL processor: {str(e)}")
            # Criar uma consulta simples baseada em palavras-chave na pergunta
            if "categoria" in question.lower() or "category" in question.lower():
                sql = f"SELECT category, SUM(total_amount) AS total_sales FROM {csv_filename} GROUP BY category ORDER BY total_sales DESC"
                logger.info(f"Usando SQL simplificado baseado em categoria: {sql}")
            elif "produto" in question.lower() or "product" in question.lower():
                limit = 5
                if "top" in question.lower():
                    # Extrair número após "top"
                    import re
                    matches = re.findall(r'top\s+(\d+)', question.lower())
                    if matches:
                        try:
                            limit = int(matches[0])
                        except:
                            pass
                sql = f"SELECT product, SUM(quantity) AS total_quantity, SUM(total_amount) AS total_sales FROM {csv_filename} GROUP BY product ORDER BY total_quantity DESC LIMIT {limit}"
                logger.info(f"Usando SQL simplificado baseado em produto: {sql}")
            else:
                # Consulta genérica
                sql = f"SELECT * FROM {csv_filename} LIMIT 10"
                logger.info(f"Usando SQL genérico: {sql}")
        
        # Garantir que a consulta tem o FROM corretamente
        if "FROM" not in sql.upper():
            # Adicionar FROM se não existir
            sql = f"{sql.rstrip(';')} FROM {csv_filename};"
        elif not re.search(r'FROM\s+["\']?' + re.escape(csv_filename) + r'["\']?', sql, re.IGNORECASE):
            # Substituir o FROM existente
            sql = re.sub(r'FROM\s+([^\s,;]+)', f"FROM {csv_filename}", sql, flags=re.IGNORECASE)
        
        # Verificar se as colunas mencionadas na consulta realmente existem no CSV
        # Extrair nomes de colunas da consulta SELECT
        select_match = re.search(r'SELECT\s+(.+?)\s+FROM', sql, re.IGNORECASE)
        if select_match:
            select_cols = select_match.group(1).strip()
            if select_cols != '*':
                # Remover funções de agregação para verificar apenas nomes de coluna
                select_cols = re.sub(r'(?:SUM|AVG|COUNT|MIN|MAX)\s*\(\s*([^)]+)\s*\)(?:\s+AS\s+[^,]+)?', r'\1', select_cols)
                select_cols = [col.strip().split(' AS ')[0].strip() for col in select_cols.split(',')]
                
                # Verificar se todas as colunas existem
                csv_columns = [col_name for col_name in csv_schema.get("columns", {}).keys()]
                for col in select_cols:
                    if col != '*' and col not in csv_columns and not re.match(r'(?:SUM|AVG|COUNT|MIN|MAX)', col):
                        logger.warning(f"Coluna '{col}' na consulta SQL não existe no CSV. Colunas disponíveis: {csv_columns}")
                        # Tentar mapear automaticamente nomes de colunas
                        if col.lower() == 'categoria' and 'category' in csv_columns:
                            sql = sql.replace(col, 'category')
                        elif col.lower() == 'receita_total' and 'total_amount' in csv_columns:
                            sql = sql.replace(col, 'total_amount')
                        elif col.lower() == 'quantidade_vendida' and 'quantity' in csv_columns:
                            sql = sql.replace(col, 'quantity')
        
        # Executar consulta com o conector CSV
        result = self.query_executor.execute(sql, use_cache=use_cache, connector_name="csv")
        
        # Gerar explicação se solicitado
        explanation = None
        if explain_sql:
            explanation = self.nl_processor.explain_sql(sql, question)
            
        # Gerar ID único para a consulta
        query_id = str(uuid.uuid4())
        
        # Armazenar resultados no sistema de armazenamento
        query_results_storage.set(query_id, result)
        
        # Construir resposta
        response = {
            "query_id": query_id,
            "question": question,
            "csv_filename": csv_filename,
            "sql": sql,
            "data": result.data.to_dict(orient='records'),
            "metadata": {
                "execution_time": result.execution_time,
                "row_count": result.row_count,
                "column_count": result.column_count,
                "from_cache": result.from_cache,
                "executed_at": result.executed_at.isoformat()
            }
        }
        
        if explanation:
            response["explanation"] = explanation
            
        return response
        
    def upload_csv(self, file_path: str, filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Faz upload de um arquivo CSV para o sistema
        
        Args:
            file_path: Caminho para o arquivo temporário
            filename: Nome de arquivo opcional
            
        Returns:
            Dict: Informações sobre o arquivo carregado
        """
        # Verificar se o conector CSV está disponível
        if "csv" not in self.data_connectors:
            raise ValueError("Conector CSV não disponível no sistema")
            
        csv_connector = self.data_connectors["csv"]
        
        # Fazer upload do CSV
        try:
            uploaded_filename = csv_connector.upload_csv(file_path, filename)
            
            # Obter schema do arquivo
            schema = csv_connector.get_table_schema(uploaded_filename)
            
            # Obter preview dos dados
            df = csv_connector.loaded_files[uploaded_filename]
            
            # Criar resposta
            response = {
                "filename": uploaded_filename,
                "row_count": len(df),
                "columns": list(df.columns),
                "preview": df.head(5).to_dict(orient='records'),
                "schema": schema
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Erro ao fazer upload do CSV: {str(e)}")
            raise
    
    def execute_sql_query(self, query: str, 
                         params: Optional[Dict[str, Any]] = None,
                         use_cache: bool = True,
                         connector: Optional[str] = None) -> Dict[str, Any]:
        """
        Executa uma consulta SQL direta
        
        Args:
            query: Consulta SQL
            params: Parâmetros para a consulta
            use_cache: Se deve usar o cache
            connector: Nome do conector a utilizar (opcional)
            
        Returns:
            Dict: Resultados da consulta e metadados
        """
        # Validar SQL para segurança
        unsafe_patterns = [
            "DROP ", "DELETE ", "TRUNCATE ", "UPDATE ", "INSERT ", 
            "ALTER ", "CREATE ", "GRANT ", "REVOKE ", "PRAGMA ", 
            "ATTACH ", "DETACH "
        ]
        
        if any(pattern.upper() in query.upper() for pattern in unsafe_patterns):
            raise ValueError("Consulta SQL contém comandos potencialmente perigosos")
            
        # Executar consulta com o conector especificado ou o padrão
        result = self.query_executor.execute(query, params, use_cache=use_cache, connector_name=connector)
        
        # Gerar ID único para a consulta
        query_id = str(uuid.uuid4())
        
        # Armazenar resultados no sistema de armazenamento
        query_results_storage.set(query_id, result)
        
        # Construir resposta
        response = {
            "query_id": query_id,
            "sql": query,
            "data": result.data.to_dict(orient='records'),
            "metadata": {
                "execution_time": result.execution_time,
                "row_count": result.row_count,
                "column_count": result.column_count,
                "from_cache": result.from_cache,
                "executed_at": result.executed_at.isoformat()
            }
        }
            
        return response
    
    def generate_visualization(self, query_id: str, 
                             viz_type: str,
                             options: Optional[Dict[str, Any]] = None) -> str:
        """
        Gera visualização para resultados de consulta
        
        Args:
            query_id: ID da consulta
            viz_type: Tipo de visualização
            options: Opções de configuração
            
        Returns:
            str: Código HTML da visualização
        """
        from app.query_executor.query_executor import VisualizationGenerator
        
        # Verificar se a consulta existe
        result = query_results_storage.get(query_id)
        if not result:
            raise ValueError(f"Consulta com ID {query_id} não encontrada ou expirada")
            
        # Validar tipo de visualização
        allowed_viz_types = ['bar', 'line', 'pie', 'scatter', 'table']
        if viz_type not in allowed_viz_types:
            raise ValueError(f"Tipo de visualização '{viz_type}' não suportado. Tipos suportados: {', '.join(allowed_viz_types)}")
            
        # Inicializar gerador de visualização
        viz_generator = VisualizationGenerator()
        
        # Sanitizar opções para evitar injeção de código
        safe_options = options or {}
        
        # Verificar tipos de valores nas opções
        for key, value in safe_options.items():
            if not isinstance(value, (str, int, float, bool, list, dict, type(None))):
                safe_options[key] = str(value)
                
        # Gerar HTML
        html_result = viz_generator.generate_visualization_code(
            result=result,
            viz_type=viz_type,
            options=safe_options
        )
        
        return html_result

# Instância do sistema GenBI
genbi_system = None

# Configurar middleware de rate limiting
# Rate limiter será adicionado manualmente, não via add_middleware
rate_limit = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "60"))
rate_limiter = RateLimiter(requests_per_minute=rate_limit)

# Adicionar middleware manualmente
@app.middleware("http")
async def rate_limiting_middleware(request: Request, call_next):
    return await rate_limiter(request, call_next)

@app.on_event("startup")
async def startup_event():
    """Evento de inicialização do servidor"""
    global genbi_system
    
    # Carregar configuração do ambiente se disponível
    config_path = os.environ.get("GENBI_CONFIG_PATH", "config/config.json")
    
    # Inicializar sistema GenBI
    genbi_system = GenBISystem(config_path=config_path)
    logger.info(f"Sistema GenBI inicializado com configuração de {config_path}")
    
    # Agendar tarefa de limpeza de cache
    @app.on_event("shutdown")
    async def shutdown_event():
        """Evento de finalização do servidor"""
        logger.info("Finalizando sistema GenBI")
        
        # Limpar recursos
        if hasattr(genbi_system, 'data_connector') and genbi_system.data_connector:
            genbi_system.data_connector.disconnect()
            
        # Limpar cache de consultas em memória
        removed = query_results_storage.clear()
        logger.info(f"Cache limpo: {removed} itens removidos")

# Endpoint para verificação de saúde
@app.get("/health")
async def health_check():
    """Endpoint para verificação de saúde"""
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

# Rotas da API
@app.get("/")
async def root():
    """Endpoint raiz"""
    return {"message": "GenBI API", "version": "1.0.0"}

@app.post("/query/natural_language")
async def natural_language_query(request: QuestionRequest):
    """Endpoint para consultas em linguagem natural"""
    try:
        response = genbi_system.process_natural_language_query(
            question=request.question,
            use_cache=request.use_cache,
            explain_sql=request.explain_sql,
            connector=request.connector
        )
        return response
    except Exception as e:
        logger.error(f"Erro ao processar consulta em linguagem natural: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao processar consulta: {str(e)}"
        )

@app.post("/query/csv")
async def csv_question(request: CSVQuestionRequest):
    """Endpoint para consultas em linguagem natural sobre CSV"""
    try:
        response = genbi_system.process_csv_question(
            question=request.question,
            csv_filename=request.csv_filename,
            use_cache=request.use_cache,
            explain_sql=request.explain_sql
        )
        return response
    except ValueError as e:
        # Tratar erros esperados como 404 ou 400
        if "não encontrado" in str(e):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(e)
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
    except Exception as e:
        logger.error(f"Erro ao processar consulta CSV: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao processar consulta: {str(e)}"
        )

@app.post("/query/sql")
async def sql_query(request: SQLRequest):
    """Endpoint para consultas SQL diretas"""
    try:
        response = genbi_system.execute_sql_query(
            query=request.query,
            params=request.params,
            use_cache=request.use_cache,
            connector=request.connector
        )
        return response
    except Exception as e:
        logger.error(f"Erro ao executar consulta SQL: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao executar consulta: {str(e)}"
        )

@app.post("/csv/upload")
async def upload_csv_file(file: UploadFile = File(...), custom_filename: Optional[str] = Form(None)):
    """Endpoint para upload de arquivo CSV"""
    if not file.filename.lower().endswith('.csv'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Apenas arquivos CSV são permitidos"
        )
        
    try:
        # Criar arquivo temporário para o upload
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_file:
            temp_path = temp_file.name
            # Escrever conteúdo no arquivo temporário
            shutil.copyfileobj(file.file, temp_file)
            
        # Processar o arquivo
        try:
            result = genbi_system.upload_csv(temp_path, custom_filename or file.filename)
            return result
        finally:
            # Limpar arquivo temporário
            os.unlink(temp_path)
            
    except Exception as e:
        logger.error(f"Erro ao fazer upload do CSV: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao processar arquivo CSV: {str(e)}"
        )
        
@app.get("/csv/list")
async def list_csv_files():
    """Lista todos os arquivos CSV disponíveis no sistema"""
    try:
        if "csv" not in genbi_system.data_connectors:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Suporte a CSV não está disponível"
            )
            
        csv_connector = genbi_system.data_connectors["csv"]
        files = csv_connector.list_tables()
        
        # Obter informações básicas sobre cada arquivo
        result = []
        for filename in files:
            try:
                schema = csv_connector.get_table_schema(filename)
                df = csv_connector.loaded_files[filename]
                file_info = {
                    "filename": filename,
                    "row_count": len(df),
                    "column_count": len(df.columns),
                    "column_names": list(df.columns)
                }
                result.append(file_info)
            except Exception as e:
                logger.warning(f"Erro ao obter informações do arquivo {filename}: {str(e)}")
                # Incluir informações mínimas
                result.append({
                    "filename": filename,
                    "error": str(e)
                })
                
        return {"files": result}
    except Exception as e:
        logger.error(f"Erro ao listar arquivos CSV: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao listar arquivos CSV: {str(e)}"
        )

@app.post("/visualization")
async def generate_visualization(request: VisualizationRequest):
    """Endpoint para gerar visualizações"""
    try:
        html = genbi_system.generate_visualization(
            query_id=request.query_id,
            viz_type=request.type,
            options=request.options
        )
        return HTMLResponse(content=html)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Erro ao gerar visualização: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao gerar visualização: {str(e)}"
        )

@app.put("/catalog")
async def update_catalog(catalog: CatalogConfig):
    """Endpoint para atualizar o catálogo de dados"""
    try:
        genbi_system.update_catalog(catalog.dict())
        return {"message": "Catálogo atualizado com sucesso"}
    except Exception as e:
        logger.error(f"Erro ao atualizar catálogo: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao atualizar catálogo: {str(e)}"
        )

@app.get("/catalog")
async def get_catalog():
    """Endpoint para obter o catálogo de dados atual"""
    return genbi_system.config.get("catalog", {})

@app.delete("/cache")
async def clear_cache():
    """Endpoint para limpar o cache"""
    if genbi_system.cache:
        count = genbi_system.cache.invalidate()
        return {"message": f"Cache limpo com sucesso. {count} entradas removidas."}
    else:
        return {"message": "Cache não está habilitado"}