"""
GenBI API Server - Módulo da API REST para o sistema GenBI
"""

from app.llm_integration.nl_processor import NLtoSQLProcessor
from fastapi import FastAPI, HTTPException, Query, Body, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import json
import os
import logging
import time
import random
import uuid
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
    
class SQLRequest(BaseModel):
    """Modelo para solicitações de consultas SQL diretas"""
    query: str = Field(..., description="Consulta SQL")
    params: Optional[Dict[str, Any]] = Field(None, description="Parâmetros para a consulta")
    use_cache: bool = Field(True, description="Se deve usar o cache para a consulta")
    
class VisualizationRequest(BaseModel):
    """Modelo para solicitações de visualizações"""
    query_id: str = Field(..., description="ID da consulta para visualizar")
    type: str = Field(..., description="Tipo de visualização (bar, line, pie, scatter)")
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
        from app.data_connector.data_connector import SQLiteConnector
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
        
        # Inicializar conector de dados
        # Por enquanto, usar apenas o primeiro conector configurado
        # Em produção, poderia suportar múltiplos conectores
        if self.config.get("data_sources"):
            ds_config = self.config["data_sources"][0]
            if ds_config["type"] == "sqlite":
                self.data_connector = SQLiteConnector(ds_config["config"]["database"])
            else:
                raise ValueError(f"Tipo de fonte de dados não suportado: {ds_config['type']}")
        else:
            raise ValueError("Nenhuma fonte de dados configurada")
            
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
        
        # Inicializar executor de consultas
        self.query_executor = QueryExecutor(
            data_connector=self.data_connector,
            cache=self.cache
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
                                      explain_sql: bool = False) -> Dict[str, Any]:
        """
        Processa uma pergunta em linguagem natural
        
        Args:
            question: Pergunta em linguagem natural
            use_cache: Se deve usar o cache
            explain_sql: Se deve explicar a consulta SQL
            
        Returns:
            Dict: Resultados da consulta e metadados
        """
        # Gerar SQL a partir da pergunta
        sql = self.nl_processor.generate_sql(question)
        
        # Executar consulta
        result = self.query_executor.execute(sql, use_cache=use_cache)
        
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
    
    def execute_sql_query(self, query: str, 
                         params: Optional[Dict[str, Any]] = None,
                         use_cache: bool = True) -> Dict[str, Any]:
        """
        Executa uma consulta SQL direta
        
        Args:
            query: Consulta SQL
            params: Parâmetros para a consulta
            use_cache: Se deve usar o cache
            
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
            
        # Executar consulta
        result = self.query_executor.execute(query, params, use_cache=use_cache)
        
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
            explain_sql=request.explain_sql
        )
        return response
    except Exception as e:
        logger.error(f"Erro ao processar consulta em linguagem natural: {str(e)}")
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
            use_cache=request.use_cache
        )
        return response
    except Exception as e:
        logger.error(f"Erro ao executar consulta SQL: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao executar consulta: {str(e)}"
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