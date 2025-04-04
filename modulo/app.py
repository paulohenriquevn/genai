"""
API REST para o Motor de Consulta em Linguagem Natural
=====================================================

Este módulo implementa uma API REST que expõe o motor de consulta
em linguagem natural como um serviço web utilizando FastAPI.

Para executar:
1. Instale as dependências: pip install fastapi uvicorn
2. Execute: uvicorn api_service:app --reload
3. Acesse: http://localhost:8000/docs para a documentação interativa
"""
import os
import json
import base64
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

# Import FastAPI
from fastapi import FastAPI, HTTPException, Query, Body, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Import nosso motor de consulta
from natural_query_engine import NaturalLanguageQueryEngine

# Modelos Pydantic para a API
class QueryRequest(BaseModel):
    query: str
    output_type: Optional[str] = None
    datasources: Optional[List[str]] = None

class QueryResponse(BaseModel):
    query: str
    result_type: str
    result: Any
    execution_time: float
    timestamp: str
    code_executed: Optional[str] = None

# Inicializa o aplicativo FastAPI
app = FastAPI(
    title="API de Consulta em Linguagem Natural",
    description="API para realizar consultas em linguagem natural sobre dados estruturados",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configuração CORS para permitir requisições de qualquer origem
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cria uma pasta para armazenar visualizações temporárias
os.makedirs("temp_visualizations", exist_ok=True)

# Configura a pasta de visualizações como diretório estático
app.mount("/visualizations", StaticFiles(directory="temp_visualizations"), name="visualizations")

# Inicializa o motor de consulta
engine = NaturalLanguageQueryEngine()

# Adiciona endpoints adicionais para recursos avançados
@app.post("/upload_data")
async def upload_data(
    file: UploadFile = File(...),
    data_source_name: str = Query(..., description="Nome da fonte de dados"),
    data_format: str = Query("csv", description="Formato dos dados (csv, excel, json)")
):
    """
    Faz upload de um arquivo de dados para ser usado nas consultas
    
    Args:
        file: Arquivo carregado
        data_source_name: Nome da fonte de dados
        data_format: Formato dos dados
        
    Returns:
        Mensagem de confirmação
    """
    try:
        # Salva o arquivo temporariamente
        content = await file.read()
        file_path = f"temp_data/{file.filename}"
        os.makedirs("temp_data", exist_ok=True)
        
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Infere o formato baseado na extensão se não especificado
        if data_format == "auto":
            if file.filename.endswith(".csv"):
                data_format = "csv"
            elif file.filename.endswith((".xls", ".xlsx")):
                data_format = "excel"
            elif file.filename.endswith(".json"):
                data_format = "json"
            else:
                data_format = "csv"  # Default
        
        # Carrega os dados
        if data_format == "csv":
            df = pd.read_csv(file_path)
        elif data_format == "excel":
            df = pd.read_excel(file_path)
        elif data_format == "json":
            df = pd.read_json(file_path)
        else:
            raise ValueError(f"Formato não suportado: {data_format}")
        
        # Adiciona ao motor de consulta
        from core.dataframe import DataFrameWrapper
        engine.dataframes[data_source_name] = DataFrameWrapper(df, data_source_name)
        engine.agent_state.dfs = list(engine.dataframes.values())
        
        # Retorna resposta de sucesso
        return {
            "message": f"Dados carregados com sucesso: {data_source_name}",
            "rows": len(df),
            "columns": list(df.columns)
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao carregar dados: {str(e)}"
        )

@app.get("/dataset_info/{dataset_name}")
async def get_dataset_info(dataset_name: str):
    """
    Retorna informações sobre um dataset específico
    
    Args:
        dataset_name: Nome do dataset
        
    Returns:
        Informações sobre o dataset
    """
    if dataset_name not in engine.dataframes:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset não encontrado: {dataset_name}"
        )
    
    # Obtém o DataFrame
    df = engine.dataframes[dataset_name].dataframe
    
    # Prepara informações básicas
    info = {
        "name": dataset_name,
        "rows": len(df),
        "columns": list(df.columns),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "sample": df.head(5).to_dict(orient='records'),
        "null_counts": {col: int(df[col].isna().sum()) for col in df.columns}
    }
    
    # Adiciona estatísticas para colunas numéricas
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if numeric_cols:
        info["numeric_stats"] = df[numeric_cols].describe().to_dict()
    
    return info

@app.post("/execute_sql")
async def execute_sql(query: str = Body(..., embed=True)):
    """
    Executa uma consulta SQL diretamente
    
    Args:
        query: Consulta SQL
        
    Returns:
        Resultados da consulta
    """
    try:
        result = engine.execute_sql_query(query)
        return {
            "rows": len(result),
            "columns": list(result.columns),
            "data": result.head(1000).to_dict(orient='records')  # Limita a 1000 linhas na resposta
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao executar SQL: {str(e)}"
        )

@app.get("/visualizations/{viz_id}")
async def get_visualization(viz_id: str):
    """
    Retorna uma visualização gerada
    
    Args:
        viz_id: ID da visualização
        
    Returns:
        Arquivo de imagem da visualização
    """
    file_path = f"temp_visualizations/{viz_id}"
    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=404,
            detail=f"Visualização não encontrada: {viz_id}"
        )
    
    return FileResponse(file_path)

@app.get("/", include_in_schema=False)
async def redirect_to_docs():
    """Redireciona a raiz para a documentação da API"""
    return RedirectResponse(url="/docs")

@app.get("/datasources", response_model=List[str])
async def get_datasources():
    """Retorna a lista de fontes de dados disponíveis"""
    return list(engine.dataframes.keys())

@app.get("/stats")
async def get_stats():
    """Retorna estatísticas de uso do motor de consulta"""
    return engine.get_stats()

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Processa uma consulta em linguagem natural
    
    Args:
        request: Objeto com a consulta e parâmetros opcionais
        
    Returns:
        Objeto com o resultado da consulta
    """
    try:
        # Registra o tempo de início
        start_time = datetime.now()
        
        # Configura o tipo de saída se especificado
        if request.output_type:
            engine.agent_state.output_type = request.output_type
        
        # Processa a consulta
        response = engine.execute_query(request.query)
        
        # Calcula o tempo de execução
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Processa o resultado de acordo com o tipo
        if response.type == "dataframe":
            result_data = response.value.to_dict(orient='records')
        elif response.type == "plot":
            # Salva a visualização e retorna o caminho
            visualization_id = f"viz_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
            visualization_path = f"temp_visualizations/{visualization_id}"
            response.save(visualization_path)
            
            # Retorna o caminho relativo
            result_data = {
                "visualization_url": f"/visualizations/{visualization_id}",
                "base64": response.value[:1000] + "..." if len(response.value) > 1000 else response.value
            }
        else:
            result_data = response.value
        
        # Prepara a resposta
        result = {
            "query": request.query,
            "result_type": response.type,
            "result": result_data,
            "execution_time": execution_time,
            "timestamp": datetime.now().isoformat(),
            "code_executed": response.last_code_executed
        }
        
        return result
        
    except Exception as e:
        # Registra o erro e retorna uma resposta de erro
        import traceback
        error_trace = traceback.format_exc()
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "traceback": error_trace,
                "query": request.query
            }
        )