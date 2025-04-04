#!/usr/bin/env python3
"""
API REST para o Sistema de Consulta em Linguagem Natural
=====================================================

Este módulo implementa uma API REST que expõe as funcionalidades do sistema 
como serviços web. Utiliza FastAPI para fornecer:

- Execução de consultas em linguagem natural
- Execução direta de SQL
- Gerenciamento de fontes de dados
- Estatísticas e informações do sistema
- Visualizações e análises de dados

Para executar:
1. Instale as dependências: pip install fastapi uvicorn
2. Execute: uvicorn api:app --reload
3. Acesse: http://localhost:8000/docs para a documentação interativa
"""

import os
import json
import base64
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

# Import FastAPI
from fastapi import FastAPI, HTTPException, Query, Body, UploadFile, File, Depends
from fastapi.responses import JSONResponse, FileResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd

# Referência para o motor de consulta em linguagem natural
# Esta variável será definida pelo sistema integrado
engine = None

# Modelos de dados para API
class QueryRequest(BaseModel):
    """Modelo para solicitação de consulta em linguagem natural"""
    query: str
    output_type: Optional[str] = None
    datasources: Optional[List[str]] = None

class QueryResponse(BaseModel):
    """Modelo para resposta de consulta em linguagem natural"""
    query: str
    result_type: str
    result: Any
    execution_time: float
    timestamp: str
    code_executed: Optional[str] = None

class SQLRequest(BaseModel):
    """Modelo para solicitação de consulta SQL direta"""
    query: str

# Inicializa o aplicativo FastAPI
app = FastAPI(
    title="API do Sistema de Consulta em Linguagem Natural",
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
os.makedirs("output", exist_ok=True)

# Configura a pasta de visualizações como diretório estático
app.mount("/output", StaticFiles(directory="output"), name="output")


@app.get("/")
async def root():
    """Endpoint raiz que redireciona para a documentação"""
    return RedirectResponse(url="/docs")


@app.get("/health")
async def health_check():
    """Verifica a saúde do sistema"""
    if engine is None:
        return {"status": "error", "message": "Engine not initialized"}
    
    try:
        # Verifica se o motor está funcional tentando acessar dataframes
        datasources = list(engine.dataframes.keys())
        return {
            "status": "ok",
            "message": "System is healthy",
            "datasources_count": len(datasources),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/datasources", response_model=List[str])
async def get_datasources():
    """Retorna a lista de fontes de dados disponíveis"""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    return list(engine.dataframes.keys())


@app.get("/dataset_info/{dataset_name}")
async def get_dataset_info(dataset_name: str):
    """
    Retorna informações sobre um dataset específico
    
    Args:
        dataset_name: Nome do dataset
        
    Returns:
        Informações sobre o dataset
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
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
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
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


@app.post("/execute_sql")
async def execute_sql(query: str = Body(..., embed=True)):
    """
    Executa uma consulta SQL diretamente
    
    Args:
        query: Consulta SQL
        
    Returns:
        Resultados da consulta
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
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


@app.get("/stats")
async def get_stats():
    """Retorna estatísticas de uso do motor de consulta"""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    try:
        return engine.get_stats()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao obter estatísticas: {str(e)}"
        )


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Processa uma consulta em linguagem natural
    
    Args:
        request: Objeto com a consulta e parâmetros opcionais
        
    Returns:
        Objeto com o resultado da consulta
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
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
            visualization_path = f"output/{visualization_id}"
            response.save(visualization_path)
            
            # Retorna o caminho relativo
            result_data = {
                "visualization_url": f"/output/{visualization_id}",
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
            "code_executed": response.last_code_executed if hasattr(response, 'last_code_executed') else None
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


@app.get("/visualizations/{viz_id}")
async def get_visualization(viz_id: str):
    """
    Retorna uma visualização gerada
    
    Args:
        viz_id: ID da visualização
        
    Returns:
        Arquivo de imagem da visualização
    """
    file_path = f"output/{viz_id}"
    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=404,
            detail=f"Visualização não encontrada: {viz_id}"
        )
    
    return FileResponse(file_path)


@app.get("/visualize_data/{dataset_name}")
async def visualize_data(dataset_name: str, visualization_type: str = "summary"):
    """
    Gera visualizações para um conjunto de dados
    
    Args:
        dataset_name: Nome do dataset
        visualization_type: Tipo de visualização (summary, histogram, correlation, etc.)
        
    Returns:
        Informações sobre as visualizações geradas
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    if dataset_name not in engine.dataframes:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset não encontrado: {dataset_name}"
        )
    
    try:
        # Cria diretório para salvar visualizações
        output_dir = f"output/{dataset_name}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Obtém o DataFrame
        df = engine.dataframes[dataset_name].dataframe
        
        # Listas para armazenar informações sobre visualizações
        visualizations = []
        
        # Cria visualizações com base no tipo solicitado
        if visualization_type in ["summary", "all"]:
            # Cria visualizações resumidas (histogramas e barras)
            
            # Visualizações numéricas
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            for col in numeric_cols[:3]:  # Limita a 3 colunas
                import matplotlib.pyplot as plt
                
                # Cria histograma
                plt.figure(figsize=(10, 6))
                plt.hist(df[col].dropna(), bins=20)
                plt.title(f'Distribuição de {col}')
                plt.xlabel(col)
                plt.ylabel('Frequência')
                plt.grid(True, alpha=0.3)
                
                # Salva a visualização
                filename = f"{col}_histogram.png"
                filepath = os.path.join(output_dir, filename)
                plt.savefig(filepath)
                plt.close()
                
                visualizations.append({
                    "type": "histogram",
                    "column": col,
                    "url": f"/output/{dataset_name}/{filename}"
                })
            
            # Visualizações categóricas
            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            for col in cat_cols[:3]:  # Limita a 3 colunas
                if df[col].nunique() <= 15:  # Apenas se tiver poucas categorias
                    import matplotlib.pyplot as plt
                    
                    # Cria gráfico de barras
                    plt.figure(figsize=(10, 6))
                    df[col].value_counts().plot(kind='bar')
                    plt.title(f'Contagem de {col}')
                    plt.xlabel(col)
                    plt.ylabel('Contagem')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    
                    # Salva a visualização
                    filename = f"{col}_barplot.png"
                    filepath = os.path.join(output_dir, filename)
                    plt.savefig(filepath)
                    plt.close()
                    
                    visualizations.append({
                        "type": "barplot",
                        "column": col,
                        "url": f"/output/{dataset_name}/{filename}"
                    })
        
        elif visualization_type == "correlation":
            # Cria matriz de correlação para colunas numéricas
            numeric_df = df.select_dtypes(include=['number'])
            
            if len(numeric_df.columns) >= 2:
                import matplotlib.pyplot as plt
                import seaborn as sns
                
                # Calcula a correlação
                corr = numeric_df.corr()
                
                # Cria o heatmap
                plt.figure(figsize=(12, 10))
                sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
                plt.title('Matriz de Correlação')
                plt.tight_layout()
                
                # Salva a visualização
                filename = "correlation_matrix.png"
                filepath = os.path.join(output_dir, filename)
                plt.savefig(filepath)
                plt.close()
                
                visualizations.append({
                    "type": "correlation",
                    "columns": list(numeric_df.columns),
                    "url": f"/output/{dataset_name}/{filename}"
                })
        
        elif visualization_type == "scatter":
            # Cria gráficos de dispersão entre colunas numéricas
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            
            if len(numeric_cols) >= 2:
                import matplotlib.pyplot as plt
                
                # Cria gráficos de dispersão para as primeiras 3 combinações
                for i in range(min(len(numeric_cols), 3)):
                    for j in range(i+1, min(len(numeric_cols), 4)):
                        col1 = numeric_cols[i]
                        col2 = numeric_cols[j]
                        
                        plt.figure(figsize=(10, 8))
                        plt.scatter(df[col1], df[col2], alpha=0.5)
                        plt.title(f'{col1} vs {col2}')
                        plt.xlabel(col1)
                        plt.ylabel(col2)
                        plt.grid(True, alpha=0.3)
                        
                        # Salva a visualização
                        filename = f"scatter_{col1}_vs_{col2}.png"
                        filepath = os.path.join(output_dir, filename)
                        plt.savefig(filepath)
                        plt.close()
                        
                        visualizations.append({
                            "type": "scatter",
                            "columns": [col1, col2],
                            "url": f"/output/{dataset_name}/{filename}"
                        })
        
        return {
            "dataset": dataset_name,
            "visualization_type": visualization_type,
            "visualizations": visualizations
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao gerar visualizações: {str(e)}"
        )


# Se for executado diretamente, inicia o servidor
if __name__ == "__main__":
    import uvicorn
    
    # Inicializa o motor diretamente para testes
    from natural_query_engine import NaturalLanguageQueryEngine
    engine = NaturalLanguageQueryEngine()
    
    # Inicia o servidor
    uvicorn.run(app, host="0.0.0.0", port=8000)