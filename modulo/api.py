from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uuid
import os
from typing import Optional, Dict, Any
import pandas as pd
import json
import logging

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("api")

# Importações do sistema
from core.engine.analysis_engine import AnalysisEngine
from utils.file_manager import FileManager

app = FastAPI(title="Sistema de Consulta em Linguagem Natural API")

# Configuração de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Modificar para origens específicas em produção
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicialização do gerenciador de arquivos
file_manager = FileManager(base_dir="uploads")

# Dicionário para armazenar engines por ID de sessão
engines: Dict[str, AnalysisEngine] = {}

@app.post("/upload/")
async def upload_file(
    file: UploadFile = File(...),
    description: Optional[str] = Form(None)
):
    """
    Endpoint para upload de arquivo.
    Retorna um identificador único para o arquivo.
    """
    # Gera um ID único para o arquivo
    file_id = str(uuid.uuid4())
    
    # Salva o arquivo com o identificador único
    file_path = await file_manager.save_file(file, file_id)
    
    # Cria uma instância do motor de análise para este arquivo
    engine = AnalysisEngine(
        model_type="openai",  # Configurar com base nas variáveis de ambiente
        model_name="gpt-3.5-turbo",
        api_key=os.environ.get("OPENAI_API_KEY")
    )
    
    # Carrega o arquivo no motor de análise
    try:
        engine.load_data(
            data=file_path,
            name="dataset",
            description=description or f"Dados carregados de {file.filename}"
        )
        # Armazena o engine associado ao ID
        engines[file_id] = engine
        
        return {"file_id": file_id, "filename": file.filename, "status": "success"}
    except Exception as e:
        logger.error(f"Erro ao processar arquivo: {str(e)}")
        # Remove o arquivo em caso de erro
        await file_manager.delete_file(file_id)
        raise HTTPException(status_code=400, detail=f"Erro ao processar arquivo: {str(e)}")

@app.post("/query/")
async def process_query(
    file_id: str = Form(...),
    query: str = Form(...),
    background_tasks: BackgroundTasks = None
):
    """
    Processa uma consulta em linguagem natural sobre o arquivo.
    Retorna o resultado da consulta e uma análise.
    """
    # Verifica se o ID do arquivo existe
    if file_id not in engines:
        raise HTTPException(status_code=404, detail="Arquivo não encontrado")
    
    engine = engines[file_id]
    
    try:
        # Processa a consulta
        result = engine.process_query(query)
        
        # Prepara resposta com base no tipo de resultado
        response = {
            "type": result.type,
            "query": query,
            "analysis": engine.generate_analysis(result, query)
        }
        
        # Adiciona o valor específico baseado no tipo
        if result.type == "dataframe":
            # Converte DataFrame para JSON
            response["data"] = result.value.to_dict(orient="records")
        elif result.type == "chart":
            # Retorna configuração de gráfico
            if hasattr(result, "chart_format") and result.chart_format == "apex":
                response["chart"] = result.to_apex_json()
            else:
                # Fallback para imagem ou outro formato
                response["data"] = str(result.value)
        else:
            # Para string, number e outros tipos
            response["data"] = result.value
        
        return JSONResponse(content=response)
    except Exception as e:
        logger.error(f"Erro ao processar consulta: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao processar consulta: {str(e)}")

@app.post("/visualization/")
async def generate_visualization(
    file_id: str = Form(...),
    chart_type: str = Form(...),
    x_column: Optional[str] = Form(None),
    y_column: Optional[str] = Form(None),
    title: Optional[str] = Form(None)
):
    """
    Gera uma visualização para os dados do arquivo.
    Retorna configuração JSON do gráfico (ApexCharts).
    """
    # Verifica se o ID do arquivo existe
    if file_id not in engines:
        raise HTTPException(status_code=404, detail="Arquivo não encontrado")
    
    engine = engines[file_id]
    
    try:
        # Obtém o dataset principal
        dataset = engine.datasets.get("dataset")
        if not dataset:
            dataset = next(iter(engine.datasets.values()))
        
        # Determina colunas x e y automaticamente se não fornecidas
        if not x_column:
            # Tenta encontrar a primeira coluna categórica ou de data
            for col in dataset.dataframe.columns:
                if dataset.dataframe[col].dtype == 'object' or pd.api.types.is_datetime64_any_dtype(dataset.dataframe[col]):
                    x_column = col
                    break
            if not x_column and dataset.dataframe.columns.size > 0:
                x_column = dataset.dataframe.columns[0]  # Fallback
        
        if not y_column:
            # Tenta encontrar a primeira coluna numérica
            for col in dataset.dataframe.columns:
                if pd.api.types.is_numeric_dtype(dataset.dataframe[col]):
                    y_column = col
                    break
            if not y_column and dataset.dataframe.columns.size > 1:
                y_column = dataset.dataframe.columns[1]  # Fallback
        
        # Gera o gráfico no formato ApexCharts
        chart_response = engine.generate_chart(
            data=dataset.dataframe,
            chart_type=chart_type,
            x=x_column,
            y=y_column,
            title=title or f"Visualização de {x_column} e {y_column}",
            chart_format="apex"
        )
        
        # Retorna a configuração do gráfico
        return JSONResponse(content={
            "chart": chart_response.to_apex_json(),
            "type": "chart",
            "x_column": x_column,
            "y_column": y_column
        })
    except Exception as e:
        logger.error(f"Erro ao gerar visualização: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao gerar visualização: {str(e)}")

def _generate_analysis(result, query: str) -> str:
    """
    Gera uma análise simplificada do resultado da consulta.
    """
    if result.type == "dataframe":
        df = result.value
        return f"A consulta retornou {len(df)} registros com {len(df.columns)} colunas."
    elif result.type == "chart":
        return f"Visualização gerada com base na consulta: '{query}'."
    elif result.type == "number":
        return f"O valor numérico obtido foi {result.value}."
    else:
        return f"Consulta processada com sucesso."

@app.delete("/session/{file_id}")
async def cleanup_session(file_id: str):
    """
    Remove os recursos associados a uma sessão.
    """
    if file_id in engines:
        # Remove o engine
        del engines[file_id]
        # Remove o arquivo
        await file_manager.delete_file(file_id)
        return {"status": "success", "message": "Sessão encerrada com sucesso"}
    else:
        raise HTTPException(status_code=404, detail="Sessão não encontrada")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)