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

# Armazena informações da sessão do usuário
session_data: Dict[str, Dict[str, Any]] = {}

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
        
        # Inicializa dados da sessão se não existirem
        if file_id not in session_data:
            session_data[file_id] = {}
        
        # Armazena o último resultado e consulta na sessão
        session_data[file_id]["last_query"] = query
        session_data[file_id]["last_result"] = result
        
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
            # Adiciona indicador de que uma visualização está disponível
            response["visualization_available"] = True
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
    chart_type: Optional[str] = Form(None),
    x_column: Optional[str] = Form(None),
    y_column: Optional[str] = Form(None),
    title: Optional[str] = Form(None)
):
    """
    Gera uma visualização inteligente baseada na última consulta do usuário.
    Se parâmetros específicos não forem fornecidos, determina automaticamente o tipo de gráfico mais adequado.
    Retorna configuração JSON do gráfico (ApexCharts).
    """
    # Verifica se o ID do arquivo existe
    if file_id not in engines:
        raise HTTPException(status_code=404, detail="Arquivo não encontrado")
    
    # Verifica se existe uma consulta anterior
    if file_id not in session_data or "last_result" not in session_data[file_id]:
        raise HTTPException(status_code=400, detail="Nenhuma consulta anterior encontrada. Faça uma consulta primeiro.")
    
    engine = engines[file_id]
    last_result = session_data[file_id]["last_result"]
    last_query = session_data[file_id].get("last_query", "")
    
    try:
        # Se o último resultado já for um gráfico, retorna-o diretamente
        if last_result.type == "chart" and hasattr(last_result, "chart_format") and last_result.chart_format == "apex":
            return JSONResponse(content={
                "chart": last_result.to_apex_json(),
                "type": "chart",
                "query": last_query,
                "description": "Visualização da última consulta"
            })
        
        # Se o último resultado for um DataFrame, gera uma visualização
        if last_result.type == "dataframe":
            df = last_result.value
            
            # Obtém o conjunto de dados
            data = df
            
            # Determina colunas x e y automaticamente se não fornecidas
            if not x_column:
                # Tenta encontrar a primeira coluna categórica ou de data
                for col in df.columns:
                    if (df[col].dtype == 'object' or 
                        pd.api.types.is_datetime64_any_dtype(df[col]) or 
                        col.lower() in ['data', 'date', 'mes', 'month', 'ano', 'year', 'period', 'período', 'categoria', 'category']):
                        x_column = col
                        break
                
                # Se ainda não encontrou, usa a primeira coluna
                if not x_column and not df.empty and df.columns.size > 0:
                    x_column = df.columns[0]
            
            if not y_column:
                # Tenta encontrar a primeira coluna numérica
                for col in df.columns:
                    if (pd.api.types.is_numeric_dtype(df[col]) and 
                        col != x_column and 
                        col.lower() not in ['id', 'código', 'code']):
                        y_column = col
                        break
                
                # Se ainda não encontrou, usa a segunda coluna ou a primeira
                if not y_column and not df.empty:
                    if df.columns.size > 1 and df.columns[1] != x_column:
                        y_column = df.columns[1]
                    else:
                        # Fallback para a primeira coluna numérica
                        for col in df.columns:
                            if pd.api.types.is_numeric_dtype(df[col]):
                                y_column = col
                                break
            
            # Determina o tipo de gráfico mais adequado se não fornecido
            if not chart_type:
                # Analisa o conteúdo e a estrutura dos dados para sugerir um tipo de gráfico
                
                # Quantidade de valores únicos na coluna x
                unique_x_values = df[x_column].nunique() if x_column and x_column in df.columns else 0
                
                # Verifica se a coluna x é temporal
                is_time_series = (pd.api.types.is_datetime64_any_dtype(df[x_column]) if x_column and x_column in df.columns else False)
                is_date_like = (is_time_series or 
                                (x_column and x_column.lower() in ['data', 'date', 'mes', 'month', 'ano', 'year', 'dia', 'day']))
                
                # Número de colunas numéricas
                num_numeric_cols = len(df.select_dtypes(include=['number']).columns)
                
                # Análise do conteúdo da consulta para inferir intenção
                query_lower = last_query.lower()
                
                # Lógica para decidir o tipo de gráfico
                if 'pizza' in query_lower or 'pie' in query_lower or 'distribuição' in query_lower or 'distribution' in query_lower:
                    chart_type = 'pie'
                elif 'área' in query_lower or 'area' in query_lower:
                    chart_type = 'area'
                elif 'linha' in query_lower or 'line' in query_lower or 'evolução' in query_lower or 'tendência' in query_lower or 'trend' in query_lower or 'evolution' in query_lower:
                    chart_type = 'line'
                elif 'dispersão' in query_lower or 'scatter' in query_lower or 'correlação' in query_lower or 'correlation' in query_lower:
                    chart_type = 'scatter'
                elif 'calor' in query_lower or 'heat' in query_lower:
                    chart_type = 'heatmap'
                else:
                    # Decisão baseada na estrutura dos dados
                    if is_date_like and unique_x_values > 1:
                        chart_type = 'line'  # Séries temporais geralmente ficam boas em linhas
                    elif df.shape[0] <= 10:  # Poucos dados
                        if unique_x_values <= 12:
                            chart_type = 'bar'  # Para categorias pequenas, barras são mais legíveis
                        else:
                            chart_type = 'line'  # Para muitas categorias, linhas são mais compactas
                    else:  # Muitos dados
                        chart_type = 'bar'  # Padrão para exploração de dados
            
            # Define um título significativo
            if not title:
                if y_column and x_column:
                    title = f"{y_column} por {x_column}"
                    # Capitaliza a primeira letra
                    title = title[0].upper() + title[1:]
                else:
                    # Gera um título a partir da consulta
                    title = f"Visualização de {last_query[:30]}{'...' if len(last_query) > 30 else ''}"
            
            # Gera o gráfico no formato ApexCharts
            chart_response = engine.generate_chart(
                data=data,
                chart_type=chart_type,
                x=x_column,
                y=y_column,
                title=title,
                chart_format="apex"
            )
            
            # Gera uma descrição da visualização
            description = f"Visualização gerada a partir da consulta: '{last_query}'. "
            description += f"Gráfico do tipo {chart_type} mostrando {y_column} por {x_column}."
            
            # Retorna a configuração do gráfico com contexto
            return JSONResponse(content={
                "chart": chart_response.to_apex_json(),
                "type": "chart",
                "chart_type": chart_type,
                "x_column": x_column,
                "y_column": y_column,
                "query": last_query,
                "description": description
            })
        
        # Se o resultado não for um DataFrame nem um gráfico
        raise HTTPException(
            status_code=400, 
            detail="A última consulta não retornou dados que possam ser visualizados."
        )
        
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
        
        # Remove dados da sessão
        if file_id in session_data:
            del session_data[file_id]
        
        # Remove o arquivo
        await file_manager.delete_file(file_id)
        return {"status": "success", "message": "Sessão encerrada com sucesso"}
    else:
        raise HTTPException(status_code=404, detail="Sessão não encontrada")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)