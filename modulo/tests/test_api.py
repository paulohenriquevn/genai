import pytest
from fastapi.testclient import TestClient
import os
import tempfile
import pandas as pd

from api import app

client = TestClient(app)

@pytest.fixture
def sample_csv_file():
    """Cria um arquivo CSV temporário para testes"""
    # Cria um DataFrame de exemplo
    df = pd.DataFrame({
        'data': ['2023-01-01', '2023-01-02', '2023-01-03'],
        'valor': [100, 200, 300],
        'categoria': ['A', 'B', 'C']
    })
    
    # Salva em um arquivo temporário
    fd, path = tempfile.mkstemp(suffix='.csv')
    os.close(fd)
    df.to_csv(path, index=False)
    
    yield path
    
    # Limpa o arquivo após o teste
    if os.path.exists(path):
        os.unlink(path)

def test_upload_file(sample_csv_file):
    """Testa o upload de arquivo"""
    with open(sample_csv_file, 'rb') as f:
        response = client.post(
            "/upload/",
            files={"file": ("test.csv", f, "text/csv")},
            data={"description": "Arquivo de teste"}
        )
    
    assert response.status_code == 200
    data = response.json()
    assert "file_id" in data
    assert data["status"] == "success"
    
    return data["file_id"]

def test_query_endpoint(sample_csv_file):
    """Testa o endpoint de consulta"""
    # Primeiro faz upload do arquivo
    file_id = test_upload_file(sample_csv_file)
    
    # Agora testa a consulta
    response = client.post(
        "/query/",
        data={
            "file_id": file_id,
            "query": "Qual é o total de valores?"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "type" in data
    assert "analysis" in data
    
    # Limpa no final
    client.delete(f"/session/{file_id}")

def test_visualization_endpoint(sample_csv_file):
    """Testa o endpoint de visualização"""
    # Primeiro faz upload do arquivo
    file_id = test_upload_file(sample_csv_file)
    
    # Agora testa a visualização
    response = client.post(
        "/visualization/",
        data={
            "file_id": file_id,
            "chart_type": "bar",
            "x_column": "categoria",
            "y_column": "valor",
            "title": "Teste de Visualização"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "chart" in data
    assert "type" in data
    assert data["type"] == "chart"
    assert "x_column" in data
    assert "y_column" in data
    
    # Verifica estrutura do gráfico ApexCharts
    chart = data["chart"]
    assert "chart" in chart
    assert "series" in chart
    assert "xaxis" in chart
    assert "title" in chart
    
    # Limpa no final
    client.delete(f"/session/{file_id}")

def test_cleanup_session():
    """Testa a limpeza de sessão"""
    # Cria um arquivo temporário e faz upload
    df = pd.DataFrame({"A": [1, 2, 3]})
    fd, path = tempfile.mkstemp(suffix='.csv')
    os.close(fd)
    df.to_csv(path, index=False)
    
    with open(path, 'rb') as f:
        response = client.post(
            "/upload/",
            files={"file": ("temp.csv", f, "text/csv")}
        )
    
    file_id = response.json()["file_id"]
    
    # Testa a limpeza
    response = client.delete(f"/session/{file_id}")
    assert response.status_code == 200
    
    # Verifica se o ID foi realmente removido
    response = client.post(
        "/query/",
        data={
            "file_id": file_id,
            "query": "Teste"
        }
    )
    assert response.status_code == 404
    
    # Limpa o arquivo temporário
    if os.path.exists(path):
        os.unlink(path)