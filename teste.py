#!/usr/bin/env python
"""
Teste específico para o processamento de arquivos CSV no GenBI
"""

import os
import sys
import requests
import json
import logging
import traceback
import webbrowser
from typing import Dict, Any, Optional, List

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("genbi_csv_test.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("GenBI.CSVTest")

# URL base da API GenBI
BASE_URL = os.environ.get("GENBI_BASE_URL", "http://localhost:8000")

class GenBICSVClient:
    """Cliente especializado para testes com CSV no GenBI"""
    
    def __init__(self, base_url: str = BASE_URL):
        """Inicializa o cliente GenBI com a URL base da API"""
        self.base_url = base_url
        self.session = requests.Session()
        
        # Validar conexão com o servidor
        try:
            self.check_health()
            logger.info(f"Conectado com sucesso ao servidor GenBI em {base_url}")
        except Exception as e:
            logger.error(f"Erro ao conectar ao servidor GenBI: {str(e)}")
            raise
    
    def check_health(self) -> Dict[str, Any]:
        """Verifica a saúde do sistema"""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def upload_csv(self, csv_path: str) -> Dict[str, Any]:
        """Faz upload de um arquivo CSV para o sistema"""
        with open(csv_path, 'rb') as f:
            files = {'file': f}
            response = self.session.post(f"{self.base_url}/csv/upload", files=files)
            response.raise_for_status()
            return response.json()
    
    def list_csv_files(self) -> Dict[str, Any]:
        """Lista todos os arquivos CSV disponíveis no sistema"""
        response = self.session.get(f"{self.base_url}/csv/list")
        response.raise_for_status()
        return response.json()
    
    def csv_query(self, question: str, csv_filename: str, use_cache: bool = True, explain_sql: bool = True) -> Dict[str, Any]:
        """Executa uma consulta em linguagem natural sobre um arquivo CSV"""
        payload = {
            "question": question,
            "csv_filename": csv_filename,
            "use_cache": use_cache,
            "explain_sql": explain_sql
        }
        
        response = self.session.post(f"{self.base_url}/query/csv", json=payload)
        response.raise_for_status()
        return response.json()
    
    def generate_visualization(self, query_id: str, viz_type: str, options: Optional[Dict[str, Any]] = None) -> str:
        """Gera uma visualização para uma consulta"""
        payload = {
            "query_id": query_id,
            "type": viz_type,
            "options": options or {}
        }
        
        response = self.session.post(f"{self.base_url}/visualization", json=payload)
        response.raise_for_status()
        return response.text

def test_csv_processing():
    """Função de teste para processamento de CSV"""
    print("===== Teste de Processamento de CSV no GenBI =====")
    
    try:
        # Inicializar cliente
        client = GenBICSVClient()
        
        # 1. Listar arquivos CSV existentes
        print("\n1. Listando arquivos CSV disponíveis...")
        csv_files = client.list_csv_files()
        print(json.dumps(csv_files, indent=2))
        
        # 2. Executar consulta sobre categorias e vendas
        print("\n2. Consultando total de vendas por categoria no CSV...")
        question = "Qual o total de vendas por categoria?"
        
        # Usar o primeiro arquivo CSV disponível ou um específico
        if csv_files.get('files'):
            csv_filename = csv_files['files'][0]['filename']
        else:
            csv_filename = "example_sales.csv"
            
        print(f"Usando arquivo CSV: {csv_filename}")
        print(f"Pergunta: {question}")
        
        result = client.csv_query(
            question=question,
            csv_filename=csv_filename,
            explain_sql=True
        )
        
        print("\nSQL gerado:")
        print(result.get('sql', 'SQL não disponível'))
        
        if 'explanation' in result:
            print("\nExplicação:")
            print(result['explanation'])
        
        print("\nResultados:")
        data = result.get('data', [])
        for row in data:
            print(row)
        
        # 3. Gerar visualização da consulta
        print("\n3. Gerando visualização para a consulta...")
        
        # Determinar colunas apropriadas para visualização
        if data:
            columns = list(data[0].keys())
            viz_options = {
                "title": "Total de Vendas por Categoria",
                "x_column": next((col for col in columns if "categ" in col.lower()), columns[0]),
                "y_column": next((col for col in columns if "total" in col.lower() or "amount" in col.lower() or "sum" in col.lower()), columns[1] if len(columns) > 1 else columns[0])
            }
            
            html_viz = client.generate_visualization(
                query_id=result.get('query_id'),
                viz_type="bar",
                options=viz_options
            )
            
            # Salvar visualização
            os.makedirs("outputs", exist_ok=True)
            viz_path = os.path.join("outputs", "csv_vendas_categoria.html")
            with open(viz_path, "w") as f:
                f.write(html_viz)
            print(f"Visualização salva como {viz_path}")
            
            # Abrir no navegador
            viz_url = f"file://{os.path.abspath(viz_path)}"
            print(f"Abrindo visualização no navegador: {viz_url}")
            webbrowser.open(viz_url)
            
        # 4. Consulta mais específica com produto e quantidade
        print("\n4. Consultando produtos mais vendidos no CSV...")
        question2 = "Quais são os 5 produtos mais vendidos em termos de quantidade?"
        
        result2 = client.csv_query(
            question=question2,
            csv_filename=csv_filename,
            explain_sql=True
        )
        
        print("\nSQL gerado:")
        print(result2.get('sql', 'SQL não disponível'))
        
        print("\nResultados:")
        data2 = result2.get('data', [])
        for row in data2:
            print(row)
        
        # Gerar visualização para produtos mais vendidos
        if data2:
            columns = list(data2[0].keys())
            viz_options = {
                "title": "Produtos Mais Vendidos (Quantidade)",
                "x_column": next((col for col in columns if "product" in col.lower() or "produto" in col.lower()), columns[0]),
                "y_column": next((col for col in columns if "quant" in col.lower()), columns[1] if len(columns) > 1 else columns[0])
            }
            
            html_viz = client.generate_visualization(
                query_id=result2.get('query_id'),
                viz_type="bar",
                options=viz_options
            )
            
            # Salvar visualização
            viz_path = os.path.join("outputs", "csv_produtos_quantidade.html")
            with open(viz_path, "w") as f:
                f.write(html_viz)
            print(f"Visualização salva como {viz_path}")
            
            # Abrir no navegador
            viz_url = f"file://{os.path.abspath(viz_path)}"
            print(f"Abrindo visualização no navegador: {viz_url}")
            webbrowser.open(viz_url)
            
    except Exception as e:
        logger.error(f"Erro ao executar teste CSV: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    # Confirmar que o servidor está rodando
    try:
        resp = requests.get(f"{BASE_URL}/health")
        resp.raise_for_status()
        test_csv_processing()
    except Exception as e:
        print(f"Erro ao verificar servidor: {str(e)}")
        print("Certifique-se de que o servidor está rodando antes de executar este teste.")
        print("Execute: python run.py --server")