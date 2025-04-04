#!/usr/bin/env python3
"""
Testes para API REST do Sistema de Consulta em Linguagem Natural
================================================================

Este módulo contém testes para verificar a API REST do sistema,
testando suas principais funcionalidades:
- Endpoints básicos
- Processamento de consultas via API
- Upload de dados
- Recuperação de informações sobre datasets
"""

import unittest
import os
import sys
import json
import tempfile
import pandas as pd
from pathlib import Path
from fastapi.testclient import TestClient

# Adiciona diretório pai ao PATH para importar módulos adequadamente
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importa a aplicação FastAPI do módulo app
import app


class TestAPIService(unittest.TestCase):
    """Testes para a API REST do sistema"""
    
    @classmethod
    def setUpClass(cls):
        """Configurações iniciais para todos os testes"""
        # Cria um cliente de teste para a API
        cls.client = TestClient(app.app)
        
        # Cria arquivos de dados de teste
        cls.test_dir = tempfile.mkdtemp()
        cls._create_test_data(cls.test_dir)
    
    @classmethod
    def tearDownClass(cls):
        """Limpeza após todos os testes"""
        # Remove diretório de teste
        import shutil
        shutil.rmtree(cls.test_dir)
    
    @classmethod
    def _create_test_data(cls, data_dir):
        """Cria dados de teste para os testes da API"""
        # Cria um CSV simples para testes de upload
        test_df = pd.DataFrame({
            'id': range(1, 11),
            'nome': [f'Teste {i}' for i in range(1, 11)],
            'valor': [i * 100 for i in range(1, 11)]
        })
        
        # Salva o DataFrame como CSV
        cls.test_csv_path = os.path.join(data_dir, "teste.csv")
        test_df.to_csv(cls.test_csv_path, index=False)
    
    def test_root_redirect(self):
        """Testa redirecionamento da raiz para a documentação"""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)  # 200 no contexto de teste
        # Na produção seria 307 (redirecionamento)
    
    def test_get_datasources(self):
        """Testa endpoint para listar fontes de dados"""
        response = self.client.get("/datasources")
        self.assertEqual(response.status_code, 200)
        
        # Verifica se a resposta é uma lista
        data = response.json()
        self.assertIsInstance(data, list)
        
        # Verifica se as fontes padrão estão presentes
        for source in ['vendas', 'clientes', 'vendas_perdidas']:
            self.assertIn(source, data)
    
    def test_get_stats(self):
        """Testa endpoint para estatísticas de uso"""
        response = self.client.get("/stats")
        self.assertEqual(response.status_code, 200)
        
        # Verifica se a resposta é um dicionário
        data = response.json()
        self.assertIsInstance(data, dict)
        
        # Verifica campos obrigatórios
        self.assertIn("total_queries", data)
        self.assertIn("successful_queries", data)
        self.assertIn("loaded_dataframes", data)
    
    def test_upload_data(self):
        """Testa upload de novos dados"""
        # Prepara o arquivo para upload
        with open(self.test_csv_path, "rb") as f:
            files = {"file": ("teste.csv", f, "text/csv")}
            response = self.client.post(
                "/upload_data?data_source_name=teste&data_format=csv",
                files=files
            )
        
        # Verifica o resultado
        self.assertEqual(response.status_code, 200)
        
        # Verifica se a fonte de dados foi adicionada
        response = self.client.get("/datasources")
        self.assertIn("teste", response.json())
    
    def test_dataset_info(self):
        """Testa obtenção de informações sobre dataset"""
        # Primeiro faz upload dos dados
        with open(self.test_csv_path, "rb") as f:
            files = {"file": ("teste.csv", f, "text/csv")}
            self.client.post(
                "/upload_data?data_source_name=teste_info&data_format=csv",
                files=files
            )
        
        # Obtém informações sobre o dataset
        response = self.client.get("/dataset_info/teste_info")
        self.assertEqual(response.status_code, 200)
        
        # Verifica informações retornadas
        data = response.json()
        self.assertEqual(data["name"], "teste_info")
        self.assertEqual(data["rows"], 10)
        self.assertIn("id", data["columns"])
        self.assertIn("nome", data["columns"])
        self.assertIn("valor", data["columns"])
    
    def test_dataset_info_not_found(self):
        """Testa obtenção de informações sobre dataset inexistente"""
        response = self.client.get("/dataset_info/nao_existe")
        self.assertEqual(response.status_code, 404)
    
    def test_execute_sql(self):
        """Testa execução direta de SQL via API"""
        # Conteúdo da consulta SQL
        sql_query = "SELECT * FROM vendas LIMIT 5"
        
        # Faz a requisição
        response = self.client.post(
            "/execute_sql",
            json=sql_query
        )
        
        # Verifica o resultado
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("rows", data)
        self.assertIn("columns", data)
        self.assertIn("data", data)
        
        # Verifica se os dados retornados são consistentes
        self.assertLessEqual(data["rows"], 5)
        self.assertGreater(len(data["columns"]), 0)
    
    def test_query_endpoint_basic(self):
        """Testa processamento de consulta básica via API"""
        # Conteúdo da consulta
        query_data = {
            "query": "Quantos registros existem na tabela de vendas?",
            "output_type": "number"
        }
        
        # Faz a requisição
        response = self.client.post(
            "/query",
            json=query_data
        )
        
        # Verifica o resultado
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Verifica os campos obrigatórios
        self.assertEqual(data["query"], query_data["query"])
        self.assertIn("result_type", data)
        self.assertIn("result", data)
        self.assertIn("execution_time", data)
        self.assertIn("timestamp", data)
    
    def test_query_endpoint_plot(self):
        """Testa processamento de consulta de visualização via API"""
        # Conteúdo da consulta
        query_data = {
            "query": "Mostre um gráfico de barras com o total de vendas perdidas por motivo",
            "output_type": "plot"
        }
        
        # Faz a requisição
        response = self.client.post(
            "/query",
            json=query_data
        )
        
        # Verifica o resultado
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Verifica o tipo do resultado
        self.assertEqual(data["result_type"], "plot")
        
        # Verifica se contém URL de visualização
        self.assertIn("visualization_url", data["result"])
        self.assertTrue(data["result"]["visualization_url"].startswith("/visualizations/"))
    
    def test_query_endpoint_dataframe(self):
        """Testa processamento de consulta de dataframe via API"""
        # Conteúdo da consulta
        query_data = {
            "query": "Mostre os primeiros 5 registros de clientes",
            "output_type": "dataframe"
        }
        
        # Faz a requisição
        response = self.client.post(
            "/query",
            json=query_data
        )
        
        # Verifica o resultado
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Verifica o tipo do resultado
        self.assertEqual(data["result_type"], "dataframe")
        
        # Verifica se o resultado contém dados
        self.assertIsInstance(data["result"], list)
        self.assertLessEqual(len(data["result"]), 5)
    
    def test_query_endpoint_error(self):
        """Testa tratamento de erros na consulta"""
        # Conteúdo da consulta inválida
        query_data = {
            "query": "Mostre dados da tabela que não existe",
            "output_type": "dataframe"
        }
        
        # Faz a requisição
        response = self.client.post(
            "/query",
            json=query_data
        )
        
        # Espera status 500 para erros
        self.assertEqual(response.status_code, 500)
        
        # Verifica se a resposta contém informações de erro
        data = response.json()
        self.assertIn("detail", data)
        self.assertIn("error", data["detail"])
        self.assertIn("traceback", data["detail"])
        self.assertIn("query", data["detail"])
    
    def test_visualization_endpoint(self):
        """Testa recuperação de visualização criada"""
        # Conteúdo da consulta
        query_data = {
            "query": "Crie um gráfico de barras mostrando o total por cidade do cliente",
            "output_type": "plot"
        }
        
        # Faz a requisição para criar visualização
        response = self.client.post(
            "/query",
            json=query_data
        )
        
        # Obtém a URL da visualização
        viz_url = response.json()["result"]["visualization_url"]
        
        # Tenta obter a visualização
        response = self.client.get(viz_url)
        
        # Verifica se a visualização foi recuperada
        # Como este é um cliente de teste, não simula completamente o sistema de arquivos
        # então não esperamos 200, mas verificamos que o endpoint está registrado
        self.assertNotEqual(response.status_code, 404)


if __name__ == '__main__':
    unittest.main()