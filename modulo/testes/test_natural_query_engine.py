#!/usr/bin/env python3
"""
Testes para o Motor de Consulta em Linguagem Natural
===================================================

Este módulo contém testes para verificar o funcionamento do motor de consulta 
em linguagem natural, testando suas principais funcionalidades:
- Carregamento de dados
- Consultas básicas
- Agregações
- Visualizações
- Tratamento de erros
"""

import unittest
import os
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path

# Adiciona diretório pai ao PATH para importar módulos adequadamente
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from natural_query_engine import NaturalLanguageQueryEngine
from core.response.dataframe import DataFrameResponse
from core.response.number import NumberResponse
from core.response.string import StringResponse
from core.response.chart import ChartResponse
from core.response.error import ErrorResponse


class TestNaturalQueryEngine(unittest.TestCase):
    """Testes para o motor de consulta em linguagem natural"""
    
    @classmethod
    def setUpClass(cls):
        """Configurações iniciais para todos os testes"""
        # Cria um diretório temporário para dados de teste
        cls.test_data_dir = tempfile.mkdtemp()
        
        # Cria dados de teste
        cls._create_test_data(cls.test_data_dir)
        
        # Cria arquivos de configuração para testes
        cls._create_config_files(cls.test_data_dir)
        
        # Inicializa o motor com os dados de teste
        cls.engine = NaturalLanguageQueryEngine(
            data_config_path=os.path.join(cls.test_data_dir, "datasources.json"),
            metadata_config_path=os.path.join(cls.test_data_dir, "metadata.json"),
            base_data_path=cls.test_data_dir
        )
    
    @classmethod
    def tearDownClass(cls):
        """Limpeza após todos os testes"""
        # Remove o diretório temporário e todos os seus arquivos
        shutil.rmtree(cls.test_data_dir)
    
    @classmethod
    def _create_test_data(cls, data_dir):
        """Cria dados de teste para os testes"""
        # DataFrame de vendas
        vendas_df = pd.DataFrame({
            'id_venda': range(1, 101),
            'data_venda': pd.date_range(start='2023-01-01', periods=100),
            'valor': np.random.uniform(100, 1000, 100).round(2),
            'id_cliente': np.random.randint(1, 11, 100),
            'id_produto': np.random.randint(1, 6, 100)
        })
        
        # DataFrame de clientes
        clientes_df = pd.DataFrame({
            'id_cliente': range(1, 11),
            'nome': [f'Cliente {i}' for i in range(1, 11)],
            'cidade': np.random.choice(['São Paulo', 'Rio de Janeiro', 'Belo Horizonte', 'Curitiba', 'Brasília'], 10),
            'segmento': np.random.choice(['Varejo', 'Corporativo', 'Governo'], 10)
        })
        
        # DataFrame de vendas perdidas
        vendas_perdidas_df = pd.DataFrame({
            'id': range(1, 51),
            'Motivo': np.random.choice(['Preço', 'Concorrência', 'Prazo', 'Produto indisponível', 'Desistência'], 50),
            'ImpactoFinanceiro': np.random.uniform(500, 5000, 50).round(2),
            'EstagioPerda': np.random.choice(['Proposta', 'Negociação', 'Fechamento'], 50),
            'id_cliente': np.random.randint(1, 11, 50)
        })
        
        # Salva os DataFrames como arquivos CSV
        vendas_df.to_csv(os.path.join(data_dir, "vendas.csv"), index=False)
        clientes_df.to_csv(os.path.join(data_dir, "clientes.csv"), index=False)
        vendas_perdidas_df.to_csv(os.path.join(data_dir, "vendas_perdidas.csv"), index=False)
    
    @classmethod
    def _create_config_files(cls, data_dir):
        """Cria arquivos de configuração para os testes"""
        # Configuração de datasources
        datasources_config = {
            "data_sources": [
                {
                    "id": "vendas",
                    "type": "csv",
                    "path": os.path.join(data_dir, "vendas.csv"),
                    "delimiter": ",",
                    "encoding": "utf-8"
                },
                {
                    "id": "clientes",
                    "type": "csv",
                    "path": os.path.join(data_dir, "clientes.csv"),
                    "delimiter": ",",
                    "encoding": "utf-8"
                },
                {
                    "id": "vendas_perdidas",
                    "type": "csv",
                    "path": os.path.join(data_dir, "vendas_perdidas.csv"),
                    "delimiter": ",",
                    "encoding": "utf-8"
                }
            ]
        }
        
        # Configuração de metadados básicos
        metadata_config = {
            "datasets": [
                {
                    "name": "vendas",
                    "description": "Dados de vendas realizadas",
                    "source": "sistema_erp",
                    "columns": [
                        {"name": "id_venda", "description": "ID único da venda", "type": "int"},
                        {"name": "data_venda", "description": "Data da venda", "type": "date"},
                        {"name": "valor", "description": "Valor da venda em reais", "type": "float"},
                        {"name": "id_cliente", "description": "ID do cliente", "type": "int"},
                        {"name": "id_produto", "description": "ID do produto", "type": "int"}
                    ]
                },
                {
                    "name": "clientes",
                    "description": "Dados dos clientes",
                    "source": "sistema_crm",
                    "columns": [
                        {"name": "id_cliente", "description": "ID único do cliente", "type": "int"},
                        {"name": "nome", "description": "Nome do cliente", "type": "string"},
                        {"name": "cidade", "description": "Cidade do cliente", "type": "string"},
                        {"name": "segmento", "description": "Segmento do cliente", "type": "string"}
                    ]
                },
                {
                    "name": "vendas_perdidas",
                    "description": "Dados de oportunidades perdidas",
                    "source": "sistema_crm",
                    "columns": [
                        {"name": "id", "description": "ID único da oportunidade perdida", "type": "int"},
                        {"name": "Motivo", "description": "Motivo da perda", "type": "string"},
                        {"name": "ImpactoFinanceiro", "description": "Valor perdido em reais", "type": "float"},
                        {"name": "EstagioPerda", "description": "Estágio em que a venda foi perdida", "type": "string"},
                        {"name": "id_cliente", "description": "ID do cliente", "type": "int"}
                    ]
                }
            ]
        }
        
        # Salva os arquivos de configuração
        import json
        with open(os.path.join(data_dir, "datasources.json"), "w") as f:
            json.dump(datasources_config, f, indent=2)
        
        with open(os.path.join(data_dir, "metadata.json"), "w") as f:
            json.dump(metadata_config, f, indent=2)
    
    def test_engine_initialization(self):
        """Testa se o motor é inicializado corretamente"""
        self.assertIsNotNone(self.engine)
        self.assertEqual(len(self.engine.dataframes), 3)
        self.assertIn("vendas", self.engine.dataframes)
        self.assertIn("clientes", self.engine.dataframes)
        self.assertIn("vendas_perdidas", self.engine.dataframes)
    
    def test_basic_sql_query(self):
        """Testa a execução direta de consultas SQL"""
        # Consulta básica de vendas
        result = self.engine.execute_sql_query("SELECT * FROM vendas LIMIT 5")
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 5)
        
        # Consulta com JOIN
        result = self.engine.execute_sql_query("""
            SELECT v.id_venda, v.valor, c.nome 
            FROM vendas v 
            JOIN clientes c ON v.id_cliente = c.id_cliente 
            LIMIT 10
        """)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("nome", result.columns)
    
    def test_basic_query(self):
        """Testa consultas básicas em linguagem natural"""
        # Consulta simples
        response = self.engine.execute_query("Mostre as primeiras 5 linhas de vendas")
        self.assertIsInstance(response, DataFrameResponse)
        self.assertEqual(len(response.value), 5)
        
        # Consulta de contagem
        response = self.engine.execute_query("Quantos registros existem na tabela de vendas?")
        self.assertIsInstance(response, (NumberResponse, StringResponse))
    
    def test_aggregation_query(self):
        """Testa consultas com agregações"""
        # Soma total
        response = self.engine.execute_query("Qual é o valor total de vendas?")
        self.assertIsInstance(response, (NumberResponse, StringResponse))
        
        # Média
        response = self.engine.execute_query("Qual é o valor médio das vendas?")
        self.assertIsInstance(response, (NumberResponse, StringResponse))
    
    def test_group_by_query(self):
        """Testa consultas com agrupamento"""
        # Agrupamento básico
        response = self.engine.execute_query("Agrupe as vendas por cliente e mostre o total")
        self.assertIsInstance(response, (DataFrameResponse, ChartResponse))
        
        # Agrupamento com filtro
        response = self.engine.execute_query("Mostre o total de vendas por cidade do cliente")
        self.assertIsInstance(response, (DataFrameResponse, ChartResponse))
        
    def test_visualization_query(self):
        """Testa consultas com visualização"""
        # Gráfico básico
        response = self.engine.execute_query("Mostre um gráfico de barras com o total de vendas por cliente")
        self.assertIsInstance(response, ChartResponse)
        self.assertTrue(response.value.startswith("data:image/png;base64,"))
        
        # Histograma
        response = self.engine.execute_query("Crie um histograma dos valores de venda")
        self.assertIsInstance(response, ChartResponse)
        self.assertTrue(response.value.startswith("data:image/png;base64,"))
    
    def test_error_handling(self):
        """Testa o tratamento de erros"""
        # Consulta com tabela inexistente
        response = self.engine.execute_query("Mostre dados da tabela produtos inexistente")
        self.assertIsInstance(response, ErrorResponse)
        self.assertIn("erro", response.value.lower())
        
        # Consulta vazia
        response = self.engine.execute_query("")
        self.assertIsInstance(response, ErrorResponse)
    
    def test_complex_query(self):
        """Testa consultas mais complexas"""
        # Consulta combinando dados de várias tabelas
        response = self.engine.execute_query(
            "Quais são os clientes com maior valor total de vendas e em quais cidades eles estão?")
        self.assertIsInstance(response, (DataFrameResponse, StringResponse))
        
        # Consulta analítica
        response = self.engine.execute_query(
            "Compare o impacto financeiro das vendas perdidas por motivo usando um gráfico de barras")
        self.assertIsInstance(response, ChartResponse)


if __name__ == '__main__':
    unittest.main()