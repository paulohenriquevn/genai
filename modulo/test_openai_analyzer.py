#!/usr/bin/env python3
"""
Teste do Analisador OpenAI
=========================

Este script testa o analisador de dados com OpenAI usando um modo de simulação.
"""

import os
import sys
import logging
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

# Configura o logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Adiciona o diretório pai ao PATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class TestOpenAIAnalyzer(unittest.TestCase):
    """Testes para o analisador OpenAI"""
    
    @classmethod
    def setUpClass(cls):
        """Configuração inicial"""
        # Diretório de dados para testes
        cls.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dados")
        
        # Diretório de saída para testes
        cls.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "test_analyzer")
        os.makedirs(cls.output_dir, exist_ok=True)
        
        # Define a chave de API de teste
        os.environ["OPENAI_API_KEY"] = "sk-fake-key-for-testing"
    
    @classmethod
    def tearDownClass(cls):
        """Limpeza após os testes"""
        # Remove a chave de API de teste
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]
    
    def setUp(self):
        # Aplica os patches antes de cada teste
        self.mock_openai = patch('openai.chat.completions.create').start()
        self.mock_natural_query = patch('natural_query_engine.NaturalLanguageQueryEngine.execute_query').start()
        
        # Configura o mock do OpenAI para retornar respostas simuladas
        self.mock_openai.return_value = MagicMock()
        self.mock_openai.return_value.choices = [MagicMock()]
        self.mock_openai.return_value.choices[0].message = MagicMock()
        self.mock_openai.return_value.choices[0].message.content = "Resposta simulada da API OpenAI"
        
        # Agora podemos importar OpenAIAnalyzer depois de configurar os mocks
        from openai_analyzer import OpenAIAnalyzer
        from core.response.dataframe import DataFrameResponse
        
        # Cria uma instância do analisador
        self.analyzer = OpenAIAnalyzer(
            api_key="sk-fake-key-for-testing",
            data_dir=self.data_dir,
            output_dir=self.output_dir,
            model="gpt-3.5-turbo"  # Usa um modelo mais simples para testes
        )
        
        # Configuração adicional do mock de execução de consulta
        df = pd.DataFrame({
            'id_cliente': [1, 2, 3],
            'valor': [100, 200, 300]
        })
        self.mock_natural_query.return_value = DataFrameResponse(df)
    
    def tearDown(self):
        # Para todos os patches após cada teste
        patch.stopall()
    
    def test_initialization(self):
        """Testa a inicialização do analisador"""
        self.assertIsNotNone(self.analyzer)
        self.assertIsNotNone(self.analyzer.query_engine)
    
    def test_generate_analysis_questions(self):
        """Testa a geração de perguntas analíticas"""
        # Configura o mock para retornar perguntas específicas
        self.mock_openai.return_value.choices[0].message.content = (
            "Qual é o total de vendas por cliente?\n"
            "Mostre um gráfico de barras das vendas perdidas por motivo.\n"
            "Qual é o valor médio das vendas?"
        )
        
        questions = self.analyzer.generate_analysis_questions("vendas", 3)
        
        self.assertEqual(len(questions), 3)
        self.assertIsInstance(questions, list)
        self.assertIsInstance(questions[0], str)
        self.assertEqual(questions[0], "Qual é o total de vendas por cliente?")
    
    def test_run_query(self):
        """Testa a execução de uma consulta"""
        # Configura o mock para análise
        self.mock_openai.return_value.choices[0].message.content = "Análise detalhada dos resultados..."
        
        # Executa a consulta
        result, analysis = self.analyzer.run_query("Qual é o total de vendas por cliente?")
        
        # Verifica se o método execute_query foi chamado
        self.mock_natural_query.assert_called_once()
        
        # Verifica se a análise foi gerada
        self.assertEqual(analysis, "Análise detalhada dos resultados...")
    
    @patch('openai_analyzer.OpenAIAnalyzer.save_visualization')
    @patch('openai_analyzer.OpenAIAnalyzer.generate_html_report')
    def test_run_analysis(self, mock_report, mock_save_viz):
        """Testa a execução de uma análise completa"""
        # Configura os mocks
        self.mock_openai.return_value.choices[0].message.content = (
            "Qual é o total de vendas?\n"
            "Mostre um gráfico das vendas por cliente."
        )
        mock_report.return_value = os.path.join(self.output_dir, "report.html")
        mock_save_viz.return_value = os.path.join(self.output_dir, "viz.png")
        
        # Executa a análise
        results = self.analyzer.run_analysis("vendas", num_questions=2)
        
        # Verifica os resultados básicos
        self.assertIsInstance(results, dict)
        self.assertIn("id", results)
        self.assertIn("topic", results)
        self.assertEqual(results["topic"], "vendas")
        self.assertIn("queries", results)
        self.assertEqual(len(results["queries"]), 2)


if __name__ == "__main__":
    unittest.main()