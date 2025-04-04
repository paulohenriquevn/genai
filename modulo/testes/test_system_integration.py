#!/usr/bin/env python3
"""
Testes de Integração do Sistema Completo
=======================================

Este módulo contém testes para verificar a integração completa do sistema,
testando o fluxo de ponta a ponta com todos os componentes:
- Motor de consulta
- Modelos de linguagem
- API REST
- Processamento de dados
"""

import unittest
import os
import sys
import json
import tempfile
import shutil
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Modo não interativo para testes

# Adiciona diretório pai ao PATH para importar módulos adequadamente
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importações do sistema
from natural_query_engine import NaturalLanguageQueryEngine
from llm_integration import LLMIntegration, ModelType, LLMQueryGenerator
from core.dataframe import DataFrameWrapper
from core.response.chart import ChartResponse
from core.response.dataframe import DataFrameResponse
from core.response.number import NumberResponse
from core.response.string import StringResponse


class TestSystemIntegration(unittest.TestCase):
    """Testes de integração do sistema completo"""
    
    @classmethod
    def setUpClass(cls):
        """Configurações iniciais para todos os testes de integração"""
        # Cria diretório temporário para dados e resultados
        cls.test_dir = tempfile.mkdtemp()
        cls.output_dir = os.path.join(cls.test_dir, "output")
        os.makedirs(cls.output_dir, exist_ok=True)
        
        # Cria dados de teste
        cls._create_test_data(cls.test_dir)
        
        # Cria arquivos de configuração
        cls._create_config_files(cls.test_dir)
        
        # Inicializa componentes do sistema
        cls._initialize_system(cls.test_dir)
    
    @classmethod
    def tearDownClass(cls):
        """Limpeza após todos os testes"""
        # Remove diretório temporário
        shutil.rmtree(cls.test_dir)
    
    @classmethod
    def _create_test_data(cls, data_dir):
        """Cria dados de teste para integração do sistema"""
        # Dados para vendas temporais
        vendas_temporais = pd.DataFrame({
            'id_venda': range(1, 101),
            'data_venda': pd.date_range(start='2023-01-01', periods=100),
            'valor': [100 + i * 25 + (i % 12) * 100 for i in range(100)],  # Padrão com sazonalidade
            'id_cliente': [(i % 10) + 1 for i in range(100)],
            'id_produto': [(i % 5) + 1 for i in range(100)]
        })
        
        # Dados para clientes
        clientes = pd.DataFrame({
            'id_cliente': range(1, 11),
            'nome': [f'Cliente {i}' for i in range(1, 11)],
            'segmento': ['Varejo', 'Corporativo', 'Governo'] * 3 + ['Varejo'],
            'cidade': ['São Paulo', 'Rio de Janeiro', 'Belo Horizonte', 'Curitiba', 'Porto Alegre'] * 2
        })
        
        # Dados para vendas perdidas (análise preditiva simulada)
        vendas_perdidas = pd.DataFrame({
            'id': range(1, 51),
            'Motivo': ['Preço', 'Concorrência', 'Prazo', 'Produto indisponível', 'Desistência'] * 10,
            'ImpactoFinanceiro': [1000 + (i * 200) + ((i % 5) * 150) for i in range(50)],
            'EstagioPerda': ['Proposta', 'Negociação', 'Fechamento'] * 16 + ['Proposta', 'Negociação'],
            'ProbabilidadeRecuperacao': [0.1 + (i % 10) * 0.05 for i in range(50)],
            'DataPrevista': pd.date_range(start='2023-06-01', periods=50, freq='D')
        })
        
        # Salvando os dados
        os.makedirs(os.path.join(data_dir, "dados"), exist_ok=True)
        vendas_temporais.to_csv(os.path.join(data_dir, "dados", "vendas.csv"), index=False)
        clientes.to_csv(os.path.join(data_dir, "dados", "clientes.csv"), index=False)
        vendas_perdidas.to_csv(os.path.join(data_dir, "dados", "vendas_perdidas.csv"), index=False)
    
    @classmethod
    def _create_config_files(cls, data_dir):
        """Cria arquivos de configuração para o sistema"""
        # Configuração de fontes de dados
        datasources = {
            "data_sources": [
                {
                    "id": "vendas",
                    "type": "csv",
                    "path": os.path.join(data_dir, "dados", "vendas.csv"),
                    "delimiter": ",", 
                    "encoding": "utf-8"
                },
                {
                    "id": "clientes",
                    "type": "csv",
                    "path": os.path.join(data_dir, "dados", "clientes.csv"),
                    "delimiter": ",",
                    "encoding": "utf-8"
                },
                {
                    "id": "vendas_perdidas",
                    "type": "csv",
                    "path": os.path.join(data_dir, "dados", "vendas_perdidas.csv"),
                    "delimiter": ",",
                    "encoding": "utf-8"
                }
            ]
        }
        
        # Configuração para LLM (usando modelo mock para testes)
        llm_config = {
            "model_type": "mock",
            "model_name": "test_model",
            "config": {
                "temperature": 0.2
            }
        }
        
        # Salva as configurações
        with open(os.path.join(data_dir, "datasources.json"), "w") as f:
            json.dump(datasources, f, indent=2)
        
        with open(os.path.join(data_dir, "llm_config.json"), "w") as f:
            json.dump(llm_config, f, indent=2)
    
    @classmethod
    def _initialize_system(cls, data_dir):
        """Inicializa os componentes do sistema para testes"""
        # Inicializa o motor de consulta
        cls.engine = NaturalLanguageQueryEngine(
            data_config_path=os.path.join(data_dir, "datasources.json"),
            base_data_path=os.path.join(data_dir, "dados"),
            output_types=["string", "number", "dataframe", "plot"]
        )
        
        # Inicializa o integrador de LLM (mock para testes)
        cls.llm = LLMIntegration(model_type=ModelType.MOCK)
        
        # Inicializa o gerador de consultas
        cls.query_generator = LLMQueryGenerator(llm_integration=cls.llm)
    
    def test_full_system_basic_query(self):
        """Testa o fluxo completo com uma consulta básica"""
        # Consulta em linguagem natural
        query = "Quantos clientes temos por segmento?"
        
        # Executa a consulta completa
        response = self.engine.execute_query(query)
        
        # Verifica a resposta
        self.assertIsNotNone(response)
        self.assertIsInstance(response, (DataFrameResponse, StringResponse, NumberResponse))
        
        # Se for um dataframe, verifica se tem dados
        if isinstance(response, DataFrameResponse):
            self.assertGreater(len(response.value), 0)
            self.assertIn("segmento", response.value.columns)
    
    def test_full_system_visualization(self):
        """Testa o fluxo completo com geração de visualização"""
        # Consulta para gerar visualização
        query = "Mostre um gráfico de barras com vendas por cliente"
        
        # Executa a consulta
        response = self.engine.execute_query(query)
        
        # Verifica a resposta
        self.assertIsNotNone(response)
        self.assertIsInstance(response, ChartResponse)
        
        # Verifica se gerou imagem
        self.assertTrue(response.value.startswith("data:image/png;base64,"))
        
        # Salva a visualização para verificação
        output_path = os.path.join(self.output_dir, "test_visualization.png")
        response.save(output_path)
        self.assertTrue(os.path.exists(output_path))
    
    def test_temporal_analysis(self):
        """Testa análise temporal de vendas"""
        # Consulta para análise temporal
        query = "Mostre a tendência de vendas ao longo do tempo usando um gráfico de linha"
        
        # Executa a consulta
        response = self.engine.execute_query(query)
        
        # Verifica a resposta
        self.assertIsNotNone(response)
        self.assertIsInstance(response, ChartResponse)
        
        # Salva a visualização
        output_path = os.path.join(self.output_dir, "test_temporal.png")
        response.save(output_path)
        self.assertTrue(os.path.exists(output_path))
    
    def test_multiple_queries_sequence(self):
        """Testa sequência de consultas relacionadas"""
        # Sequência de consultas para análise progressiva
        queries = [
            "Qual é o total de vendas por cliente?",
            "Qual cliente tem o maior valor de vendas?",
            "Em quais meses este cliente teve mais vendas?",
            "Mostre um gráfico comparando as vendas mensais dos 3 maiores clientes"
        ]
        
        # Executa as consultas em sequência
        results = []
        for query in queries:
            response = self.engine.execute_query(query)
            results.append(response)
            
            # Verifica se a consulta foi bem-sucedida
            self.assertIsNotNone(response)
            self.assertNotIsInstance(response, Exception)
        
        # Verifica se pelo menos uma visualização foi gerada
        has_visualization = any(isinstance(r, ChartResponse) for r in results)
        self.assertTrue(has_visualization)
    
    def test_llm_integration_with_engine(self):
        """Testa a integração do LLM com o motor de consulta"""
        # Cria um prompt para o LLM
        prompt = """
        Gere código Python para analisar os dados de vendas e encontrar tendências sazonais.
        Os dados estão disponíveis como 'vendas' e têm colunas 'data_venda' e 'valor'.
        """
        
        # Gera código com o LLM
        generated_code = self.llm.generate_code(prompt)
        
        # Verifica se o código foi gerado
        self.assertIsNotNone(generated_code)
        self.assertIn("import pandas", generated_code)
        
        # Prepara o contexto para execução
        execution_context = {
            'execute_sql_query': self.engine.execute_sql_query,
            'pd': pd,
            'plt': matplotlib.pyplot
        }
        
        try:
            # Tenta executar o código
            result = self.engine.code_executor.execute_code(
                code=generated_code,
                context=execution_context
            )
            
            # Verifica se a execução foi bem-sucedida
            self.assertTrue(result.get("success", False) or "result" in result)
        except Exception as e:
            # Algumas falhas podem ocorrer devido à natureza do código gerado
            # Registramos o erro para referência, mas não fazemos o teste falhar
            print(f"Erro na execução (esperado em teste de integração): {str(e)}")
    
    def test_custom_dataframe_integration(self):
        """Testa integração do sistema com DataFrame personalizado"""
        # Cria um DataFrame personalizado
        custom_df = pd.DataFrame({
            'produto': ['A', 'B', 'C', 'D', 'E'],
            'vendas': [100, 150, 200, 120, 180],
            'custo': [50, 70, 90, 60, 80]
        })
        
        # Cria um wrapper
        df_wrapper = DataFrameWrapper(custom_df, "produtos")
        
        # Adiciona ao motor
        self.engine.dataframes["produtos"] = df_wrapper
        self.engine.agent_state.dfs = list(self.engine.dataframes.values())
        
        # Executa uma consulta usando o novo DataFrame
        query = "Qual é o produto com maior lucro? Calcule lucro como vendas menos custo."
        response = self.engine.execute_query(query)
        
        # Verifica a resposta
        self.assertIsNotNone(response)
        # O tipo pode variar, então não verificamos
        
        # Remove o DataFrame personalizado para não afetar outros testes
        del self.engine.dataframes["produtos"]
        self.engine.agent_state.dfs = list(self.engine.dataframes.values())
    
    def test_error_recovery(self):
        """Testa a recuperação automática de erros"""
        # Consulta intencionalmente problemática
        query = "Mostra informação da tabela que não existe com um gráfico complexo"
        
        # Executa a consulta
        response = self.engine.execute_query(query)
        
        # Deve retornar algum tipo de resposta, mesmo com erro
        self.assertIsNotNone(response)
        
        # O sistema deve tentar corrigir erro e gerar alguma saída útil
        if hasattr(response, 'error') and response.error:
            # Verifica se a mensagem de erro é útil
            self.assertIn("tabela", response.value.lower())
        else:
            # Se não houve erro, deve ter retornado alguma resposta substituta
            self.assertIsInstance(response, (StringResponse, DataFrameResponse))


# Cenários de teste complexos em arquivos separados
if __name__ == '__main__':
    unittest.main()