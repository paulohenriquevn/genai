#!/usr/bin/env python3
"""
Teste Completo de Integração com OpenAI
========================================

Este script demonstra a funcionalidade do sistema de consulta 
em linguagem natural usando o modelo OpenAI para análise de dados.
"""

import os
import sys
import unittest
import pandas as pd
import matplotlib.pyplot as plt
import json

# Adiciona o diretório pai ao PATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importações do sistema
from natural_query_engine import NaturalLanguageQueryEngine
from llm_integration import LLMIntegration, ModelType
from core.response import ResponseParser
from core.response.chart import ChartResponse
from core.response.dataframe import DataFrameResponse
from core.response.number import NumberResponse
from core.response.string import StringResponse
from core.dataframe import DataFrameWrapper

class OpenAIIntegrationTest(unittest.TestCase):
    """
    Teste de integração abrangente usando OpenAI para análise de dados.
    
    Requer uma API key válida da OpenAI configurada como variável de ambiente.
    """
    
    @classmethod
    def setUpClass(cls):
        """
        Configuração inicial para o teste de integração.
        Prepara o ambiente, carrega dados e inicializa componentes.
        """
        # Verifica se a API key está disponível
        if not os.environ.get("OPENAI_API_KEY"):
            raise unittest.SkipTest("Chave da API OpenAI não configurada")
        
        # Cria diretório temporário para dados de teste
        cls.test_data_dir = os.path.join(os.getcwd(), "testes", "output", "openai_test")
        os.makedirs(cls.test_data_dir, exist_ok=True)
        
        # Cria dataset de exemplo
        dataset_path = cls._create_sample_dataset()
        
        # Prepara configuração de datasources
        datasources_config = {
            "data_sources": [
                {
                    "id": "vendas_tech",
                    "type": "csv",
                    "path": dataset_path,
                    "delimiter": ",",
                    "encoding": "utf-8"
                }
            ]
        }
        
        # Salva arquivo de configuração de datasources
        datasources_path = os.path.join(cls.test_data_dir, "datasources.json")
        with open(datasources_path, 'w') as f:
            json.dump(datasources_config, f, indent=2)
        
        # Inicializa o motor de consulta com o arquivo de configuração
        cls.engine = NaturalLanguageQueryEngine(
            data_config_path=datasources_path,
            base_data_path=cls.test_data_dir,
            output_types=["string", "number", "dataframe", "plot"]
        )
        
        # Verifica se os dataframes foram carregados
        print("Dataframes carregados:", list(cls.engine.dataframes.keys()))
        
        # Inicializa integração com OpenAI
        cls.llm = LLMIntegration(
            model_type=ModelType.OPENAI,
            model_name="gpt-3.5-turbo",
            api_key=os.environ.get("OPENAI_API_KEY")
        )
        
        # Inicializa parser de respostas
        cls.response_parser = ResponseParser()
    
    @classmethod
    def _create_sample_dataset(cls):
        """
        Cria um dataset de exemplo para análise.
        Simula dados de vendas de uma empresa de tecnologia.
        """
        # Cria dados de vendas com informações detalhadas
        import numpy as np
        import pandas as pd
        
        # Define seed para reprodutibilidade
        np.random.seed(42)
        
        # Gera dados de vendas
        num_records = 500
        produtos = ['Smartphone', 'Laptop', 'Tablet', 'Smartwatch', 'Acessórios']
        canais = ['E-commerce', 'Loja Física', 'Parceiros', 'Televendas']
        regioes = ['Sudeste', 'Sul', 'Nordeste', 'Centro-Oeste', 'Norte']
        
        df = pd.DataFrame({
            'data_venda': pd.date_range(start='2023-01-01', periods=num_records),
            'produto': np.random.choice(produtos, num_records),
            'canal_venda': np.random.choice(canais, num_records),
            'regiao': np.random.choice(regioes, num_records),
            'valor_venda': np.random.normal(1500, 500, num_records).round(2),
            'quantidade': np.random.randint(1, 5, num_records)
        })
        
        # Adiciona alguns padrões intencionais
        df.loc[df['produto'] == 'Smartphone', 'valor_venda'] *= 1.5
        df.loc[df['canal_venda'] == 'E-commerce', 'valor_venda'] *= 1.2
        
        # Salva o dataset
        output_path = os.path.join(cls.test_data_dir, "vendas_tecnologia.csv")
        df.to_csv(output_path, index=False)
        
        return output_path
    
    def test_comprehensive_openai_data_analysis(self):
        """
        Teste abrangente de análise de dados usando OpenAI:
        1. Consultas básicas
        2. Geração de insights
        3. Visualizações
        4. Análise preditiva
        """
        # Adiciona log para verificar dataframes
        print("Dataframes disponíveis:", list(self.engine.dataframes.keys()))
        for nome, df_wrapper in self.engine.dataframes.items():
            print(f"Dataframe {nome}:")
            print(df_wrapper.dataframe.head())
        
        # Lista de consultas para teste progressivo
        consultas = [
            {
                "query": "Qual é o valor total de vendas do dataset de tecnologia?",
                "tipo_esperado": NumberResponse,
                "descricao": "Totalização de vendas"
            },
            {
                "query": "Mostre o total de vendas por produto em um gráfico de barras",
                "tipo_esperado": ChartResponse,
                "descricao": "Vendas por produto"
            },
            {
                "query": "Qual o canal de vendas mais eficiente em termos de valor total?",
                "tipo_esperado": (DataFrameResponse, StringResponse),
                "descricao": "Análise de canais de venda"
            },
            {
                "query": "Mostre a distribuição de vendas por região usando um gráfico de pizza",
                "tipo_esperado": ChartResponse,
                "descricao": "Distribuição regional de vendas"
            },
            {
                "query": "Crie um gráfico de linhas mostrando a tendência de vendas ao longo do tempo",
                "tipo_esperado": ChartResponse,
                "descricao": "Tendência temporal de vendas"
            }
        ]
        
        # Diretório para salvar visualizações
        visualizacoes_dir = os.path.join(self.test_data_dir, "visualizacoes")
        os.makedirs(visualizacoes_dir, exist_ok=True)
        
        # Resultados para consolidação
        resultados_analise = []
        
        # Executa cada consulta
        for i, consulta in enumerate(consultas, 1):
            print(f"\n--- Executando Consulta {i}: {consulta['descricao']} ---")
            
            # Executa a consulta
            try:
                response = self.engine.execute_query(consulta['query'])
                
                # Log detalhado para diagnóstico
                print(f"Resposta recebida: {response}")
                print(f"Tipo da resposta: {type(response)}")
                
                # Verifica o tipo de resposta
                if isinstance(consulta['tipo_esperado'], tuple):
                    self.assertTrue(
                        any(isinstance(response, tipo) for tipo in consulta['tipo_esperado']),
                        f"Tipo de resposta incorreto para: {consulta['query']}"
                    )
                else:
                    self.assertIsInstance(
                        response, consulta['tipo_esperado'], 
                        f"Tipo de resposta incorreto para: {consulta['query']}"
                    )
                
                # Salva visualizações, se aplicável
                if isinstance(response, ChartResponse):
                    caminho_visualizacao = os.path.join(
                        visualizacoes_dir, 
                        f"visualizacao_{i}_{consulta['descricao'].replace(' ', '_')}.png"
                    )
                    response.save(caminho_visualizacao)
                    print(f"Visualização salva em: {caminho_visualizacao}")
                
                # Registra o resultado
                resultados_analise.append({
                    "consulta": consulta['query'],
                    "descricao": consulta['descricao'],
                    "tipo_resposta": type(response).__name__,
                    "sucesso": True
                })
                
                # Imprime um preview do resultado
                if isinstance(response, NumberResponse):
                    print(f"Resultado numérico: {response.value}")
                elif isinstance(response, DataFrameResponse):
                    print("Prévia do DataFrame:")
                    print(response.value.head())
                elif isinstance(response, ChartResponse):
                    print("Visualização gerada com sucesso")
            
            except Exception as e:
                print(f"Erro na consulta: {e}")
                resultados_analise.append({
                    "consulta": consulta['query'],
                    "descricao": consulta['descricao'],
                    "erro": str(e),
                    "sucesso": False
                })
                # Falha no teste se uma consulta crucial falhar
                self.fail(f"Falha na consulta: {consulta['query']} - {e}")
        
        # Análise preditiva avançada usando o modelo OpenAI
        print("\n--- Análise Preditiva Avançada ---")
        prompt_predicao = """
        Gere código Python para prever vendas futuras usando:
        1. Dados históricos de vendas
        2. Análise de tendência temporal
        3. Modelo de regressão simples
        4. Visualização da previsão
        """
        
        try:
            # Gera código de predição usando OpenAI
            codigo_predicao = self.llm.generate_code(prompt_predicao)
            print("Código de predição gerado:")
            print(codigo_predicao)
            
            # Executa o código gerado
            contexto_execucao = {
                'execute_sql_query': self.engine.execute_sql_query,
                'pd': pd,
                'plt': plt,
                'np': __import__('numpy'),
                'datetime': __import__('datetime')
            }
            
            resultado_exec = self.engine.code_executor.execute_code(
                code=codigo_predicao,
                context=contexto_execucao
            )
            
            # Verifica se a execução foi bem-sucedida
            self.assertTrue(resultado_exec.get('success', False), 
                            "Falha na execução do código de predição")
            
            # Registra o resultado da predição
            resultados_analise.append({
                "consulta": "Análise Preditiva",
                "descricao": "Previsão de vendas futuras",
                "sucesso": True
            })
        
        except Exception as e:
            print(f"Erro na análise preditiva: {e}")
            resultados_analise.append({
                "consulta": "Análise Preditiva",
                "descricao": "Previsão de vendas futuras",
                "erro": str(e),
                "sucesso": False
            })
            self.fail(f"Falha na análise preditiva: {e}")
        
        # Salva resultados da análise
        caminho_resultados = os.path.join(self.test_data_dir, "resultados_analise.json")
        with open(caminho_resultados, 'w') as f:
            json.dump(resultados_analise, f, indent=2)
        
        print("\n--- Análise Concluída ---")
        print(f"Resultados salvos em: {caminho_resultados}")
        print(f"Visualizações salvas em: {visualizacoes_dir}")

if __name__ == "__main__":
    unittest.main()