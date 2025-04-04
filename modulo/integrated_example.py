#!/usr/bin/env python3
"""
Exemplo Integrado do Sistema de Consulta em Linguagem Natural
===========================================================

Este arquivo demonstra o uso completo do Sistema de Consulta em Linguagem Natural,
incluindo:

1. Inicialização do sistema
2. Carregamento de dados
3. Execução de consultas em linguagem natural
4. Geração de visualizações
5. Uso da API REST

Exemplos de uso:
- python integrated_example.py                   # Executa todos os exemplos
- python integrated_example.py --api             # Inicia o servidor API
- python integrated_example.py --query "consulta" # Executa uma consulta específica
"""

import os
import sys
import argparse
import pandas as pd
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt

# Importa o sistema integrado
from natural_query_engine import NaturalLanguageQueryEngine
from core.dataframe import DataFrameWrapper
from llm_integration import LLMIntegration, ModelType, LLMQueryGenerator
from api import app as api_app


def load_example_data():
    """Carrega dados de exemplo para demonstração"""
    print("Carregando dados de exemplo...")
    
    # Verifica se as pastas de dados existem
    if not os.path.exists("dados"):
        os.makedirs("dados", exist_ok=True)
    
    # Cria dados de exemplo se não existirem
    if not all(os.path.exists(f"dados/{file}") for file in ["vendas.csv", "clientes.csv", "vendas_perdidas.csv"]):
        print("Criando arquivos de dados de exemplo...")
        
        # Vendas
        vendas_df = pd.DataFrame({
            'id_venda': range(1, 101),
            'data_venda': pd.date_range(start='2023-01-01', periods=100),
            'valor': [100 + i * 25 + (i % 12) * 100 for i in range(100)],  # Padrão com sazonalidade
            'id_cliente': [(i % 10) + 1 for i in range(100)],
            'id_produto': [(i % 5) + 1 for i in range(100)]
        })
        
        # Clientes
        clientes_df = pd.DataFrame({
            'id_cliente': range(1, 11),
            'nome': [f'Cliente {i}' for i in range(1, 11)],
            'segmento': ['Varejo', 'Corporativo', 'Governo'] * 3 + ['Varejo'],
            'cidade': ['São Paulo', 'Rio de Janeiro', 'Belo Horizonte', 'Curitiba', 'Porto Alegre'] * 2
        })
        
        # Vendas perdidas
        vendas_perdidas_df = pd.DataFrame({
            'id': range(1, 51),
            'Motivo': ['Preço', 'Concorrência', 'Prazo', 'Produto indisponível', 'Desistência'] * 10,
            'ImpactoFinanceiro': [1000 + (i * 200) + ((i % 5) * 150) for i in range(50)],
            'EstagioPerda': ['Proposta', 'Negociação', 'Fechamento'] * 16 + ['Proposta', 'Negociação'],
            'ProbabilidadeRecuperacao': [0.1 + (i % 10) * 0.05 for i in range(50)],
            'DataPrevista': pd.date_range(start='2023-06-01', periods=50, freq='D')
        })
        
        # Salva os dados
        vendas_df.to_csv("dados/vendas.csv", index=False)
        clientes_df.to_csv("dados/clientes.csv", index=False)
        vendas_perdidas_df.to_csv("dados/vendas_perdidas.csv", index=False)
    
    # Cria arquivos de configuração
    if not os.path.exists("datasources.json"):
        datasources = {
            "data_sources": [
                {
                    "id": "vendas",
                    "type": "csv",
                    "path": "dados/vendas.csv",
                    "delimiter": ",", 
                    "encoding": "utf-8"
                },
                {
                    "id": "clientes",
                    "type": "csv",
                    "path": "dados/clientes.csv",
                    "delimiter": ",",
                    "encoding": "utf-8"
                },
                {
                    "id": "vendas_perdidas",
                    "type": "csv",
                    "path": "dados/vendas_perdidas.csv",
                    "delimiter": ",",
                    "encoding": "utf-8"
                }
            ]
        }
        
        with open("datasources.json", "w") as f:
            json.dump(datasources, f, indent=2)
    
    # Cria pasta de saída
    if not os.path.exists("output"):
        os.makedirs("output", exist_ok=True)
    
    print("Dados e configurações preparados com sucesso.")


class SystemDemo:
    """Classe para demonstração do sistema"""
    
    def __init__(self):
        """Inicializa o sistema de demonstração"""
        load_example_data()
        
        print("Inicializando o motor de consulta...")
        self.engine = NaturalLanguageQueryEngine(
            data_config_path="datasources.json",
            base_data_path="dados"
        )
        
        # Diretório para salvar resultados
        self.output_dir = "output/demo"
        os.makedirs(self.output_dir, exist_ok=True)
        
        print("Sistema inicializado e pronto para uso.")
    
    def executar_consulta(self, consulta):
        """Executa uma consulta e mostra o resultado"""
        print(f"\n> Executando consulta: {consulta}")
        start_time = time.time()
        
        # Executa a consulta
        resposta = self.engine.execute_query(consulta)
        
        # Tempo de execução
        exec_time = time.time() - start_time
        print(f"[Tempo de execução: {exec_time:.2f}s]")
        
        # Processa a resposta de acordo com o tipo
        if hasattr(resposta, 'type'):
            print(f"Tipo de resposta: {resposta.type}")
            
            if resposta.type == "string":
                print(f"Resultado: {resposta.value}")
                
            elif resposta.type == "number":
                print(f"Resultado: {resposta.value}")
                
            elif resposta.type == "dataframe":
                print("Primeiras linhas do DataFrame resultante:")
                print(resposta.value.head())
                
                # Salva o DataFrame
                output_path = os.path.join(self.output_dir, f"consulta_{int(time.time())}.csv")
                resposta.value.to_csv(output_path, index=False)
                print(f"DataFrame completo salvo em: {output_path}")
                
            elif resposta.type == "plot":
                # Salva a visualização
                output_path = os.path.join(self.output_dir, f"visualizacao_{int(time.time())}.png")
                resposta.save(output_path)
                print(f"Visualização salva em: {output_path}")
                
                # Exibe se estiver em ambiente interativo
                try:
                    plt.show()
                except:
                    pass
        else:
            print(f"Resposta: {resposta}")
        
        return resposta
    
    def demo_consultas_basicas(self):
        """Demonstra consultas básicas"""
        print("\n" + "="*50)
        print("DEMONSTRAÇÃO DE CONSULTAS BÁSICAS")
        print("="*50)
        
        consultas = [
            "Mostre as primeiras 5 linhas da tabela de vendas",
            "Quantos registros existem na tabela de vendas?",
            "Qual é o valor total de vendas?",
            "Quais são os clientes da cidade de São Paulo?",
            "Qual é o valor médio das vendas?"
        ]
        
        for consulta in consultas:
            self.executar_consulta(consulta)
    
    def demo_agregacoes(self):
        """Demonstra consultas com agregações"""
        print("\n" + "="*50)
        print("DEMONSTRAÇÃO DE AGREGAÇÕES")
        print("="*50)
        
        consultas = [
            "Qual é o total de vendas por cliente?",
            "Qual cidade tem mais clientes?",
            "Qual é o cliente com maior valor de vendas?",
            "Quantas vendas temos por mês?",
            "Quantas vendas perdidas temos por motivo?"
        ]
        
        for consulta in consultas:
            self.executar_consulta(consulta)
    
    def demo_visualizacoes(self):
        """Demonstra consultas com visualizações"""
        print("\n" + "="*50)
        print("DEMONSTRAÇÃO DE VISUALIZAÇÕES")
        print("="*50)
        
        consultas = [
            "Mostre um gráfico de barras com o total de vendas por cliente",
            "Crie um histograma dos valores de venda",
            "Mostre a evolução de vendas ao longo do tempo em um gráfico de linha",
            "Crie um gráfico de barras mostrando o impacto financeiro por motivo de venda perdida",
            "Mostre um gráfico comparando as vendas por segmento de cliente"
        ]
        
        for consulta in consultas:
            self.executar_consulta(consulta)
    
    def demo_analises_complexas(self):
        """Demonstra análises mais complexas"""
        print("\n" + "="*50)
        print("DEMONSTRAÇÃO DE ANÁLISES COMPLEXAS")
        print("="*50)
        
        consultas = [
            "Quais clientes têm vendas acima da média?",
            "Compare o total de vendas por cidade e mostre em um gráfico",
            "Analise se existe correlação entre o valor da venda e o mês",
            "Identifique os produtos mais vendidos para clientes do segmento Corporativo",
            "Qual é o impacto financeiro total das vendas perdidas em cada estágio e mostre em um gráfico"
        ]
        
        for consulta in consultas:
            self.executar_consulta(consulta)
    
    def demo_consulta_interativa(self):
        """Modo interativo para consultas"""
        print("\n" + "="*50)
        print("MODO DE CONSULTA INTERATIVA")
        print("="*50)
        print("Digite suas consultas em linguagem natural. Digite 'sair' para encerrar.")
        
        while True:
            consulta = input("\n> ")
            if consulta.lower() in ['sair', 'exit', 'quit']:
                break
            
            if not consulta.strip():
                continue
                
            try:
                self.executar_consulta(consulta)
            except Exception as e:
                print(f"Erro ao processar consulta: {str(e)}")
    
    def executar_demo_completo(self):
        """Executa uma demonstração completa do sistema"""
        self.demo_consultas_basicas()
        self.demo_agregacoes()
        self.demo_visualizacoes()
        self.demo_analises_complexas()
        print("\nDemonstração completa executada com sucesso.")


def iniciar_api():
    """Inicia o servidor API"""
    import uvicorn
    from natural_query_engine import NaturalLanguageQueryEngine
    
    print("Inicializando o motor para a API...")
    api_app.engine = NaturalLanguageQueryEngine(
        data_config_path="datasources.json",
        base_data_path="dados"
    )
    
    print("Iniciando servidor API...")
    print("Acesse http://localhost:8000/docs para a documentação interativa")
    uvicorn.run(api_app, host="0.0.0.0", port=8000)


def main():
    """Função principal"""
    # Configura os argumentos de linha de comando
    parser = argparse.ArgumentParser(
        description="Sistema de Consulta em Linguagem Natural - Demonstração"
    )
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--api", action="store_true", help="Inicia o servidor API")
    group.add_argument("--query", "-q", type=str, help="Executa uma consulta específica")
    group.add_argument("--interactive", "-i", action="store_true", help="Modo de consulta interativa")
    group.add_argument("--demo", "-d", action="store_true", help="Executa a demonstração completa")
    
    args = parser.parse_args()
    
    # Garante que os dados de exemplo existem
    load_example_data()
    
    # Executa a opção solicitada
    if args.api:
        iniciar_api()
    elif args.query:
        demo = SystemDemo()
        demo.executar_consulta(args.query)
    elif args.interactive:
        demo = SystemDemo()
        demo.demo_consulta_interativa()
    elif args.demo:
        demo = SystemDemo()
        demo.executar_demo_completo()
    else:
        # Por padrão, executa um conjunto selecionado de demonstrações
        demo = SystemDemo()
        demo.demo_consultas_basicas()
        demo.demo_visualizacoes()
        
        # Termina com modo interativo
        print("\nAgora você pode fazer suas próprias consultas em linguagem natural:")
        demo.demo_consulta_interativa()


if __name__ == "__main__":
    main()