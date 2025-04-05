#!/usr/bin/env python3
"""
Exemplo de integração do analisador de datasets com o AnalysisEngine.

Este script demonstra como o analisador de datasets melhora a geração
de consultas SQL através da detecção automática de estrutura e relacionamentos.
"""

import os
import pandas as pd
import logging
from pprint import pprint

from core_integration import AnalysisEngine, Dataset
from utils.dataset_analyzer import DatasetAnalyzer

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    print("\n" + "="*70)
    print("🔎 Sistema de Consulta Aprimorado com Análise Dinâmica de Datasets")
    print("="*70)
    
    # Detecta credenciais de API para LLM (OpenAI ou Anthropic)
    openai_key = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    
    print("\n🔑 Inicializando motor de análise...")
    if openai_key:
        model_type = "openai"
        model_name = "gpt-3.5-turbo"
        api_key = openai_key
        print("  🔑 Chave OpenAI encontrada. Usando modelo GPT-3.5.")
    elif anthropic_key:
        model_type = "anthropic"
        model_name = "claude-3-haiku-20240307"
        api_key = anthropic_key
        print("  🔑 Chave Anthropic encontrada. Usando modelo Claude Haiku.")
    else:
        model_type = "mock"
        model_name = None
        api_key = None
        print("  ℹ️ Nenhuma chave de API encontrada. Usando modo simulado.")
    
    # Inicializa o motor de análise
    engine = AnalysisEngine(
        agent_description="Assistente de Análise de Dados Inteligente com Detecção Automática de Estrutura",
        default_output_type="dataframe",
        direct_sql=False,
        model_type=model_type,
        model_name=model_name,
        api_key=api_key
    )
    
    # Carrega datasets de exemplo
    print("\n📊 Carregando datasets...")
    dados_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dados")
    
    # Carrega o dataset de vendas
    vendas_path = os.path.join(dados_dir, "vendas.csv")
    engine.load_data(vendas_path, "vendas", "Registro de vendas com data, valor e cliente")
    print("  ✅ Dataset 'vendas' carregado com sucesso")
    
    # Carrega o dataset de clientes
    clientes_path = os.path.join(dados_dir, "clientes.csv")
    engine.load_data(clientes_path, "clientes", "Cadastro de clientes com nome e localização")
    print("  ✅ Dataset 'clientes' carregado com sucesso")
    
    # Carrega o dataset de vendas perdidas
    vendas_perdidas_path = os.path.join(dados_dir, "vendas_perdidas.csv")
    engine.load_data(vendas_perdidas_path, "vendas_perdidas", "Registro de oportunidades de vendas perdidas e seus motivos")
    print("  ✅ Dataset 'vendas_perdidas' carregado com sucesso")
    
    # Exibe informações sobre os datasets e relacionamentos detectados
    print("\n📋 Datasets disponíveis:", ", ".join(engine.list_datasets()))
    
    # Exibe metadados detectados
    print("\n🔍 Metadados detectados automaticamente:")
    for name in engine.list_datasets():
        dataset = engine.get_dataset(name)
        print(f"\n  📊 Dataset: {name}")
        print(f"    • Descrição: {dataset.description}")
        print(f"    • Registros: {len(dataset.dataframe)}")
        
        if dataset.primary_key:
            print(f"    • Chave Primária: {dataset.primary_key}")
        
        if dataset.potential_foreign_keys:
            print(f"    • Chaves Estrangeiras Potenciais: {', '.join(dataset.potential_foreign_keys)}")
        
        # Exibe tipos de colunas detectados
        if hasattr(dataset, 'column_types') and dataset.column_types:
            print("    • Tipos de colunas detectados:")
            for col, col_type in dataset.column_types.items():
                # Marca se for chave primária ou estrangeira
                suffix = ""
                if col == dataset.primary_key:
                    suffix = " (Chave Primária)"
                elif col in dataset.potential_foreign_keys:
                    suffix = " (Chave Estrangeira)"
                
                print(f"      - {col}: {col_type}{suffix}")
    
    # Exibe relacionamentos detectados
    print("\n🔗 Relacionamentos detectados:")
    relationships_found = False
    
    for name, dataset in engine.datasets.items():
        if hasattr(dataset, 'analyzed_metadata') and dataset.analyzed_metadata:
            if 'relationships' in dataset.analyzed_metadata:
                rel_info = dataset.analyzed_metadata['relationships']
                
                # Relações outgoing
                if 'outgoing' in rel_info and rel_info['outgoing']:
                    relationships_found = True
                    for rel in rel_info['outgoing']:
                        print(f"  • {name}.{rel['source_column']} → {rel['target_dataset']}.{rel['target_column']}")
    
    if not relationships_found:
        print("  Nenhum relacionamento detectado automaticamente.")
    
    # Demonstração do motor de consulta
    print("\n🔄 Processando consultas de exemplo...")
    
    # Lista de consultas para testar
    queries = [
        "Qual é o total de vendas por cliente?",
        "Quais são os 3 principais motivos de vendas perdidas?",
        "Qual é o valor médio de vendas?",
        "Mostre os clientes de São Paulo",
        "Liste todas as vendas do cliente João Silva"
    ]
    
    # Processa cada consulta
    for i, query in enumerate(queries, 1):
        print(f"\n--------- Consulta {i} ---------")
        print(f"📝 Consulta: {query}")
        
        # Processa a consulta
        response = engine.process_query(query)
        
        # Exibe o tipo de resposta
        print(f"🔄 Tipo de resposta: {response.type}")
        
        # Exibe o resultado baseado no tipo
        if response.type == "dataframe":
            # Limita a exibição para datasets grandes
            df = response.value
            max_rows = 5
            print(f"📊 Resultado ({min(len(df), max_rows)} linhas):")
            print(df.head(max_rows))
        elif response.type == "number":
            print(f"🔢 Resultado numérico: {response.value}")
        elif response.type == "string":
            print(f"📝 Resultado textual: {response.value}")
        elif response.type == "plot":
            print(f"📈 Visualização gerada: {response.value}")
        elif response.type == "error":
            print(f"❌ Erro: {response.value}")
        
        print("-" * 30)
    
    # Modo interativo opcional
    print("\n💬 Modo interativo. Digite 'sair' para encerrar.")
    while True:
        user_query = input("\nDigite sua consulta: ")
        if user_query.lower() in ['sair', 'exit', 'quit']:
            break
            
        # Processa a consulta do usuário
        response = engine.process_query(user_query)
        
        # Exibe o tipo de resposta
        print(f"🔄 Tipo de resposta: {response.type}")
        
        # Exibe o resultado baseado no tipo
        if response.type == "dataframe":
            # Limita a exibição para datasets grandes
            df = response.value
            max_rows = 5
            print(f"📊 Resultado ({min(len(df), max_rows)} linhas):")
            print(df.head(max_rows))
        elif response.type == "number":
            print(f"🔢 Resultado numérico: {response.value}")
        elif response.type == "string":
            print(f"📝 Resultado textual: {response.value}")
        elif response.type == "plot":
            print(f"📈 Visualização gerada: {response.value}")
        elif response.type == "error":
            print(f"❌ Erro: {response.value}")

if __name__ == "__main__":
    main()