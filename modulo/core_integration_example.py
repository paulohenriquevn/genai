#!/usr/bin/env python3
"""
Exemplo de Integração do Módulo Core
====================================

Este script demonstra como utilizar o módulo de integração core
para análise de dados com processamento de linguagem natural.

Exemplos:
    $ python core_integration_example.py
"""

import os
from core_integration import AnalysisEngine
from core.response.dataframe import DataFrameResponse
from core.response.chart import ChartResponse
from core.response.string import StringResponse
from core.response.number import NumberResponse
from core.response.error import ErrorResponse


def main():
    """Função principal para demonstração do módulo de integração core."""
    print("🔧 Inicializando Motor de Análise...")
    
    # Verifica variáveis de ambiente para chaves de API
    openai_key = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    
    # Determina o modelo a usar com base nas chaves disponíveis
    model_type = "mock"  # Padrão
    model_name = None
    api_key = None
    
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
        print("  ℹ️ Nenhuma chave de API encontrada. Usando modo simulado (mock).")
    
    # Inicializa o motor de análise
    engine = AnalysisEngine(
        agent_description="Assistente avançado de análise de dados com foco em insights de vendas",
        default_output_type="dataframe",
        direct_sql=True,
        model_type=model_type,
        model_name=model_name,
        api_key=api_key
    )
    
    # Carrega os datasets disponíveis
    print("\n📊 Carregando datasets...")
    
    # Verifica se os arquivos existem
    data_files = {
        "vendas": "dados/vendas.csv",
        "clientes": "dados/clientes.csv",
        "vendas_perdidas": "dados/vendas_perdidas.csv"
    }
    
    # Carrega os datasets disponíveis
    for name, file_path in data_files.items():
        if os.path.exists(file_path):
            engine.load_data(
                data=file_path,
                name=name,
                description=f"Dados de {name} da empresa"
            )
            print(f"  ✅ Dataset '{name}' carregado com sucesso")
        else:
            print(f"  ❌ Arquivo não encontrado: {file_path}")
    
    # Lista os datasets carregados
    print(f"\n📋 Datasets disponíveis: {', '.join(engine.list_datasets())}")
    
    # Define consultas de exemplo
    example_queries = [
        "Qual é o total de vendas por cliente?",
        "Quais são os 3 principais motivos de vendas perdidas?",
        "Gere um gráfico de barras mostrando o impacto financeiro por motivo de vendas perdidas",
        "Qual é o valor médio de vendas?",
        "Mostre os clientes de São Paulo",
        "Quais produtos estão com estoque baixo?",
        "Qual é o total de vendas por mês?",
        "Quais são os produtos mais vendidos?",
        "Mostre as vendas por região",
    ]
    
    # Processar algumas consultas de exemplo
    print("\n🔍 Processando consultas de exemplo...\n")
    
    for i, query in enumerate(example_queries, 1):
        print(f"\n--------- Consulta {i} ---------")
        print(f"📝 Consulta: {query}")
        
        # Processa a consulta
        result = engine.process_query(query)
        
        # Exibe o resultado de acordo com o tipo
        print(f"🔄 Tipo de resposta: {result.type}")
        
        if isinstance(result, DataFrameResponse):
            if len(result.value) > 5:
                print(f"📊 Resultado (primeiras 5 linhas de {len(result.value)}):")
                print(result.value.head(5))
            else:
                print(f"📊 Resultado ({len(result.value)} linhas):")
                print(result.value)
                
        elif isinstance(result, ChartResponse):
            print(f"📈 Gráfico gerado: {result.value}")
            
        elif isinstance(result, NumberResponse):
            print(f"🔢 Resultado numérico: {result.value}")
            
        elif isinstance(result, StringResponse):
            print(f"📝 Resultado textual: {result.value}")
            
        elif isinstance(result, ErrorResponse):
            print(f"❌ Erro: {result.value}")
            
        print("-" * 30)
    
    # Demonstração interativa
    print("\n💬 Modo interativo. Digite 'sair' para encerrar.")
    
    while True:
        user_query = input("\nDigite sua consulta: ")
        
        if user_query.lower() in ['sair', 'exit', 'quit']:
            break
            
        # Processa a consulta do usuário
        result = engine.process_query(user_query)
        
        # Exibe o resultado
        print(f"\n🔄 Tipo de resposta: {result.type}")
        
        if isinstance(result, DataFrameResponse):
            if len(result.value) > 5:
                print(f"📊 Resultado (primeiras 5 linhas de {len(result.value)}):")
                print(result.value.head(5))
            else:
                print(f"📊 Resultado ({len(result.value)} linhas):")
                print(result.value)
                
        elif isinstance(result, ChartResponse):
            print(f"📈 Gráfico gerado: {result.value}")
            
        elif isinstance(result, NumberResponse):
            print(f"🔢 Resultado numérico: {result.value}")
            
        elif isinstance(result, StringResponse):
            print(f"📝 Resultado textual: {result.value}")
            
        elif isinstance(result, ErrorResponse):
            print(f"❌ Erro: {result.value}")
    
    print("\n👋 Sessão encerrada!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Sessão interrompida pelo usuário!")
    except Exception as e:
        print(f"\n❌ Erro: {str(e)}")