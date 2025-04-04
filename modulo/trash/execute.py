"""
Exemplo de uso do Motor de Consulta em Linguagem Natural
=======================================================

Este script demonstra como usar o motor de consulta em linguagem natural
para analisar dados em diversos contextos com comandos em português.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from natural_query_engine import NaturalLanguageQueryEngine

# Cria uma pasta temporária para as saídas
output_dir = "exemplos_output"
os.makedirs(output_dir, exist_ok=True)

def testar_consultas():
    """Execute uma bateria de consultas para testar o motor"""
    print("\n===== Testando o Motor de Consulta em Linguagem Natural =====\n")
    
    # Inicializa o motor com configurações padrão
    engine = NaturalLanguageQueryEngine(
        data_config_path="datasources.json",
        metadata_config_path="metadata.json",
        base_data_path="dados"
    )
    
    # Lista as fontes de dados disponíveis
    fontes_disponiveis = engine.dataframes.keys()
    print(f"Fontes de dados disponíveis: {', '.join(fontes_disponiveis)}")
    
    # Lista de consultas para teste
    consultas = [
        # Consultas básicas
        "Quantos registros existem na tabela de vendas?",
        "Mostre os primeiros 5 registros de clientes",
        
        # Consultas com agregações
        "Qual é o valor total de vendas?",
        "Qual é o valor médio de impacto financeiro nas vendas perdidas?",
        
        # Consultas com agrupamento
        "Agrupe as vendas perdidas por motivo e mostre o total de impacto financeiro",
        "Quais são as cidades com mais clientes?",
        
        # Consultas com visualização
        "Crie um gráfico mostrando o impacto financeiro por motivo de venda perdida",
        "Faça uma visualização do total de vendas por cliente"
    ]
    
    # Processa cada consulta
    for i, consulta in enumerate(consultas):
        print(f"\n\n{'='*50}")
        print(f"CONSULTA {i+1}: {consulta}")
        print(f"{'='*50}\n")
        
        try:
            # Processa a consulta
            resposta = engine.execute_query(consulta)
            
            # Exibe a resposta formatada com base no tipo
            print("\n----- RESPOSTA -----\n")
            
            if resposta.type == "dataframe":
                print(resposta.value.head(10))
                print(f"\n[{len(resposta.value)} linhas no total]")
                
                # Salva o dataframe como CSV
                nome_arquivo = f"{output_dir}/resultado_consulta_{i+1}.csv"
                resposta.value.to_csv(nome_arquivo, index=False)
                print(f"\nResultado salvo em: {nome_arquivo}")
                
            elif resposta.type == "plot":
                print("[Visualização gerada]")
                
                # Salva a visualização
                nome_arquivo = f"{output_dir}/visualizacao_consulta_{i+1}.png"
                resposta.save(nome_arquivo)
                print(f"\nVisualização salva em: {nome_arquivo}")
                
            else:
                print(resposta.value)
                
            print("\n-------------------\n")
            
        except Exception as e:
            print(f"\nERRO AO PROCESSAR CONSULTA: {str(e)}\n")
    
    # Exibe estatísticas
    stats = engine.get_stats()
    print("\n===== Estatísticas de Uso =====\n")
    print(f"Total de consultas: {stats['total_queries']}")
    print(f"Consultas bem-sucedidas: {stats['successful_queries']}")
    print(f"Taxa de sucesso: {stats['success_rate']:.1f}%")
    print(f"Fontes de dados disponíveis: {', '.join(stats['loaded_dataframes'])}")

def exemplo_analise_vendas_perdidas():
    """Exemplo específico de análise do conjunto de vendas perdidas"""
    print("\n===== Análise de Vendas Perdidas =====\n")
    
    # Inicializa o motor
    engine = NaturalLanguageQueryEngine()
    
    # Sequência de consultas analíticas
    analises = [
        "Qual é o total de impacto financeiro das vendas perdidas?",
        "Qual é o motivo mais comum para vendas perdidas?",
        "Qual estágio tem maior impacto financeiro: Proposta ou Negociação?",
        "Mostre um gráfico comparando o impacto financeiro por motivo",
        "Calcule o impacto financeiro médio por estágio de perda"
    ]
    
    # Executa as consultas em sequência, como uma análise progressiva
    for i, consulta in enumerate(analises):
        print(f"\n--- Análise {i+1}: {consulta} ---\n")
        
        resposta = engine.execute_query(consulta)
        
        if resposta.type == "dataframe":
            print(resposta.value)
        elif resposta.type == "plot":
            print("[Visualização gerada]")
            nome_arquivo = f"{output_dir}/analise_vendas_perdidas_{i+1}.png"
            resposta.save(nome_arquivo)
            print(f"Visualização salva em: {nome_arquivo}")
        else:
            print(resposta.value)
        
        print("\n")
    
    print("Análise de vendas perdidas concluída!")

def interacao_usuario():
    """Interface para interação direta do usuário com o motor"""
    print("\n===== Interação com o Motor de Consulta =====\n")
    
    # Inicializa o motor
    engine = NaturalLanguageQueryEngine()
    
    print(f"Fontes de dados disponíveis: {', '.join(engine.dataframes.keys())}")
    print("\nDigite suas consultas em linguagem natural. Digite 'sair' para encerrar.\n")
    
    while True:
        # Solicita a consulta
        consulta = input("\nConsulta > ")
        
        if consulta.lower() in ['sair', 'exit', 'quit']:
            break
        
        if not consulta.strip():
            continue
        
        try:
            # Processa a consulta
            print("\nProcessando consulta...\n")
            resposta = engine.execute_query(consulta)
            
            # Exibe a resposta formatada
            print("\n----- Resposta -----\n")
            
            if resposta.type == "dataframe":
                print(resposta.value.head(10))
                print(f"\n[{len(resposta.value)} linhas no total]")
            elif resposta.type == "plot":
                print("[Visualização gerada]")
                nome_arquivo = f"{output_dir}/consulta_interativa.png"
                resposta.save(nome_arquivo)
                print(f"Visualização salva em: {nome_arquivo}")
            else:
                print(resposta.value)
                
            print("\n-------------------\n")
            
        except Exception as e:
            print(f"\nErro ao processar consulta: {str(e)}\n")
    
    print("\nObrigado por usar o Motor de Consulta em Linguagem Natural!")

if __name__ == "__main__":
    # Execute cada modo de demonstração
    testar_consultas()
    exemplo_analise_vendas_perdidas()
    interacao_usuario()