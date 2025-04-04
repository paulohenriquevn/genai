#!/usr/bin/env python3
"""
Cenário de Teste 1: Consultas Básicas
====================================

Este cenário testa as operações básicas do sistema com consultas simples,
incluindo seleções, filtros, agregações e visualizações básicas.
"""

import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import shutil

# Adiciona diretório pai ao PATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Importa componentes do sistema
from natural_query_engine import NaturalLanguageQueryEngine


def execute_scenario():
    """Executa o cenário de teste de consultas básicas"""
    print("Iniciando Cenário 1: Consultas Básicas")
    
    # Inicializa o motor de consulta
    engine = NaturalLanguageQueryEngine()
    
    # Lista fontes de dados disponíveis
    sources = list(engine.dataframes.keys())
    print(f"Fontes de dados disponíveis: {', '.join(sources)}")
    
    # Cria diretório para saída
    output_dir = os.path.join(os.getcwd(), "testes/output/scenario1")
    os.makedirs(output_dir, exist_ok=True)
    
    # Lista de consultas básicas para teste
    basic_queries = [
        # Consultas SELECT básicas
        {
            "name": "select_basic",
            "query": "Mostre as primeiras 5 linhas da tabela de vendas",
            "description": "SELECT básico com limitação de linhas"
        },
        {
            "name": "select_where",
            "query": "Mostre os clientes da cidade de São Paulo",
            "description": "SELECT com filtro WHERE"
        },
        
        # Consultas de contagem e totalização
        {
            "name": "count_total",
            "query": "Quantos registros existem na tabela de vendas?",
            "description": "Contagem total de registros"
        },
        {
            "name": "sum_total",
            "query": "Qual é o valor total de vendas?",
            "description": "Soma total de uma coluna"
        },
        
        # Consultas com agregação
        {
            "name": "avg_value",
            "query": "Qual é o valor médio das vendas?",
            "description": "Cálculo de média"
        },
        {
            "name": "max_value",
            "query": "Qual é o maior valor de venda registrado?",
            "description": "Valor máximo"
        },
        
        # Consultas com GROUP BY
        {
            "name": "group_by_city",
            "query": "Quantos clientes temos por cidade?",
            "description": "GROUP BY com contagem"
        },
        {
            "name": "group_by_sum",
            "query": "Qual é o total de vendas por cliente?",
            "description": "GROUP BY com soma"
        },
        
        # Consultas com visualização
        {
            "name": "bar_chart",
            "query": "Crie um gráfico de barras mostrando o total de vendas por cliente",
            "description": "Visualização com gráfico de barras"
        },
        {
            "name": "histogram",
            "query": "Mostre um histograma dos valores de venda",
            "description": "Visualização com histograma"
        }
    ]
    
    # Executa cada consulta e registra os resultados
    results = []
    for i, query_info in enumerate(basic_queries):
        print(f"\nExecutando consulta {i+1}/{len(basic_queries)}: {query_info['name']}")
        print(f"Consulta: {query_info['query']}")
        
        try:
            # Executa a consulta
            response = engine.execute_query(query_info['query'])
            
            # Registra o resultado
            result = {
                "name": query_info['name'],
                "query": query_info['query'],
                "description": query_info['description'],
                "type": response.type,
                "success": True,
                "error": None
            }
            
            # Salva visualizações ou dados conforme o tipo
            if response.type == "plot":
                # Salva a visualização
                output_path = os.path.join(output_dir, f"{query_info['name']}.png")
                response.save(output_path)
                result["output_file"] = output_path
                print(f"Visualização salva em: {output_path}")
            
            elif response.type == "dataframe":
                # Salva o DataFrame
                output_path = os.path.join(output_dir, f"{query_info['name']}.csv")
                response.value.to_csv(output_path, index=False)
                result["output_file"] = output_path
                result["row_count"] = len(response.value)
                result["columns"] = list(response.value.columns)
                print(f"DataFrame salvo em: {output_path} ({len(response.value)} linhas)")
            
            else:
                # Valor simples (string ou número)
                result["value"] = str(response.value)
                print(f"Resultado: {response.value}")
            
        except Exception as e:
            # Registra erro
            result = {
                "name": query_info['name'],
                "query": query_info['query'],
                "description": query_info['description'],
                "success": False,
                "error": str(e)
            }
            print(f"ERRO: {str(e)}")
        
        # Adiciona à lista de resultados
        results.append(result)
    
    # Salva o relatório de resultados em JSON
    report_path = os.path.join(output_dir, "results.json")
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Calcula e exibe estatísticas de execução
    success_count = sum(1 for r in results if r['success'])
    failure_count = len(results) - success_count
    success_rate = (success_count / len(results)) * 100
    
    print("\n" + "="*50)
    print(f"Cenário 1 concluído: {success_count}/{len(results)} consultas bem-sucedidas ({success_rate:.1f}%)")
    print(f"Relatório salvo em: {report_path}")
    print("="*50)
    
    return results


if __name__ == "__main__":
    # Executa o cenário de forma independente
    execute_scenario()