#!/usr/bin/env python3
"""
Cenário de Teste 2: Análise de Dados Avançada
===========================================

Este cenário testa funcionalidades mais avançadas do sistema, incluindo:
- Análise de séries temporais
- Detecção de padrões sazonais
- Segmentação de dados
- Visualizações complexas
"""

import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Adiciona diretório pai ao PATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Importa componentes do sistema
from natural_query_engine import NaturalLanguageQueryEngine
from llm_integration import LLMIntegration, ModelType


def execute_scenario():
    """Executa o cenário de análise de dados avançada"""
    print("Iniciando Cenário 2: Análise de Dados Avançada")
    
    # Inicializa o motor de consulta
    engine = NaturalLanguageQueryEngine()
    
    # Utiliza modelo mock para teste
    llm = LLMIntegration(model_type=ModelType.MOCK)
    
    # Cria diretório para saída
    output_dir = os.path.join(os.getcwd(), "testes/output/scenario2")
    os.makedirs(output_dir, exist_ok=True)
    
    # Lista de consultas avançadas para testes
    advanced_queries = [
        # Análise temporal
        {
            "name": "time_series_trend",
            "query": "Mostre a tendência de vendas ao longo do tempo usando um gráfico de linhas",
            "description": "Análise de tendência temporal"
        },
        {
            "name": "monthly_comparison",
            "query": "Compare o total de vendas mês a mês. Mostre em um gráfico de barras.",
            "description": "Comparação mensal"
        },
        {
            "name": "seasonal_pattern",
            "query": "Analise se existe um padrão sazonal nas vendas e mostre em um gráfico",
            "description": "Detecção de sazonalidade"
        },
        
        # Segmentação de clientes
        {
            "name": "customer_segmentation",
            "query": "Agrupe os clientes por segmento e calcule o valor médio de compra por segmento",
            "description": "Segmentação de clientes"
        },
        {
            "name": "top_customers",
            "query": "Quais são os 5 clientes com maior volume de compras e de quais cidades eles são?",
            "description": "Identificação de principais clientes"
        },
        
        # Análise de vendas perdidas
        {
            "name": "lost_sales_impact",
            "query": "Qual é o impacto financeiro total das vendas perdidas por motivo?",
            "description": "Análise de impacto financeiro"
        },
        {
            "name": "recovery_probability",
            "query": "Analise a probabilidade de recuperação das vendas perdidas por estágio de perda",
            "description": "Análise de probabilidade de recuperação"
        },
        
        # Visualizações complexas
        {
            "name": "multi_dimension",
            "query": "Crie uma visualização que compare vendas por cliente e segmento",
            "description": "Visualização multidimensional"
        },
        {
            "name": "correlation_analysis",
            "query": "Existe correlação entre valor de venda e probabilidade de recuperação em vendas perdidas?",
            "description": "Análise de correlação"
        }
    ]
    
    # Executa cada consulta avançada e registra os resultados
    results = []
    for i, query_info in enumerate(advanced_queries):
        print(f"\nExecutando consulta avançada {i+1}/{len(advanced_queries)}: {query_info['name']}")
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
                "error": None,
                "timestamp": datetime.now().isoformat()
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
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            print(f"ERRO: {str(e)}")
        
        # Adiciona à lista de resultados
        results.append(result)
    
    # Teste de análise preditiva simples com LLM
    print("\nExecutando análise preditiva usando LLM...")
    prediction_query = """
    Analise os dados de vendas e crie uma previsão para os próximos 3 meses 
    baseada na tendência histórica. Use regressão linear simples.
    """
    
    try:
        # Gera código para a análise preditiva
        prediction_code = llm.generate_code(prediction_query)
        
        # Executa o código gerado
        execution_result = engine.code_executor.execute_code(
            code=prediction_code,
            context={
                'execute_sql_query': engine.execute_sql_query,
                'pd': pd,
                'plt': plt,
                'np': __import__('numpy'),
                'datetime': datetime
            }
        )
        
        # Registra o resultado da previsão
        if execution_result.get("success", False) and "result" in execution_result:
            # Processa o resultado
            result = engine.response_parser.parse(
                execution_result["result"], 
                last_code_executed=prediction_code
            )
            
            # Salva o resultado
            prediction_result = {
                "name": "predictive_analysis",
                "query": prediction_query,
                "description": "Análise preditiva para os próximos meses",
                "type": result.type,
                "success": True,
                "error": None,
                "timestamp": datetime.now().isoformat()
            }
            
            # Salva visualizações ou dados conforme o tipo
            if result.type == "plot":
                # Salva a visualização
                output_path = os.path.join(output_dir, "predictive_analysis.png")
                result.save(output_path)
                prediction_result["output_file"] = output_path
                print(f"Visualização preditiva salva em: {output_path}")
            else:
                # Outro tipo de resultado
                prediction_result["value"] = str(result.value)
                print(f"Resultado da análise preditiva: {result.value}")
        else:
            # Registra erro na execução
            prediction_result = {
                "name": "predictive_analysis",
                "query": prediction_query,
                "description": "Análise preditiva para os próximos meses",
                "success": False,
                "error": execution_result.get("error", "Erro desconhecido na execução"),
                "timestamp": datetime.now().isoformat()
            }
            print(f"ERRO na análise preditiva: {execution_result.get('error', 'Erro desconhecido')}")
        
        # Adiciona à lista de resultados
        results.append(prediction_result)
        
    except Exception as e:
        # Registra erro na análise preditiva
        prediction_result = {
            "name": "predictive_analysis",
            "query": prediction_query,
            "description": "Análise preditiva para os próximos meses",
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
        print(f"ERRO na análise preditiva: {str(e)}")
        results.append(prediction_result)
    
    # Salva o relatório de resultados em JSON
    report_path = os.path.join(output_dir, "results.json")
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Gera um relatório HTML para visualização dos resultados
    html_report = generate_html_report(results, output_dir)
    html_path = os.path.join(output_dir, "report.html")
    with open(html_path, 'w') as f:
        f.write(html_report)
    
    # Calcula e exibe estatísticas de execução
    success_count = sum(1 for r in results if r['success'])
    failure_count = len(results) - success_count
    success_rate = (success_count / len(results)) * 100
    
    print("\n" + "="*50)
    print(f"Cenário 2 concluído: {success_count}/{len(results)} consultas avançadas bem-sucedidas ({success_rate:.1f}%)")
    print(f"Relatório salvo em: {report_path}")
    print(f"Relatório HTML: {html_path}")
    print("="*50)
    
    return results


def generate_html_report(results, output_dir):
    """Gera um relatório HTML dos resultados para visualização"""
    html = """<!DOCTYPE html>
<html>
<head>
    <title>Relatório de Análise de Dados Avançada</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #2c3e50; }
        h2 { color: #3498db; }
        .query { border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px; }
        .success { border-left: 5px solid #2ecc71; }
        .failure { border-left: 5px solid #e74c3c; }
        .description { color: #7f8c8d; font-style: italic; }
        .visualization { max-width: 100%; margin: 10px 0; }
        .error { color: #e74c3c; font-family: monospace; background: #f9f9f9; padding: 10px; border-radius: 3px; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .summary { background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-top: 20px; }
    </style>
</head>
<body>
    <h1>Relatório de Análise de Dados Avançada</h1>
    <div class="summary">
        <h2>Resumo de Execução</h2>
        <p>Total de consultas: {total_queries}</p>
        <p>Consultas bem-sucedidas: {success_count} ({success_rate:.1f}%)</p>
        <p>Consultas com falha: {failure_count}</p>
        <p>Data de geração: {generation_date}</p>
    </div>
    
    <h2>Resultados das Consultas</h2>
    
    {query_results}
</body>
</html>
"""
    
    # Gera o conteúdo para cada consulta
    query_results = ""
    for result in results:
        success_class = "success" if result.get("success", False) else "failure"
        
        query_html = f"""
        <div class="query {success_class}">
            <h3>{result['name']}</h3>
            <p><strong>Consulta:</strong> {result['query']}</p>
            <p class="description">{result['description']}</p>
            <p><strong>Tipo de resultado:</strong> {result.get('type', 'N/A')}</p>
        """
        
        if result.get("success", False):
            # Adiciona visualização se disponível
            if result.get("output_file") and result.get("type") == "plot":
                # Caminho relativo para a imagem
                img_path = os.path.basename(result["output_file"])
                query_html += f'<p><strong>Visualização:</strong></p><img src="{img_path}" class="visualization" alt="Visualização">'
            
            # Adiciona tabela se for dataframe
            elif result.get("type") == "dataframe" and result.get("output_file"):
                try:
                    # Lê o CSV para mostrar uma prévia
                    df = pd.read_csv(result["output_file"])
                    table_html = df.head(5).to_html(classes="dataframe")
                    query_html += f'<p><strong>Prévia dos dados ({result.get("row_count", "?")} linhas total):</strong></p>{table_html}'
                except Exception:
                    query_html += f'<p><strong>Dados salvos em:</strong> {os.path.basename(result["output_file"])}</p>'
            
            # Adiciona valor se for string ou número
            elif result.get("value"):
                query_html += f'<p><strong>Resultado:</strong> {result["value"]}</p>'
        else:
            # Adiciona mensagem de erro
            query_html += f'<div class="error"><p><strong>Erro:</strong> {result.get("error", "Erro desconhecido")}</p></div>'
        
        query_html += f'<p><small>Timestamp: {result.get("timestamp", "N/A")}</small></p>'
        query_html += '</div>'
        
        query_results += query_html
    
    # Calcula estatísticas
    total_queries = len(results)
    success_count = sum(1 for r in results if r.get('success', False))
    failure_count = total_queries - success_count
    success_rate = (success_count / total_queries) * 100 if total_queries else 0
    generation_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Completa o template HTML
    html = html.format(
        total_queries=total_queries,
        success_count=success_count,
        failure_count=failure_count,
        success_rate=success_rate,
        generation_date=generation_date,
        query_results=query_results
    )
    
    return html


if __name__ == "__main__":
    # Executa o cenário de forma independente
    execute_scenario()