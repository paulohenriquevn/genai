"""
Exemplo de uso do sistema de consulta em linguagem natural refatorado.
"""

import pandas as pd
from natural_language_query_system import NaturalLanguageQuerySystem


def main():
    """
    Demonstração do uso do sistema de consulta em linguagem natural.
    """
    print("Inicializando sistema de consulta em linguagem natural...")
    
    # Inicializa o sistema
    # Para uso real, forneça sua API key e modelo específico:
    # nlq = NaturalLanguageQuerySystem(model_type="openai", model_name="gpt-4", api_key="sua-api-key")
    nlq = NaturalLanguageQuerySystem(model_type="mock")  # Modo simulado para demo
    
    # Carrega os datasets de exemplo
    print("\nCarregando datasets de exemplo...")
    
    # Dataset de vendas
    try:
        nlq.load_data("dados/vendas.csv", "vendas", 
                    description="Registro de vendas com produtos, valores e datas")
        print("Dataset 'vendas' carregado com sucesso.")
    except Exception as e:
        print(f"Erro ao carregar vendas: {str(e)}")
    
    # Dataset de clientes
    try:
        nlq.load_data("dados/clientes.csv", "clientes",
                    description="Informações de clientes incluindo dados demográficos")
        print("Dataset 'clientes' carregado com sucesso.")
    except Exception as e:
        print(f"Erro ao carregar clientes: {str(e)}")
    
    # Dataset de vendas perdidas
    try:
        nlq.load_data("dados/vendas_perdidas.csv", "vendas_perdidas",
                    description="Oportunidades de venda não concretizadas e motivos")
        print("Dataset 'vendas_perdidas' carregado com sucesso.")
    except Exception as e:
        print(f"Erro ao carregar vendas_perdidas: {str(e)}")
    
    # Lista datasets carregados
    datasets = nlq.list_datasets()
    print(f"\nDatasets disponíveis: {', '.join(datasets)}")
    
    # Exemplos de consultas em linguagem natural
    print("\n--- Exemplos de Consultas em Linguagem Natural ---")
    
    consultas = [
        "Qual o total de vendas por mês?",
        "Quais são os 5 clientes que mais compraram?",
        "Mostre um gráfico das vendas perdidas por motivo",
        "Qual é o ticket médio de compras por cliente?"
    ]
    
    for i, consulta in enumerate(consultas, 1):
        print(f"\nConsulta {i}: {consulta}")
        print("Processando...")
        
        try:
            resultado = nlq.ask(consulta)
            
            print(f"Tipo de resposta: {resultado.type}")
            
            if resultado.type == "dataframe":
                # Exibe apenas as primeiras 5 linhas para dataframes
                df = resultado.get_value()
                print(df.head(5))
            elif resultado.type == "plot":
                print(f"Gráfico gerado em: {resultado.get_value()}")
            else:
                print(resultado.get_value())
        except Exception as e:
            print(f"Erro ao processar consulta: {str(e)}")
    
    # Exemplo de consulta com feedback
    print("\n--- Exemplo de Consulta com Feedback ---")
    consulta = "Quais os principais motivos de vendas perdidas?"
    feedback = "Mostre em um gráfico de barras horizontal"
    
    print(f"Consulta: {consulta}")
    print(f"Feedback: {feedback}")
    print("Processando...")
    
    try:
        resultado = nlq.ask_with_feedback(consulta, feedback)
        
        print(f"Tipo de resposta: {resultado.type}")
        if resultado.type == "plot":
            print(f"Gráfico gerado em: {resultado.get_value()}")
        else:
            print(resultado.get_value())
    except Exception as e:
        print(f"Erro ao processar consulta com feedback: {str(e)}")
    
    # Exemplo de consulta SQL direta
    print("\n--- Exemplo de Consulta SQL Direta ---")
    sql = "SELECT * FROM vendas ORDER BY valor DESC LIMIT 3"
    print(f"SQL: {sql}")
    
    try:
        resultado = nlq.execute_sql(sql)
        print(resultado.get_value())
    except Exception as e:
        print(f"Erro ao executar SQL: {str(e)}")
    
    print("\nDemonstração concluída!")


if __name__ == "__main__":
    main()