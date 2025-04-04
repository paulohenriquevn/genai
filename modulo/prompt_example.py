"""
Exemplo corrigido de uso do módulo para análise de dados com processamento em linguagem natural.
Este exemplo incorpora a classe DataFrameWrapper para resolver o erro de serialize_dataframe.
"""

import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import os
import sys

# Adiciona o diretório pai ao path para permitir importações relativas
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.agent.state import AgentState, AgentMemory, AgentConfig
from core.dataframe import DataFrameWrapper
from core.prompts import get_chat_prompt_for_sql, get_correct_error_prompt_for_sql
from core.response import ResponseParser
from core.response.error import ErrorResponse
from core.user_query import UserQuery
from core.exceptions import InvalidOutputValueMismatch, ExecuteSQLQueryNotUsed


# Simulando um modelo de IA para fins de demonstração
class MockAIModel:
    """Simula um modelo de IA para o exemplo"""
    
    def generate_code(self, prompt):
        """Simula a geração de código pelo modelo de IA"""
        print(f"[AI] Gerando código com base no prompt...\n")
        # Em um caso real, aqui seria a chamada para o modelo de IA
        # Retornamos um código de exemplo predefinido para demonstração
        return """
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

# Executando a consulta SQL para obter vendas por mês
df_vendas_por_mes = execute_sql_query('''
    SELECT 
        strftime('%Y-%m', data_venda) as mes,
        SUM(valor) as total_vendas,
        COUNT(*) as num_vendas
    FROM vendas
    GROUP BY mes
    ORDER BY mes
''')

# Preparando dados para visualização
meses = df_vendas_por_mes['mes'].tolist()
valores = df_vendas_por_mes['total_vendas'].tolist()

# Criando o gráfico
plt.figure(figsize=(10, 6))
plt.bar(meses, valores, color='skyblue')
plt.title('Total de Vendas por Mês')
plt.xlabel('Mês')
plt.ylabel('Valor Total (R$)')
plt.xticks(rotation=45)
plt.tight_layout()

# Salvando o gráfico em um buffer e convertendo para base64
buffer = io.BytesIO()
plt.savefig(buffer, format='png')
buffer.seek(0)
img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
img_data = f"data:image/png;base64,{img_str}"

# Definindo o resultado
result = {
    "type": "plot",
    "value": img_data
}
"""


# Simulando um executor SQL para fins de demonstração
def execute_sql_query(query, dataframes):
    """
    Simula a execução de uma consulta SQL em um dataframe
    Em um caso real, isso usaria uma biblioteca SQL como pandasql ou duckdb
    """
    print(f"[SQL] Executando consulta:\n{query}\n")
    # Aqui, simplificamos e apenas retornamos um dataframe simulado
    # Em um caso real, executaríamos a consulta SQL propriamente dita
    
    # Verificamos a consulta para simular o comportamento esperado
    if "vendas" in query.lower() and "group by" in query.lower():
        # Criando um dataframe de resultado simulado
        return pd.DataFrame({
            'mes': ['2023-01', '2023-02', '2023-03', '2023-04', '2023-05', '2023-06'],
            'total_vendas': [45000, 52000, 48000, 61000, 57000, 63000],
            'num_vendas': [120, 145, 135, 160, 150, 170]
        })
    else:
        # Retornamos um dataframe vazio para outras consultas
        return pd.DataFrame()


# Função para executar o código gerado em um ambiente controlado
def execute_code(code, dataframes):
    """
    Executa o código Python gerado em um ambiente controlado
    e retorna o resultado
    """
    print("[Executor] Executando código gerado...\n")
    # Variáveis locais para armazenar o resultado da execução
    local_vars = {'execute_sql_query': lambda query: execute_sql_query(query, dataframes)}
    
    try:
        # Executamos o código em um ambiente controlado
        exec(code, globals(), local_vars)
        
        # Verificamos se o resultado foi definido corretamente
        if 'result' in local_vars:
            print("[Executor] Código executado com sucesso!\n")
            return local_vars['result']
        else:
            raise ValueError("O código executado não definiu a variável 'result'")
    except Exception as e:
        print(f"[Executor] Erro na execução do código: {str(e)}\n")
        # Se há erro na execução, podemos verificar o tipo e retornar um prompt de correção
        if isinstance(e, NameError) and "execute_sql_query" in str(e):
            raise ExecuteSQLQueryNotUsed("O código não utiliza a função execute_sql_query corretamente")
        raise e


# Função principal que simula o fluxo completo
def process_natural_language_query(query_text, dataframes):
    """
    Processa uma consulta em linguagem natural e retorna o resultado
    
    Args:
        query_text: Texto da consulta do usuário
        dataframes: Lista de dataframes para análise (já devem ser DataFrameWrapper)
    
    Returns:
        Um objeto de resposta formatado
    """
    print(f"[Sistema] Processando consulta: '{query_text}'\n")
    
    # 1. Criando objetos de consulta e estado
    user_query = UserQuery(query_text)
    
    # Criando um objeto de memória e adicionando a consulta
    memory = AgentMemory("Assistente de análise de dados com SQL")
    memory.add_message(query_text)
    
    # Configuração do agente
    config = AgentConfig(direct_sql=True)
    
    # Estado do agente
    state = AgentState(
        dfs=dataframes, 
        memory=memory,
        config=config,
        output_type="plot"  # Especificando que queremos um gráfico como saída
    )
    
    # 2. Gerando o prompt para o modelo de IA
    prompt = get_chat_prompt_for_sql(state)
    prompt_text = prompt.to_string()
    print(f"[Sistema] Prompt gerado para o modelo de IA:\n{prompt_text[:300]}...\n")
    
    # 3. Enviando o prompt para o modelo de IA e obtendo o código
    ai_model = MockAIModel()
    generated_code = ai_model.generate_code(prompt_text)
    print(f"[Sistema] Código gerado pelo modelo:\n{generated_code[:300]}...\n")
    
    # Atualizando o último código gerado no estado
    state.set("last_code_generated", generated_code)
    
    # 4. Executando o código gerado
    try:
        execution_result = execute_code(generated_code, dataframes)
        
        # 5. Processando o resultado
        parser = ResponseParser()
        response = parser.parse(execution_result, last_code_executed=generated_code)
        
        print(f"[Sistema] Resposta processada com sucesso! Tipo: {response.type}\n")
        return response
        
    except ExecuteSQLQueryNotUsed as e:
        # Caso o código não use a função execute_sql_query, geramos um prompt de correção
        print(f"[Sistema] Erro: {str(e)}")
        correction_prompt = get_correct_error_prompt_for_sql(state, generated_code, str(e))
        # Aqui enviaríamos o prompt de correção para o modelo e repetiríamos o processo
        return ErrorResponse(
            "O código gerado não utiliza a função execute_sql_query corretamente. Tentando corrigir...",
            last_code_executed=generated_code,
            error=str(e)
        )
        
    except InvalidOutputValueMismatch as e:
        # Caso o tipo de saída não seja o esperado
        print(f"[Sistema] Erro: {str(e)}")
        correction_prompt = get_correct_output_type_error_prompt(state, generated_code, str(e))
        # Aqui enviaríamos o prompt de correção para o modelo e repetiríamos o processo
        return ErrorResponse(
            "O formato de saída do código não corresponde ao esperado. Tentando corrigir...",
            last_code_executed=generated_code,
            error=str(e)
        )
        
    except Exception as e:
        # Qualquer outro erro
        print(f"[Sistema] Erro geral: {str(e)}")
        return ErrorResponse(
            "Ocorreu um erro ao processar sua consulta.",
            last_code_executed=generated_code,
            error=str(e)
        )


# Demo com dados de exemplo
if __name__ == "__main__":
    # Criando um dataframe de vendas de exemplo
    vendas_df = pd.DataFrame({
        'id': range(1, 881),
        'data_venda': pd.date_range(start='2023-01-01', periods=880),
        'produto': ['Produto A', 'Produto B', 'Produto C', 'Produto D'] * 220,
        'valor': [300 + i % 700 for i in range(880)],
        'cliente_id': [(i % 50) + 1 for i in range(880)]
    })
    
    # Encapsulando o DataFrame com nossa classe wrapper
    vendas_wrapper = DataFrameWrapper(vendas_df, "vendas")
    
    # Consulta de exemplo
    query = "Mostre um gráfico de barras com o total de vendas por mês"
    
    # Processando a consulta
    result = process_natural_language_query(query, [vendas_wrapper])
    
    # Exibindo o resultado
    if result.type == "chart":
        print("\n[Sistema] Resultado: Gráfico gerado\n")
        # Em uma aplicação real, aqui exibiríamos o gráfico ou salvaríamos em um arquivo
        try:
            # Salvando o gráfico em um arquivo
            output_path = "vendas_por_mes.png"
            result.save(output_path)
            print(f"[Sistema] Gráfico salvo em: {output_path}")
        except Exception as e:
            print(f"[Sistema] Erro ao salvar o gráfico: {str(e)}")
    else:
        print(f"\n[Sistema] Resultado ({result.type}): {result.value}\n")