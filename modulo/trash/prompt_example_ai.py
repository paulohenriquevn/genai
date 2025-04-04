"""
Exemplo que demonstra o uso do módulo com um modelo de IA simulado mais sofisticado.
Este exemplo mostra como implementar um ciclo completo de geração e correção de código.
"""

import pandas as pd
import os
import sys
import json

# Adiciona o diretório pai ao path para permitir importações relativas
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.agent.state import AgentState, AgentMemory, AgentConfig
from core.prompts import get_chat_prompt_for_sql, get_correct_error_prompt_for_sql
from core.response import ResponseParser
from core.response.error import ErrorResponse
from core.user_query import UserQuery
from core.exceptions import ExecuteSQLQueryNotUsed


class SophisticatedMockAI:
    """
    Um modelo de IA simulado mais sofisticado que pode gerar diferentes tipos de código
    e responder a prompts de correção.
    """
    
    def __init__(self):
        # Mapeamento de consultas comuns para os resultados esperados
        self.query_patterns = {
            "média": self._generate_average_code,
            "soma": self._generate_sum_code,
            "conta": self._generate_count_code,
            "máximo": self._generate_max_code,
            "mínimo": self._generate_min_code,
            "gráfico": self._generate_chart_code,
            "corrigir": self._generate_corrected_code,
        }
    
    def generate_response(self, prompt):
        """
        Gera uma resposta com base no prompt fornecido.
        
        Args:
            prompt: O prompt para o modelo de IA
            
        Returns:
            O código gerado em resposta ao prompt
        """
        # Verificamos se é um prompt de correção
        if "Fix the python code" in prompt or "resulted in the following error" in prompt:
            return self._generate_corrected_code(prompt)
        
        # Analisamos o prompt para determinar o tipo de consulta
        for keyword, generator in self.query_patterns.items():
            if keyword.lower() in prompt.lower():
                return generator(prompt)
        
        # Se não identificamos o padrão, retornamos um código padrão
        return self._generate_default_code(prompt)
    
    def _generate_average_code(self, prompt):
        """Gera código para cálculo de média"""
        return """
import pandas as pd

# Calculando a média dos valores
result_df = execute_sql_query('''
    SELECT AVG(valor) as media_valor
    FROM dados
''')

# Extraindo o valor
media = result_df['media_valor'].iloc[0]

# Definindo o resultado
result = {
    "type": "number",
    "value": media
}
"""

    def _generate_sum_code(self, prompt):
        """Gera código para cálculo de soma"""
        return """
import pandas as pd

# Calculando a soma dos valores
result_df = execute_sql_query('''
    SELECT SUM(valor) as soma_valor
    FROM dados
''')

# Extraindo o valor
soma = result_df['soma_valor'].iloc[0]

# Definindo o resultado
result = {
    "type": "number",
    "value": soma
}
"""

    def _generate_count_code(self, prompt):
        """Gera código para contagem de registros"""
        return """
import pandas as pd

# Contando registros
result_df = execute_sql_query('''
    SELECT COUNT(*) as total
    FROM dados
''')

# Extraindo o valor
total = result_df['total'].iloc[0]

# Definindo o resultado
result = {
    "type": "number",
    "value": total
}
"""

    def _generate_max_code(self, prompt):
        """Gera código para encontrar o valor máximo"""
        return """
import pandas as pd

# Encontrando o valor máximo
result_df = execute_sql_query('''
    SELECT MAX(valor) as valor_maximo
    FROM dados
''')

# Extraindo o valor
maximo = result_df['valor_maximo'].iloc[0]

# Definindo o resultado
result = {
    "type": "number",
    "value": maximo
}
"""

    def _generate_min_code(self, prompt):
        """Gera código para encontrar o valor mínimo"""
        return """
import pandas as pd

# Encontrando o valor mínimo
result_df = execute_sql_query('''
    SELECT MIN(valor) as valor_minimo
    FROM dados
''')

# Extraindo o valor
minimo = result_df['valor_minimo'].iloc[0]

# Definindo o resultado
result = {
    "type": "string",
    "value": f"O valor mínimo é {minimo}"
}
"""

    def _generate_chart_code(self, prompt):
        """Gera código para criar um gráfico"""
        return """
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

# Obtendo dados agrupados
df_result = execute_sql_query('''
    SELECT categoria, SUM(valor) as total
    FROM dados
    GROUP BY categoria
    ORDER BY total DESC
''')

# Criando o gráfico
plt.figure(figsize=(10, 6))
plt.bar(df_result['categoria'], df_result['total'])
plt.title('Total por Categoria')
plt.xlabel('Categoria')
plt.ylabel('Total')
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

    def _generate_corrected_code(self, prompt):
        """
        Gera código corrigido com base no prompt de correção.
        Se o erro for sobre execute_sql_query, corrige esse aspecto.
        """
        if "use execute_sql_query function" in prompt:
            return """
import pandas as pd

# Usando a função execute_sql_query corretamente
result_df = execute_sql_query('''
    SELECT categoria, SUM(valor) as total
    FROM dados
    GROUP BY categoria
    ORDER BY total DESC
''')

# Definindo o resultado
result = {
    "type": "dataframe",
    "value": result_df
}
"""
        elif "result type should be" in prompt:
            # Extraindo o tipo esperado
            if "should be: number" in prompt:
                return """
import pandas as pd

# Obtendo dados
result_df = execute_sql_query('''
    SELECT SUM(valor) as total
    FROM dados
''')

# Definindo o resultado com o tipo correto
result = {
    "type": "number",
    "value": result_df['total'].iloc[0]
}
"""
            elif "should be: string" in prompt:
                return """
import pandas as pd

# Obtendo dados
result_df = execute_sql_query('''
    SELECT SUM(valor) as total
    FROM dados
''')

# Definindo o resultado com o tipo correto
total = result_df['total'].iloc[0]
result = {
    "type": "string",
    "value": f"O total é {total}"
}
"""
            else:
                return """
import pandas as pd

# Obtendo dados
result_df = execute_sql_query('''
    SELECT * FROM dados
    LIMIT 100
''')

# Definindo o resultado com o tipo correto
result = {
    "type": "dataframe",
    "value": result_df
}
"""
        else:
            # Correção genérica
            return """
import pandas as pd

# Código corrigido
result_df = execute_sql_query('''
    SELECT * FROM dados
    LIMIT 10
''')

# Definindo o resultado
result = {
    "type": "dataframe",
    "value": result_df
}
"""

    def _generate_default_code(self, prompt):
        """Gera um código padrão para consultas não identificadas"""
        return """
import pandas as pd

# Consultando os dados
df_result = pd.DataFrame()  # Não uso execute_sql_query para demonstrar correção

# Definindo o resultado
result = {
    "type": "string",
    "value": "Não foi possível identificar uma consulta específica. Por favor, tente ser mais específico."
}
"""


# Função para executar o código gerado
def execute_code_with_correction(code, dataframes, state):
    """
    Executa o código Python gerado e lida com erros, incluindo correções.
    
    Args:
        code: Código Python gerado
        dataframes: Lista de dataframes para uso
        state: Estado do agente
    
    Returns:
        Objeto de resposta
    """
    print("[Executor] Tentando executar código...\n")
    
    # Nome dos dataframes para uso em SQL
    df_names = {df.name if hasattr(df, 'name') else f'df_{i}': df 
                for i, df in enumerate(dataframes)}
    
    # Variáveis locais para armazenar o resultado da execução
    local_vars = {}
    
    # Definindo a função execute_sql_query
    def mock_execute_sql_query(query):
        """Simula a execução de SQL em dataframes"""
        print(f"[SQL] Executando: {query}")
        
        # Em uma implementação real, usaria uma biblioteca como pandasql
        # Para simular, retornamos um dataframe de exemplo com alguns dados
        import pandas as pd
        if "categoria" in query.lower():
            return pd.DataFrame({
                'categoria': ['A', 'B', 'C', 'D'],
                'total': [100, 200, 150, 300]
            })
        elif "sum" in query.lower() or "avg" in query.lower() or "count" in query.lower():
            return pd.DataFrame({
                'total': [750]
            })
        else:
            # Retornamos o primeiro dataframe com algumas linhas para simulação
            if dataframes:
                return dataframes[0].head(10)
            return pd.DataFrame({'exemplo': [1, 2, 3]})
    
    local_vars['execute_sql_query'] = mock_execute_sql_query
    
    # Parser para processar o resultado
    parser = ResponseParser()
    
    try:
        # Executamos o código em um ambiente controlado
        exec(code, globals(), local_vars)
        
        # Verificamos se o resultado foi definido corretamente
        if 'result' in local_vars:
            print("[Executor] Código executado com sucesso!\n")
            # Verificamos e processamos o resultado
            result = parser.parse(local_vars['result'], last_code_executed=code)
            return result
        else:
            raise ValueError("O código executado não definiu a variável 'result'")
            
    except NameError as e:
        # Verificamos se é um erro de falta da função execute_sql_query
        if "execute_sql_query" in str(e):
            print("[Executor] Erro: Função execute_sql_query não utilizada corretamente.")
            # Geramos um prompt de correção
            correction_prompt = get_correct_error_prompt_for_sql(state, code, str(e))
            correction_text = correction_prompt.to_string()
            
            # Obtemos código corrigido do modelo
            model = SophisticatedMockAI()
            corrected_code = model.generate_response(correction_text)
            
            print("[Executor] Tentando com código corrigido...\n")
            # Tentamos novamente com o código corrigido
            return execute_code_with_correction(corrected_code, dataframes, state)
        else:
            # Outro tipo de erro de nome
            return ErrorResponse(
                f"Erro de execução: {str(e)}",
                last_code_executed=code,
                error=str(e)
            )
            
    except Exception as e:
        # Qualquer outro erro
        print(f"[Executor] Erro geral: {str(e)}")
        return ErrorResponse(
            f"Erro de execução: {str(e)}",
            last_code_executed=code,
            error=str(e)
        )


# Exemplo de uso
if __name__ == "__main__":
    # Criando um dataframe de exemplo
    dados_df = pd.DataFrame({
        'id': range(1, 101),
        'categoria': ['A', 'B', 'C', 'D'] * 25,
        'valor': [i * 10 for i in range(1, 101)],
        'data': pd.date_range(start='2023-01-01', periods=100)
    })
    
    # Definindo o nome do dataframe para uso em SQL
    dados_df.name = "dados"
    
    # Criando objetos para processamento
    memoria = AgentMemory("Assistente de análise de dados")
    config = AgentConfig(direct_sql=True)
    
    # Lista de consultas para testar
    consultas = [
        "Qual é a média dos valores?",
        "Mostre um gráfico do total por categoria",
        "Qual o valor máximo?",
        "Qual o valor mínimo na categoria A?",
        "Conte quantos registros temos no total"
    ]
    
    # Processando cada consulta
    for i, consulta in enumerate(consultas):
        print(f"\n{'=' * 60}")
        print(f"CONSULTA {i+1}: {consulta}")
        print(f"{'=' * 60}\n")
        
        # Adicionando a consulta à memória
        memoria.add_message(consulta)
        
        # Criando estado
        state = AgentState(
            dfs=[dados_df],
            memory=memoria,
            config=config
        )
        
        # Gerando prompt para o modelo
        prompt = get_chat_prompt_for_sql(state)
        prompt_text = prompt.to_string()
        
        # Obtendo resposta do modelo
        model = SophisticatedMockAI()
        generated_code = model.generate_response(prompt_text)
        
        print(f"[Modelo] Código gerado:\n{generated_code}\n")
        
        # Executando o código com suporte a correção
        response = execute_code_with_correction(generated_code, [dados_df], state)
        
        # Exibindo o resultado
        print(f"[Resultado] Tipo: {response.type}")
        if response.type == "dataframe":
            print(f"Primeiras linhas:\n{response.value.head()}\n")
        elif response.type == "plot":
            print("Gráfico gerado (base64 truncado):", response.value[:50] + "...\n")
            # Em um aplicativo real, isso seria exibido ou salvo
        else:
            print(f"Valor: {response.value}\n")
        
        # Última consulta propositalmente com erro para demonstrar correção
        if i == len(consultas) - 1:
            print("\n" + "=" * 60)
            print("CONSULTA COM ERRO PROPOSITAL (para demonstrar correção)")
            print("=" * 60 + "\n")
            
            consulta_com_erro = "Mostre os dados agrupados por categoria"
            memoria.add_message(consulta_com_erro)
            
            state = AgentState(
                dfs=[dados_df],
                memory=memoria,
                config=config
            )
            
            prompt = get_chat_prompt_for_sql(state)
            prompt_text = prompt.to_string()
            
            # Gerando código com erro proposital (sem usar execute_sql_query)
            error_code = """
import pandas as pd

# Agrupando dados (com erro proposital: não usando execute_sql_query)
df_result = pd.DataFrame({'categoria': ['A', 'B', 'C'], 'total': [100, 200, 300]})

# Definindo resultado
result = {
    "type": "dataframe",
    "value": df_result
}
"""
    
    print(f"[Modelo] Código com erro proposital:\n{error_code}\n")
    
    # Executando com correção
    response = execute_code_with_correction(error_code, [dados_df], state)
    
    print(f"[Resultado após correção] Tipo: {response.type}")
    if response.type == "dataframe":
        print(f"Primeiras linhas:\n{response.value.head()}\n")
    else:
        print(f"Valor: {response.value}\n")