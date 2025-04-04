from core.code_executor import AdvancedDynamicCodeExecutor

def main():
    executor = AdvancedDynamicCodeExecutor()
    
    # Exemplos de código para execução
    test_cases = [
        # Caso 1: Cálculo simples
        """
def calcular_media(numeros):
    return sum(numeros) / len(numeros)

result = calcular_media([1, 2, 3, 4, 5])
print(f"Média: {result}")
""",
        
        # Caso 2: Usando NumPy
        """
import numpy as np

def calcular_estatisticas(numeros):
    return {
        'media': np.mean(numeros),
        'mediana': np.median(numeros),
        'desvio_padrao': np.std(numeros)
    }

result = calcular_estatisticas([1, 2, 3, 4, 5])
print("Estatísticas:", result)
""",
        
        # Caso 3: DataFrame
        """
import pandas as pd

def processar_dados():
    dados = {
        'nome': ['Alice', 'Bob', 'Charlie'],
        'idade': [25, 30, 35],
        'salario': [5000, 6000, 7000]
    }
    df = pd.DataFrame(dados)
    return df[df['salario'] > 5500]

result = processar_dados()
print(result)
"""
    ]
    
    # Executa cada caso de teste
    for i, test_code in enumerate(test_cases, 1):
        print(f"\n--- Caso de Teste {i} ---")
        
        print("Código de Teste:")
        print(test_code)
        
        resultado = executor.execute_code(test_code)
        
        print("\nResultado da Execução:")
        print(f"Sucesso: {resultado['success']}")
        print(f"Saída: {resultado['output']}")
        print(f"Erro: {resultado['error']}")
        print(f"Resultado: {resultado['result']}")
        print(f"Tipo de Saída: {resultado['output_type']}")
        
if __name__ == "__main__":
    main()