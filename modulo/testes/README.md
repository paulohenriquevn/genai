# Testes do Sistema de Consulta em Linguagem Natural

Este diretório contém a estrutura completa de testes para o Sistema de Consulta em Linguagem Natural, permitindo verificar a funcionalidade, robustez e desempenho de todos os componentes.

## Visão Geral

A arquitetura de testes deste sistema foi projetada para garantir:

1. **Funcionalidade** - Validação de que todos os componentes funcionam corretamente
2. **Robustez** - Verificação de como o sistema lida com entradas inesperadas ou inválidas
3. **Integração** - Teste da interação entre os diferentes componentes
4. **Performance** - Monitoramento do desempenho do sistema em diferentes cenários
5. **Qualidade** - Garantia da qualidade das respostas e visualizações geradas

## Estrutura de Testes

```
testes/
├── __init__.py                       # Inicialização do pacote de testes
├── README.md                         # Esta documentação
├── run_all_tests.py                  # Script para executar todos os testes
├── test_api_service.py               # Testes da API REST
├── test_llm_integration.py           # Testes da integração com LLMs
├── test_natural_query_engine.py      # Testes do motor principal
├── test_system_integration.py        # Testes do sistema integrado
├── test_scenarios/                   # Cenários de teste complexos
│   ├── __init__.py                   # Inicialização do pacote de cenários
│   ├── scenario1_basic_queries.py    # Cenário 1: Consultas básicas
│   ├── scenario2_data_analysis.py    # Cenário 2: Análise de dados avançada
│   └── scenario3_error_handling.py   # Cenário 3: Tratamento de erros
└── output/                           # Diretório para resultados dos testes
    ├── scenario1/                    # Resultados do cenário 1
    ├── scenario2/                    # Resultados do cenário 2
    ├── scenario3/                    # Resultados do cenário 3
    └── test_results.json             # Resultados consolidados de todos os testes
```

## Tipos de Testes

### Testes Unitários

Verificam componentes individuais do sistema:

#### `test_natural_query_engine.py`

Testa as funcionalidades principais do motor de consulta em linguagem natural, incluindo:
- Inicialização e carregamento de dados
- Execução de consultas SQL diretas
- Processamento de consultas em linguagem natural
- Agrupamentos e agregações
- Geração de visualizações
- Tratamento de erros

Exemplos de casos de teste:
- `test_engine_initialization`: Verifica se o motor carrega corretamente os dataframes
- `test_basic_sql_query`: Testa a execução direta de SQL e JOINs
- `test_visualization_query`: Verifica a geração de gráficos a partir de consultas

#### `test_llm_integration.py`

Testa a integração com modelos de linguagem para geração de código a partir de linguagem natural:
- Inicialização de diferentes tipos de modelos (OpenAI, Anthropic, Hugging Face, etc.)
- Geração de código para consultas simples e complexas
- Tratamento de erros e fallbacks
- Modelos mock para testes isolados

Exemplos de casos de teste:
- `test_mock_integration`: Testa a geração de código com um modelo mock
- `test_code_cleaning`: Verifica se o código gerado é adequadamente limpo
- `test_error_recovery`: Testa a recuperação de falhas na geração de código

#### `test_api_service.py`

Testa a API REST do sistema, incluindo:
- Endpoints de consulta
- Upload e gerenciamento de dados
- Execução de SQL direta
- Visualizações
- Tratamento de erros

Exemplos de casos de teste:
- `test_query_endpoint_basic`: Testa o processamento de consultas básicas via API
- `test_upload_data`: Verifica o upload e registro de novas fontes de dados
- `test_query_endpoint_error`: Testa o tratamento de erros na API

### Testes de Integração

Verificam a interação entre componentes:

#### `test_system_integration.py`

Testa o sistema como um todo, incluindo:
- Fluxo completo de carregamento de dados, consulta e resposta
- Interação entre componentes
- Processamento end-to-end

Exemplos de casos de teste:
- `test_full_system_basic_query`: Testa o fluxo completo com uma consulta básica
- `test_error_propagation`: Verifica como os erros são propagados entre componentes
- `test_complex_query_flow`: Testa consultas complexas no fluxo completo

### Cenários de Teste

Simulam casos de uso reais com fluxos completos:

#### `scenario1_basic_queries.py`

Executa uma série de consultas básicas para validar as operações fundamentais:
- SELECTs simples
- Filtros com WHERE
- Contagens e totalizações
- Médias e agregações
- Agrupamentos simples
- Visualizações básicas

Exemplos de consultas testadas:
- "Mostre as primeiras 5 linhas da tabela de vendas"
- "Quantos registros existem na tabela de vendas?"
- "Crie um gráfico de barras mostrando o total de vendas por cliente"

#### `scenario2_data_analysis.py`

Executa análises de dados mais avançadas:
- Análises temporais
- Correlações entre variáveis
- Análises multi-tabela com JOINs
- Visualizações compostas
- Análises preditivas simples

Exemplos de consultas testadas:
- "Como as vendas evoluíram ao longo do tempo? Mostre um gráfico de linha"
- "Qual a correlação entre o valor da venda e o segmento do cliente?"
- "Faça uma análise preditiva simples para estimar vendas futuras"

#### `scenario3_error_handling.py`

Testa a robustez do sistema com casos de erro:
- Consultas com sintaxe inválida
- Consultas ambíguas
- Nomes de tabelas ou colunas inexistentes
- Tipos de dados incompatíveis
- Falhas de conexão com modelos de linguagem

Exemplos de consultas testadas:
- "Mostre dados da tabela que não existe"
- "Execute um cálculo complexo com dados incompatíveis"
- "Consulta com instrução SQL inválida"

## Executando os Testes

### Executar Todos os Testes

```bash
python -m testes.run_all_tests --all
```

### Executar Apenas Testes Unitários

```bash
python -m testes.run_all_tests --unit
```

### Executar Apenas Cenários de Teste

```bash
python -m testes.run_all_tests --scenarios
```

### Executar Cenários Específicos

```bash
python -m testes.run_all_tests --scenarios --scenario scenario1 scenario3
```

### Modo Verboso

Para ver detalhes adicionais durante a execução dos testes:

```bash
python -m testes.run_all_tests --all --verbose
```

## Configuração de Ambiente de Teste

Os testes unitários criam ambientes temporários isolados para executar cada teste, garantindo que:
1. Os testes sejam independentes
2. Não interfiram com dados de produção
3. Possam ser executados em paralelo
4. Limpem todos os recursos ao finalizar

Por exemplo, em `test_natural_query_engine.py`, os métodos `setUpClass` e `tearDownClass` gerenciam um diretório temporário com dados de teste, que é criado antes dos testes e removido após a execução.

## Resultados dos Testes

Os resultados da execução dos testes são salvos no diretório `testes/output/`:

- **test_results.json**: Resultados consolidados de todos os testes, incluindo:
  - Contagem total de testes executados
  - Número de testes bem-sucedidos/falhos
  - Taxa de sucesso
  - Tempo de execução
  - Timestamp

- **scenario1/**, **scenario2/**, **scenario3/**: Resultados detalhados de cada cenário, incluindo:
  - CSVs com resultados de consultas
  - Visualizações geradas (PNG)
  - Relatórios JSON de execução
  - Para o cenário 2, um relatório HTML interativo

## Ampliando os Testes

### Adicionar um Novo Teste Unitário

1. Crie um arquivo `test_nome_do_componente.py` no diretório `testes/`
2. Siga o padrão de testes existentes, utilizando a biblioteca `unittest`
3. Faça o teste se registrar automaticamente pelo padrão de nome

Exemplo de estrutura básica:
```python
import unittest
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importe o componente a ser testado
from componente_a_testar import Componente

class TestComponente(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Configuração inicial para todos os testes
        pass
        
    def test_funcionalidade_a(self):
        # Teste específico
        self.assertTrue(condicao)
        
    def test_funcionalidade_b(self):
        # Outro teste
        self.assertEqual(valor_esperado, valor_obtido)
        
if __name__ == '__main__':
    unittest.main()
```

### Adicionar um Novo Cenário de Teste

1. Crie um arquivo `scenario[N]_descricao.py` no diretório `testes/test_scenarios/`
2. Implemente a função `execute_scenario()` que será chamada pelo executor de testes
3. Atualize `run_all_tests.py` para incluir o novo cenário na lista de cenários disponíveis

Exemplo:
```python
def execute_scenario():
    """Executa o novo cenário de teste"""
    print("Iniciando Novo Cenário: Descrição")
    
    # Inicializa o motor de consulta
    engine = NaturalLanguageQueryEngine()
    
    # Lista de consultas para teste
    queries = [
        {"name": "consulta1", "query": "Descrição da consulta 1", "description": "O que testa"},
        {"name": "consulta2", "query": "Descrição da consulta 2", "description": "O que testa"}
    ]
    
    # Executa cada consulta e registra os resultados
    results = []
    for query_info in queries:
        # [Lógica de execução do teste]
        # [Coleta de resultados]
        
    return results

if __name__ == "__main__":
    execute_scenario()
```

## Melhores Práticas

1. **Isolamento**: Cada teste deve ser independente e não deixar "rastros"
2. **Abrangência**: Teste casos normais e de borda
3. **Mocks**: Use objetos mock para simular componentes externos como APIs de LLM
4. **Documentação**: Documente claramente o propósito de cada teste
5. **Verificações**: Verifique não apenas se algo funciona, mas se funciona corretamente

## Depuração de Falhas

Se um teste falhar:
1. Verifique os logs detalhados em `testes/output/`
2. Execute o teste específico com `--verbose` para mais detalhes
3. Use a função de depuração do Python: `python -m pdb -m testes.teste_especifico`

## Requisitos

Para executar os testes, são necessárias as seguintes dependências:

- Python 3.7+
- pandas
- numpy
- matplotlib
- fastapi (para testes da API)
- httpx (para testes de cliente HTTP)
- pytest (opcional, para testes mais avançados)
- unittest (biblioteca padrão Python)

Instale as dependências via pip:

```bash
pip install -r requirements.txt
```