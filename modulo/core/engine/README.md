# Motor de Análise de Dados em Linguagem Natural

Este diretório contém a implementação refatorada do motor de análise de dados em linguagem natural.
A refatoração aplicou o princípio de responsabilidade única (SRP), dividindo a implementação original
em componentes modulares especializados.

## Organização dos Componentes

O motor foi dividido nos seguintes componentes:

1. **Dataset** (`dataset.py`): 
   - Representa um dataset com metadados e descrição
   - Realiza análise automática de estrutura e relacionamentos
   - Fornece serialização para uso em prompts

2. **SQLExecutor** (`sql_executor.py`):
   - Gerencia a execução de consultas SQL em datasets
   - Adapta SQL para compatibilidade com diferentes dialetos
   - Registra funções personalizadas para maior compatibilidade

3. **AlternativeFlow** (`alternative_flow.py`):
   - Implementa fluxos alternativos para falhas na execução
   - Reformula consultas com base em erros encontrados
   - Gera sugestões de consultas alternativas

4. **FeedbackManager** (`feedback_manager.py`):
   - Gerencia o feedback do usuário para consultas
   - Armazena consultas bem-sucedidas para uso futuro
   - Usa feedback para otimizar novas consultas

5. **AnalysisEngine** (`analysis_engine.py`):
   - Orquestra todos os componentes do sistema
   - Implementa interface principal para processamento de consultas
   - Gerencia ciclo de vida de consultas e respostas

## Interface de Alto Nível

Para simplificar o uso, foi criada uma interface de alto nível no arquivo 
`natural_language_query_system.py` no diretório raiz.

## Exemplo de Uso

```python
from natural_language_query_system import NaturalLanguageQuerySystem

# Inicializa o sistema
nlq = NaturalLanguageQuerySystem()

# Carrega datasets
nlq.load_data("vendas.csv", "vendas")
nlq.load_data("clientes.csv", "clientes")

# Executa consulta em linguagem natural
resultado = nlq.ask("Quais são os 5 clientes com maior volume de compras?")

# Exibe o resultado
if resultado.type == "dataframe":
    print(resultado.get_value())
elif resultado.type == "plot":
    print(f"Gráfico gerado em: {resultado.get_value()}")
else:
    print(resultado.get_value())

# Executa consulta com feedback
resultado = nlq.ask_with_feedback(
    "Qual a média de vendas por mês?",
    "Mostre em formato de gráfico de barras"
)
```

## Arquitetura da Refatoração

A refatoração aplicou os seguintes princípios de design:

1. **Responsabilidade Única**: Cada classe tem uma única responsabilidade
2. **Aberto/Fechado**: Os componentes são extensíveis sem modificação
3. **Interface Segregada**: Interfaces específicas para cada componente
4. **Injeção de Dependência**: Componentes recebem suas dependências
5. **Fachada (Facade)**: Interface simplificada para o sistema complexo

O AnalysisEngine age como uma fachada, orquestrando os componentes sem implementar diretamente toda a lógica.

## Benefícios da Refatoração

- **Manutenibilidade**: Código mais organizado e focado
- **Testabilidade**: Componentes podem ser testados isoladamente
- **Extensibilidade**: Facilidade para adicionar novas funcionalidades
- **Legibilidade**: Código mais claro e autodocumentado
- **Robustez**: Melhor tratamento de erros e fluxos alternativos