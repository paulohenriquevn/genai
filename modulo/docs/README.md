# Documentação do Sistema de Consulta em Linguagem Natural

Este diretório contém a documentação completa do Sistema de Consulta em Linguagem Natural, uma plataforma para análise de dados usando linguagem natural.

## Índice de Documentação

### Funcionalidades Principais

- [Visão Geral do Sistema](overview.md) (Visão geral da arquitetura e componentes)
- [Processamento de Linguagem Natural](nlp_processing.md) (Como o sistema interpreta consultas em linguagem natural)
- [Geração de Código com LLM](code_generation.md) (Como o sistema gera código Python/SQL)
- [Conectores de Dados](data_connectors.md) (Como conectar diferentes fontes de dados)

### Visualizações

- [ApexCharts Integration](apex_charts_integration.md) (Como usar visualizações interativas com ApexCharts)
- [Formatos de Gráficos Suportados](chart_formats.md) (Opções de visualização disponíveis)

### Desenvolvimento e Extensão

- [Guia de Desenvolvimento](development_guide.md) (Como contribuir e estender o sistema)
- [API Reference](api_reference.md) (Documentação completa da API)
- [Fluxo de Trabalho com LLMs](llm_workflow.md) (Como trabalhar com modelos de linguagem)
- [Customização de Prompts](prompt_customization.md) (Como personalizar prompts para os LLMs)

### Operações e Implantação

- [Guia de Implantação](deployment_guide.md) (Como implantar o sistema em produção)
- [Segurança](security.md) (Práticas de segurança e proteções)
- [Gerenciamento de Recursos](resource_management.md) (Como gerenciar recursos computacionais)

## Guia Rápido de Início

### Instalação

```bash
# Clone o repositório
git clone https://github.com/seu-repositorio/sistema-consulta-linguagem-natural.git
cd sistema-consulta-linguagem-natural

# Crie um ambiente virtual
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate

# Instale as dependências
pip install -r requirements.txt
```

### Uso Básico

```python
from core.engine.analysis_engine import AnalysisEngine

# Inicializa o motor de análise
engine = AnalysisEngine(model_type="openai", model_name="gpt-4", api_key="sua-api-key")

# Carrega dados
engine.load_data("dados/vendas.csv", "vendas")
engine.load_data("dados/clientes.csv", "clientes")

# Processa uma consulta em linguagem natural
resposta = engine.process_query("Mostre as 5 maiores vendas do último mês")

# Exibe o resultado
print(resposta.get_value())
```

### Uso da Interface de Alto Nível

```python
from natural_language_query_system import NaturalLanguageQuerySystem

# Inicializa o sistema
nlq = NaturalLanguageQuerySystem()

# Carrega dados
nlq.load_data("dados/vendas.csv", "vendas")
nlq.load_data("dados/clientes.csv", "clientes")

# Faz uma consulta
resultado = nlq.ask("Quais são os principais produtos por volume de vendas?")

# Exibe o resultado
print(resultado)
```

## Novidades e Atualizações

### Versão Atual (Abril/2025)

- **ApexCharts Integration**: Suporte para visualizações interativas com ApexCharts
- **Refatoração do Core Integration**: Código reorganizado seguindo o princípio de responsabilidade única
- **Timeout Execution**: Implementação robusta de timeout para execução de código
- **Interface de Alto Nível**: Nova interface simplificada para uso do sistema