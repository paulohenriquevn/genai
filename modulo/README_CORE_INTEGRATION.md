# Integração do Módulo Core no Sistema de Análise de Dados

Este documento descreve a implementação da integração completa do módulo `core` no sistema de análise de dados, conforme implementado nos arquivos `core_integration.py` e `core_integration_example.py`.

## Visão Geral

A integração do módulo core permite o processamento avançado de consultas em linguagem natural, execução segura de código, gerenciamento de estado e respostas estruturadas. Esta implementação conecta os componentes principais do sistema, formando um motor de análise completo.

## Componentes Integrados

### 1. Code Executor (`core/code_executor.py`)
- Implementação de `AdvancedDynamicCodeExecutor` para execução segura de código gerado
- Sanitização e validação de código para segurança
- Avaliação controlada em ambiente isolado
- Tratamento de exceções e erros de execução

### 2. Gerenciamento de Estado (`core/agent/state.py`)
- Implementação do `AgentState` para manter contexto entre consultas
- `AgentMemory` para armazenar histórico de conversas
- `AgentConfig` para configuração de comportamento

### 3. Geração de Prompts (`core/prompts`)
- Utilização do `GeneratePythonCodeWithSQLPrompt` para gerar código baseado em consultas
- Templates renderizados com metadados dos datasets
- Customização de instruções baseado no contexto

### 4. Tratamento de Respostas (`core/response`)
- Utilização das classes específicas de resposta:
  * `DataFrameResponse`: Para tabelas e conjuntos de dados
  * `NumberResponse`: Para valores numéricos
  * `StringResponse`: Para respostas textuais
  * `ChartResponse`: Para visualizações
  * `ErrorResponse`: Para tratamento de erros

### 5. Parser de Respostas (`core/response/parser.py`)
- Conversão e validação de resultados
- Tipagem segura de respostas
- Tratamento de diferentes formatos de saída

## Classe Principal: AnalysisEngine

A classe `AnalysisEngine` serve como o ponto central de integração, oferecendo uma API completa para:

1. **Carregamento de Dados**
   - Importação de arquivos CSV, Excel, JSON e Parquet
   - Metadados e descrições para datasets
   - Gestão de múltiplos conjuntos de dados

2. **Processamento de Consultas**
   - Interpretação de consultas em linguagem natural
   - Geração de código Python e SQL
   - Execução segura e monitorada

3. **Visualização de Dados**
   - Geração de gráficos baseados nos resultados
   - Múltiplos tipos de visualização
   - Personalização de parâmetros visuais

4. **Tratamento de Erros**
   - Captura e tratamento de exceções
   - Respostas de erro estruturadas
   - Logging detalhado para depuração

## Como Utilizar

### Instalação de Dependências

```bash
pip install pandas numpy matplotlib jinja2 black
```

### Exemplos de Uso Básico

```python
from core_integration import AnalysisEngine

# Inicializa o motor de análise
engine = AnalysisEngine()

# Carrega um dataset
engine.load_data("dados/vendas.csv", "vendas", "Dados de vendas")

# Processa uma consulta em linguagem natural
result = engine.process_query("Quais são os 5 produtos mais vendidos?")

# Exibe o resultado
print(f"Tipo: {result.type}")
print(result.value)
```

### Exemplo Completo

Execute o script de exemplo para ver a integração em ação:

```bash
python core_integration_example.py
```

## Características Avançadas

### Segurança

O sistema implementa várias camadas de segurança:

- Sanitização de consultas e código
- Limitação de módulos importáveis
- Validação de código antes da execução
- Tempo limite para execução
- Prevenção de comandos perigosos

### Extensibilidade

A arquitetura foi projetada para ser extensível:

- Facilidade para adicionar novos tipos de resposta
- Suporte a múltiplos formatos de dados
- Possibilidade de integração com diferentes LLMs
- Expansão de capacidades de visualização

### Logging e Depuração

Sistema completo de logging para rastreamento de operações:

- Registro de consultas processadas
- Detalhes de erros de execução
- Informações de tempo de processamento
- Armazenamento de código gerado

## Integração com Modelos de Linguagem (LLM)

O sistema agora inclui integração completa com vários provedores de LLM para geração de código Python:

1. **Provedores de LLM Suportados**
   - **OpenAI**: GPT-3.5-Turbo e GPT-4
   - **Anthropic**: Claude (todos os modelos disponíveis)
   - **Hugging Face**: Modelos hospedados
   - **Modelos locais**: LLama, Mistral e outros
   - **Mock**: Modo simulado para testes sem API

2. **Inicialização de Modelo**
   ```python
   # Inicialização com modelo OpenAI
   engine = AnalysisEngine(
       model_type="openai",
       model_name="gpt-4",
       api_key="sua-chave-api"
   )
   
   # Inicialização com modelo Anthropic
   engine = AnalysisEngine(
       model_type="anthropic",
       model_name="claude-3-haiku-20240307",
       api_key="sua-chave-api"
   )
   
   # Modo simulado (para testes)
   engine = AnalysisEngine(model_type="mock")
   ```

3. **Geração de Prompts Avançada**
   - Prompts enriquecidos com detalhes dos datasets disponíveis
   - Incluem exemplos de dados para melhorar a precisão
   - Instruções específicas para geração de código Python executável

4. **Tratamento Automático de Erros**
   - Sistema de correção automática de erros com feedback detalhado
   - Tentativa inteligente de correção antes de reportar falha
   - Mensagens de erro contextualizadas

5. **Execução SQL Robusta**
   - Suporte para DuckDB para consultas SQL avançadas
   - Fallback para pandas se DuckDB não estiver disponível
   - Logging de consultas SQL para auditoria

2. **Execução SQL**
   - O método `execute_direct_query` é limitado ao pandas
   - Para SQL mais robusto, considere integrar DuckDB ou SQLite

3. **Gerenciamento de Memória**
   - Para datasets grandes, considere implementar mecanismos de paginação
   - Monitore o uso de memória em operações intensivas

## Próximos Passos Sugeridos

1. Integração com um LLM real (OpenAI, HuggingFace, etc.)
2. Implementação de cache de resultados para consultas frequentes
3. Suporte a bancos de dados externos (MySQL, PostgreSQL)
4. Interface web para visualização interativa
5. Testes unitários e de integração abrangentes