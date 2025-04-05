# Sistema de Consulta em Linguagem Natural

## Visão Geral

O Sistema de Consulta em Linguagem Natural é uma solução avançada que permite a análise de dados através de consultas em linguagem natural, eliminando a necessidade de conhecimentos técnicos em SQL ou programação.

### Principais Características

- **Processamento de Linguagem Natural**: Converte consultas em linguagem natural em código Python/SQL executável
- **Suporte a Múltiplas Fontes de Dados**: Integração com CSV, bancos de dados, e outras fontes
- **Visualizações Automáticas**: Geração de gráficos e visualizações de dados
- **Integração com Modelos de IA**: Utiliza modelos de linguagem para geração de código

## Arquitetura do Sistema

O sistema é composto por vários componentes modulares:

### 1. Conectores de Dados (`connector/`)
- Suporta diferentes fontes de dados
- Implementa camada semântica para interpretação de dados
- Tipos de conectores:
  - CSV
  - PostgreSQL
  - DuckDB
  - Outros bancos de dados

### 2. Motor de Consulta (`natural_query_engine.py`)
- Processamento central de consultas
- Gerenciamento de estado e memória
- Execução segura de código
- Geração de respostas

### 3. Integração com Modelos de Linguagem (`llm_integration.py`)
- Suporte a múltiplos modelos de IA:
  - OpenAI (GPT)
  - Anthropic (Claude)
  - Hugging Face
  - Modelos locais

### 4. Construção de Queries (`query_builders/`)
- Geração dinâmica de consultas SQL
- Transformações semânticas
- Otimização de queries

### 5. API REST (`api.py`)
- Endpoints para consultas
- Upload de dados
- Gestão de fontes de dados

## Instalação

### Pré-requisitos
- Python 3.7+
- Dependências listadas em `requirements.txt`

### Passos de Instalação

1. Clone o repositório
```bash
git clone https://github.com/seu-usuario/sistema-consulta-linguagem-natural.git
cd sistema-consulta-linguagem-natural
```

2. Crie um ambiente virtual
```bash
python -m venv venv
source venv/bin/activate  # No Windows use `venv\Scripts\activate`
```

3. Instale as dependências
```bash
pip install -r requirements.txt
```

## Configuração

### Configuração de Fontes de Dados

Crie um arquivo `datasources.json`:

```json
{
  "data_sources": [
    {
      "id": "vendas",
      "type": "csv",
      "path": "dados/vendas.csv",
      "delimiter": ",",
      "encoding": "utf-8"
    }
  ]
}
```

### Configuração de Modelos de Linguagem

Configure em `llm_config.json` ou através de variáveis de ambiente:

```json
{
  "model_type": "openai",
  "model_name": "gpt-3.5-turbo",
  "api_key": "sua_chave_api"
}
```

## Uso

### Interface de Linha de Comando

```bash
# Inicia o sistema
python integrated_system.py

# Executa uma consulta específica
python integrated_system.py --query "Qual é o total de vendas por cliente?"
```

### Exemplo de Código Python

```python
from integrated_system import NaturalLanguageAnalyticSystem

# Inicializa o sistema
system = NaturalLanguageAnalyticSystem()

# Processa uma consulta
resultado, tipo = system.process_query("Mostre o total de vendas por cidade")

# Exibe o resultado
print(resultado)
```

### API REST

```bash
# Inicia o servidor API
python integrated_system.py --api
```

Acesse a documentação em `http://localhost:8000/docs`

## Tipos de Consultas Suportadas

- Consultas básicas (`SELECT`)
- Agregações (`SUM`, `AVG`, `COUNT`)
- Agrupamentos (`GROUP BY`)
- Visualizações (gráficos de barras, linhas, etc.)
- Análises temporais
- Junções entre tabelas

## Exemplos de Consultas

- "Quantos clientes temos por cidade?"
- "Mostre o total de vendas por mês"
- "Crie um gráfico de barras com vendas por cliente"
- "Qual é o impacto financeiro das vendas perdidas?"

## Testes

Execute os testes usando:

```bash
python -m testes.run_all_tests --all
```

## Segurança

- Execução de código em ambiente isolado
- Sanitização de queries
- Tratamento de erros
- Prevenção de injeção de código

## Extensibilidade

- Adicione novos conectores de dados
- Integre novos modelos de linguagem
- Personalize transformações de dados

## Limitações

- Desempenho depende do modelo de linguagem
- Consultas muito complexas podem exigir ajustes
- Qualidade das respostas varia com a qualidade dos dados

## Fluxo Alternativo para Falhas da LLM

O sistema implementa um fluxo robusto para lidar com falhas na geração ou execução de consultas:

### Detecção de Entidades Inexistentes
- Antes de chamar a LLM, o sistema verifica se a consulta menciona entidades que não existem nos dados
- Oferece resposta explicativa com alternativas baseadas nos dados disponíveis

### Reformulação Automática
- Quando uma consulta falha, o sistema tenta reformulá-la automaticamente
- Adapta conceitos não mapeados para equivalentes disponíveis nos dados
- Suporta até 3 tentativas de reformulação antes de oferecer alternativas

### Coleta de Feedback do Usuário
- Permite que o usuário forneça feedback para melhorar a resposta
- Armazena o feedback para análise e melhorias futuras
- Usa o feedback para refinar a consulta em tempo real

### Sugestões Predefinidas
- Após múltiplas falhas, oferece sugestões de consultas alternativas
- Gera opções baseadas nos datasets e metadados disponíveis
- Ajuda o usuário a explorar os dados de maneira eficaz

Para testar o fluxo alternativo, execute:
```bash
python fallback_flow_example.py
```

## Contribuição

1. Faça um fork do repositório
2. Crie uma branch para sua feature
3. Commit suas mudanças
4. Crie um Pull Request

## Suporte

- Abra issues no GitHub
- Consulte a documentação
- Entre em contato com o mantenedor

## Refatoração da Arquitetura (Abril/2025)

Recentemente realizamos uma refatoração substancial do código para melhorar a manutenibilidade e extensibilidade do sistema. As principais melhorias incluem:

### 1. Aplicação do Princípio de Responsabilidade Única (SRP)
- O arquivo monolítico `core_integration.py` foi dividido em módulos menores e especializados
- Cada classe agora tem uma única responsabilidade bem definida
- Código mais organizado e fácil de entender

### 2. Nova Arquitetura Modular
- **Dataset**: Gerenciamento de dados e metadados (`core/engine/dataset.py`)
- **SQLExecutor**: Execução de consultas SQL (`core/engine/sql_executor.py`)
- **AlternativeFlow**: Tratamento de erros e fluxos alternativos (`core/engine/alternative_flow.py`)
- **FeedbackManager**: Gestão de feedback e sugestões (`core/engine/feedback_manager.py`)
- **AnalysisEngine**: Orquestração dos componentes (`core/engine/analysis_engine.py`)

### 3. Interface Simplificada
- Nova interface de alto nível `NaturalLanguageQuerySystem` para uso facilitado
- Exemplo de uso em `example_natural_language_query.py`
- Documentação atualizada em `core/engine/README.md`

### 4. Benefícios da Refatoração
- Maior testabilidade dos componentes individuais
- Facilitação da extensão de funcionalidades
- Melhor organização de dependências
- Código mais legível e autodocumentado

Para mais detalhes sobre a nova arquitetura, consulte a documentação em `core/engine/README.md`.

## Próximos Passos

- Melhorar a precisão dos modelos de linguagem
- Adicionar mais tipos de visualizações
- Expandir suporte a fontes de dados
- Implementar cache de consultas
- Desenvolver testes unitários para os novos componentes
- Adicionar novas funcionalidades usando a arquitetura modular

---

**Desenvolvido com ❤️ por [Paulo Henrique Vieira]**