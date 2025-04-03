# GenBI: Sistema Modular de Business Intelligence Generativa

Desenvolvi uma aplicação completa de Business Intelligence Generativa inspirada no WrenAI, com foco em modularidade, facilidade de uso e extensibilidade. Vamos explorar os principais componentes e funcionalidades.

## Arquitetura do Sistema

A aplicação GenBI foi projetada com uma arquitetura modular que separa claramente as responsabilidades, permitindo fácil manutenção e extensibilidade:

1. **API REST**: Backend com FastAPI para processamento de requisições
2. **Núcleo GenBI**: Componentes centrais para modelagem e execução de consultas
3. **Integração LLM**: Conectores para modelos de linguagem e processamento de linguagem natural
4. **Conectores de Dados**: Integração com diferentes fontes de dados

## Principais Funcionalidades

### 1. Modelagem de Dados Flexível (MDL)

O GenBI utiliza uma linguagem de modelagem de dados própria, inspirada no WrenAI, que permite definir:

- **Modelos**: Representações de tabelas ou consultas
- **Colunas**: Definições de campos com tipos e propriedades semânticas
- **Relacionamentos**: Conexões entre modelos (1:1, 1:N, N:1, N:N)
- **Métricas**: Definições para análises agregadas

O esquema MDL facilita que o LLM compreenda a estrutura dos dados e gere consultas SQL precisas.

### 2. Processamento de Linguagem Natural para SQL

O módulo NL-to-SQL é o coração da funcionalidade generativa:

- Recebe perguntas em linguagem natural
- Usa um LLM (como GPT-4, Claude ou Gemini) para interpretar a intenção
- Gera consultas SQL otimizadas e corretas
- Fornece explicações das consultas geradas

### 3. Motor de Execução de Consultas

Um executor robusto que:

- Executa consultas SQL em diferentes bancos de dados
- Implementa cache inteligente para melhorar performance
- Processa resultados em formato tabular
- Gera visualizações a partir dos dados

## Componentes Técnicos

- **API Server**: Endpoints REST para todas as operações
- **Data Connectors**: Integração com PostgreSQL, MySQL, Snowflake, etc.
- **LLM Integration**: Suporte para OpenAI, Anthropic e Google
- **Query Executor**: Processamento de consultas com cache

## Como Usar o GenBI

### 1. Modelagem de Dados

Primeiro, é necessário definir os modelos e relacionamentos usando a interface de modelagem de dados ou importando um esquema MDL existente.

### 2. Consultas em Linguagem Natural

Os usuários podem fazer perguntas como:
- "Quais são os 5 produtos mais vendidos no último mês?"
- "Qual é a média de vendas por região nos últimos 6 meses?"
- "Mostre-me o produto com maior margem de lucro em cada categoria"

### 3. Visualizações

Após receber os resultados, os usuários podem gerar visualizações como:
- Gráficos de barras para comparações
- Gráficos de linha para tendências temporais
- Gráficos de pizza para distribuições
- Gráficos de dispersão para correlações

## Vantagens do GenBI

1. **Acessibilidade**: Usuários não-técnicos podem obter insights sem conhecer SQL
2. **Modularidade**: Arquitetura que permite substituir componentes como o provedor LLM
3. **Extensibilidade**: Facilidade para adicionar novos conectores de dados ou visualizações
4. **Performance**: Sistema de cache que reduz tempo de resposta e custos de API
---

O GenBI é uma plataforma completa que democratiza o acesso a dados, permitindo que qualquer pessoa na organização faça perguntas complexas usando linguagem natural, transformando a maneira como as empresas extraem valor de seus dados.