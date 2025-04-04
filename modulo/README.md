# Sistema de Consulta em Linguagem Natural

## Visão Geral

O Sistema de Consulta em Linguagem Natural é uma solução avançada que permite a análise de dados através de consultas em linguagem natural, eliminando a necessidade de conhecimentos técnicos em SQL ou programação.

### Principais Características

- **Processamento de Linguagem Natural**: Converte consultas em linguagem natural em código Python/SQL executável
- **Suporte a Múltiplas Fontes de Dados**: Integração com CSV, bancos de dados, e outras fontes
- **Visualizações Automáticas**: Geração de gráficos e visualizações de dados
- **Integração com Modelos de IA**: Utiliza modelos de linguagem para geração de código
- **API REST**: Interface para integração com outras aplicações

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

## Contribuição

1. Faça um fork do repositório
2. Crie uma branch para sua feature
3. Commit suas mudanças
4. Crie um Pull Request

## Suporte

- Abra issues no GitHub
- Consulte a documentação
- Entre em contato com o mantenedor

## Próximos Passos

- Melhorar a precisão dos modelos de linguagem
- Adicionar mais tipos de visualizações
- Expandir suporte a fontes de dados
- Implementar cache de consultas

---

**Desenvolvido com ❤️ por [Paulo Henrique Vieira]**