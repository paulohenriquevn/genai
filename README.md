# GenBI: Sistema Modular de Business Intelligence Generativa

## Visão Geral

GenBI é uma aplicação de Business Intelligence Generativa projetada para processar perguntas em linguagem natural, convertê-las em consultas SQL e gerar visualizações de dados automaticamente.

## Principais Características

- **Processamento de Consultas em Linguagem Natural**: Converte perguntas em linguagem natural para consultas SQL precisas
- **Geração Automática de Visualizações**: Cria gráficos e tabelas interativas baseadas nos resultados das consultas
- **API RESTful**: Interface para integração com outros sistemas
- **Modo Interativo**: Interface de linha de comando para consultas ad-hoc
- **Sistema de Cache**: Armazena resultados de consultas para maior performance
- **Segurança Integrada**: Validação de entradas, rate limiting e proteção contra injeção SQL

## Arquitetura

### Componentes Principais

1. **API REST (app/api_server.py)**: Interface para interação com o sistema
2. **Processador NL-to-SQL (app/llm_integration/nl_processor.py)**: Converte linguagem natural em consultas SQL
3. **Executor de Consultas (app/query_executor/query_executor.py)**: Processa consultas SQL
4. **Gerador de Visualizações (app/query_executor/query_executor.py)**: Cria visualizações interativas
5. **Conector de Dados (app/data_connector/data_connector.py)**: Gerencia conexões com bancos de dados

## Requisitos

- Python 3.8+
- OpenAI API Key
- SQLite (banco de dados padrão, outros podem ser configurados)
- Bibliotecas Python: fastapi, uvicorn, openai, pandas, plotly

## Instalação e Configuração

1. Clone o repositório:
   ```bash
   git clone <url-do-repositorio>
   cd genbi
   ```

2. Crie e ative um ambiente virtual:
   ```bash
   python -m venv venv
   source venv/bin/activate  # No Windows: venv\Scripts\activate
   ```

3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure a API key da OpenAI:
   ```bash
   export OPENAI_API_KEY="sua-api-key-aqui"
   ```

5. Configure o banco de dados de exemplo:
   ```bash
   python run.py --setup-db
   ```

## Uso

### Iniciar o Servidor API

```bash
python run.py --server
```

O servidor estará disponível em `http://localhost:8000`.

### Modo Interativo

```bash
python run.py --interactive
```

Este modo permite fazer perguntas em linguagem natural diretamente pelo terminal.

### Executar Testes

```bash
python run.py --test
```

## Exemplos de Uso

### Consultas em Linguagem Natural

Aqui estão exemplos de perguntas que o sistema pode responder:

- "Qual é o faturamento total por categoria de produto?"
- "Quais são os 5 produtos mais vendidos?"
- "Quanto vendemos no último mês?"
- "Quais clientes fizeram mais compras?"

### API REST

Endpoint para consultas em linguagem natural:
```http
POST /query/natural_language
{
    "question": "Quais são os 5 produtos mais vendidos?",
    "use_cache": true,
    "explain_sql": true
}
```

Endpoint para visualização:
```http
POST /visualization
{
    "query_id": "01234567-89ab-cdef-0123-456789abcdef",
    "type": "bar",
    "options": {
        "title": "Top 5 Produtos Vendidos",
        "x_column": "produto",
        "y_column": "quantidade_vendida"
    }
}
```

## Configurações

O sistema usa o arquivo `config/config.json` para configurações principais:

- **Conexão com Banco de Dados**: Tipo, credenciais e parâmetros de conexão
- **Configuração de LLM**: Provedor, modelo e parâmetros
- **Sistema de Cache**: TTL e diretório
- **Catálogo de Dados**: Definição de modelos e relações

## Variáveis de Ambiente

- `OPENAI_API_KEY`: Chave de API da OpenAI (obrigatória)
- `GENBI_CONFIG_PATH`: Caminho para o arquivo de configuração (opcional)
- `ALLOWED_ORIGINS`: Lista de origens permitidas para CORS (opcional)
- `RATE_LIMIT_PER_MINUTE`: Limite de requisições por minuto por IP (opcional)

## Segurança

- **API Key Segura**: Use variáveis de ambiente para a API key da OpenAI
- **Rate Limiting**: Proteção contra abuso da API
- **Validação SQL**: Bloqueia comandos SQL potencialmente perigosos
- **CORS Configurável**: Restrinja o acesso à API para domínios específicos

## Personalização

### Adicionar Novos Provedores de LLM

Implemente novas classes em `app/llm_integration/llm_client.py` seguindo o padrão da interface `LLMClient`.

### Conectar a Outros Bancos de Dados

Implemente novos conectores em `app/data_connector/data_connector.py` seguindo o padrão da interface `DataConnector`.

## Limitações Atuais

- Suporte apenas para banco de dados SQLite (outros bancos podem ser adicionados)
- Visualizações limitadas aos tipos: bar, line, pie, scatter e table
- Não suporta autenticação de usuários (deve ser implementada para ambientes de produção)

## Licença

MIT License

## Contribuidores

- [Seu Nome] - Desenvolvedor Principal