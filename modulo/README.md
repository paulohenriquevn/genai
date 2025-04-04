# Sistema de Consulta em Linguagem Natural

Este projeto implementa um sistema completo para análise de dados utilizando consultas em linguagem natural. O sistema permite que usuários sem conhecimentos técnicos em programação ou SQL possam realizar análises de dados complexas através de perguntas em português.

## Visão Geral

O sistema integra vários componentes para criar uma solução completa:

1. **Conectores de Dados** - Para carregar e processar dados de diferentes fontes
2. **Motor de Consulta** - Para processamento de consultas em linguagem natural
3. **Modelos de Linguagem** - Para gerar código Python/SQL a partir de consultas
4. **Execução Segura** - Para executar código gerado de forma segura e controlada
5. **API REST** - Para acesso remoto e integração com outras aplicações

O fluxo de trabalho é:
1. O usuário faz uma pergunta sobre os dados em linguagem natural
2. O sistema utiliza modelos de IA para converter a pergunta em código Python/SQL
3. O código é executado de forma segura
4. Os resultados são processados e retornados no formato apropriado (texto, número, tabela, gráfico)
5. Se houver erros, o sistema tenta corrigi-los automaticamente

## Requisitos

- Python 3.7 ou superior
- Dependências principais (veja em `requirements.txt`):
  - pandas
  - matplotlib
  - duckdb
  - numpy
  - fastapi
  - uvicorn
  - (opcional) openai, anthropic ou huggingface para integração com modelos de IA avançados

## Instalação

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/sistema-consulta-linguagem-natural.git
cd sistema-consulta-linguagem-natural

# Instale as dependências
pip install -r requirements.txt

# Para usar modelos de IA avançados (opcional)
pip install openai anthropic huggingface_hub
```

## Estrutura do Projeto

```
sistema-consulta-linguagem-natural/
├── connector/            # Conectores para diferentes fontes de dados
├── core/                 # Módulos core para execução, estado e respostas
├── query_builders/       # Construtores de consultas SQL otimizadas
├── utils/                # Utilitários para análise de datasets
├── dados/                # Arquivos de dados de exemplo
├── natural_query_engine.py  # Motor de consulta em linguagem natural
├── llm_integration.py    # Integração com modelos de linguagem
├── api_service.py        # API REST para acesso remoto
├── integrated_system.py  # Sistema integrado completo
├── example_usage.py      # Exemplos de uso
├── requirements.txt      # Dependências do projeto
└── README.md             # Este arquivo
```

## Uso Básico

### Interface de Linha de Comando

```bash
# Inicia a interface interativa
python integrated_system.py

# Executa uma consulta específica
python integrated_system.py --query "Qual é o total de vendas por cliente?"

# Inicia apenas o servidor API
python integrated_system.py --api
```

### Uso como Biblioteca

```python
from integrated_system import NaturalLanguageAnalyticSystem

# Inicializa o sistema
system = NaturalLanguageAnalyticSystem(
    data_dir="caminho/para/dados",
    output_dir="caminho/para/saida"
)

# Carrega dados adicionais
system.load_data_from_file("caminho/para/arquivo.csv", "nome_da_fonte")

# Processa uma consulta
result, result_type = system.process_query("Qual é o total de vendas por cliente?")

# Usa o resultado de acordo com o tipo
if result_type == "dataframe":
    print(result)  # DataFrame pandas
elif result_type == "plot":
    # Exibe ou salva o gráfico
    viz_path = "grafico.png"
    import base64, io
    from PIL import Image
    img_data = result.split(",")[1]
    img = Image.open(io.BytesIO(base64.b64decode(img_data)))
    img.save(viz_path)
else:
    print(result)  # Texto ou número
```

### API REST

```bash
# Inicia o servidor API
python integrated_system.py --api
```

Acesse a documentação interativa em http://localhost:8000/docs para explorar todos os endpoints disponíveis:

- `POST /query`: Processa uma consulta em linguagem natural
- `GET /datasources`: Lista as fontes de dados disponíveis
- `GET /stats`: Retorna estatísticas de uso do sistema
- `POST /upload_data`: Faz upload de um arquivo de dados
- `POST /execute_sql`: Executa uma consulta SQL diretamente

## Exemplos de Consultas

O sistema suporta diversos tipos de consultas, como:

- **Consultas básicas**: "Quais são os primeiros 5 registros da tabela de clientes?"
- **Agregações**: "Qual é o valor total de vendas?"
- **Agrupamentos**: "Mostre o total de vendas por cliente"
- **Visualizações**: "Crie um gráfico de barras das vendas por mês"
- **Combinações**: "Qual cliente teve o maior valor de compra e de qual cidade ele é?"
- **Análises temporais**: "Como as vendas evoluíram ao longo do tempo?"
- **Análises por categoria**: "Qual o impacto financeiro por motivo de venda perdida?"

## Configuração

O sistema pode ser configurado através de arquivos JSON:

- `config.json`: Configuração geral do sistema
- `datasources.json`: Configuração de fontes de dados
- `metadata.json`: Metadados para enriquecer a análise
- `llm_config.json`: Configuração para modelos de linguagem

### Exemplo de `config.json`

```json
{
  "data_sources": {
    "files": [
      {"name": "vendas", "path": "dados/vendas.csv", "type": "csv"},
      {"name": "clientes", "path": "dados/clientes.csv", "type": "csv"}
    ],
    "connections": [
      {
        "name": "postgres_db",
        "type": "postgres",
        "host": "localhost",
        "database": "mydatabase",
        "username": "user",
        "password": "pass"
      }
    ]
  },
  "output_types": ["string", "number", "dataframe", "plot"],
  "api": {
    "enabled": true,
    "host": "0.0.0.0",
    "port": 8000
  }
}
```

## Integração com Modelos de Linguagem

Por padrão, o sistema usa uma implementação simulada para gerar código Python/SQL. Para usar modelos de IA avançados, configure o arquivo `llm_config.json`:

```json
{
  "model_type": "openai",
  "model_name": "gpt-3.5-turbo",
  "api_key": "sua-chave-api"
}
```

Ou defina variáveis de ambiente:

```bash
export LLM_MODEL_TYPE=openai
export LLM_MODEL_NAME=gpt-3.5-turbo
export LLM_API_KEY=sua-chave-api
```

Modelos suportados:
- OpenAI (GPT-3.5, GPT-4)
- Anthropic (Claude)
- Hugging Face (diversos modelos)
- Modelos locais (via llama-cpp-python ou gpt4all)

## Extensão

O sistema foi projetado para ser extensível:

1. **Novos Conectores**: Adicione conectores para outras fontes de dados
2. **Novos Tipos de Saída**: Adicione formatos de saída personalizados
3. **Novos Modelos de IA**: Integre com outros modelos de linguagem
4. **Transformações de Dados**: Adicione transformações personalizadas

## Limitações Conhecidas

- O sistema funciona melhor com conjuntos de dados estruturados e bem formatados
- A qualidade da análise depende do modelo de linguagem utilizado
- Consultas muito complexas podem exigir várias interações ou refinamentos
- As visualizações estão limitadas às capacidades do matplotlib

## Licença

Este projeto está licenciado sob a licença MIT - veja o arquivo LICENSE para detalhes.

## Contribuições

Contribuições são bem-vindas! Por favor, abra um issue para discutir alterações significativas antes de enviar um pull request.