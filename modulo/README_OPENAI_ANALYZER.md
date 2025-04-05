# OpenAI Analyzer

Sistema avançado de análise de dados que utiliza a API do OpenAI para gerar perguntas analíticas, processar consultas em linguagem natural, criar visualizações e gerar relatórios completos de análise.

## Recursos Principais

- **Análise Automática de Datasets**: Detecta automaticamente tipos de campo, gera metadados e esquemas semânticos
- **Geração de Consultas Analíticas**: Cria automaticamente perguntas relevantes sobre seus dados
- **Interpretação em Linguagem Natural**: Traduz perguntas em linguagem natural para consultas de dados
- **Visualizações Automáticas**: Gera gráficos e visualizações apropriados para os dados
- **Análise de Resultados com IA**: Fornece insights detalhados sobre os resultados de consultas
- **Relatórios HTML Interativos**: Gera relatórios completos com visualizações e análises
- **Logging Detalhado**: Mantém registros de consultas SQL, erros e fluxo de execução

## Instalação

1. Clone o repositório:
   ```bash
   git clone https://github.com/seu-usuario/natural-language-query-system.git
   cd natural-language-query-system
   ```

2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure a chave de API do OpenAI:
   ```bash
   export OPENAI_API_KEY=sua-chave-de-api
   ```

## Uso Básico

### Linha de Comando

```bash
# Análise básica com tópico padrão (vendas perdidas)
python openai_analyzer.py

# Análise com tópico personalizado
python openai_analyzer.py --topic "vendas" --questions 8

# Executar uma consulta específica
python openai_analyzer.py --query "Mostre um gráfico de barras com o total de vendas perdidas por motivo"

# Processar um dataset específico e gerar esquema semântico
python openai_analyzer.py --dataset vendas_2023.csv --process-dataset-only

# Análise completa com dataset específico
python openai_analyzer.py --dataset vendas_2023.csv --topic "performance de vendas"

# Abrir relatório no navegador após a análise
python openai_analyzer.py --open-report
```

### Como Biblioteca Python

```python
from openai_analyzer import OpenAIAnalyzer

# Inicializar o analisador
analyzer = OpenAIAnalyzer(
    api_key="sua-chave-de-api",  # Opcional se definida como variável de ambiente
    model="gpt-4",               # Modelo OpenAI a usar
    data_dir="/caminho/para/dados",
    output_dir="/caminho/para/saída"
)

# Método 1: Processar um dataset específico
schema = analyzer.process_dataset("/caminho/para/dataset.csv")

# Método 2: Executar uma consulta específica
result, analysis = analyzer.run_query("Quais são os principais motivos de vendas perdidas?")

# Método 3: Executar análise completa
analysis_results = analyzer.run_analysis("vendas", num_questions=5)

# Abrir o relatório no navegador
analyzer.open_report(analysis_results["report_path"])
```

## Fluxo de Processamento

1. **Inicialização**:
   - Carrega configurações e inicializa componentes
   - Configura logging e integração LLM

2. **Processamento de Dataset** (opcional):
   - Lê o dataset usando `DatasetAnalyzer`
   - Analisa campos e gera metadados
   - Cria esquema semântico com tipagem e relações

3. **Geração de Análise**:
   - Gera perguntas analíticas relevantes usando OpenAI
   - Executa consultas através do motor de consulta
   - Analisa resultados e gera insights

4. **Visualização e Relatório**:
   - Salva visualizações geradas
   - Cria relatório HTML com todas as análises e gráficos
   - Gera um resumo executivo da análise

## Opções de Linha de Comando

| Opção | Descrição |
|-------|-----------|
| `--api-key KEY` | Chave de API do OpenAI |
| `--model MODEL` | Modelo OpenAI a ser utilizado (padrão: "gpt-4") |
| `--data-dir DIR` | Diretório onde os dados estão armazenados |
| `--output-dir DIR` | Diretório para salvar relatórios e visualizações |
| `--dataset PATH` | Caminho para um arquivo de dataset específico |
| `--dataset-name NAME` | Nome personalizado para o dataset |
| `--topic TOPIC` | Tópico para análise automática (padrão: "vendas perdidas") |
| `--questions NUM` | Número de perguntas a serem geradas (padrão: 5) |
| `--query QUERY` | Executar uma consulta específica |
| `--open-report` | Abrir relatório no navegador após a análise |
| `--log-level LEVEL` | Nível de logging (DEBUG, INFO, WARNING, ERROR) |
| `--process-dataset-only` | Apenas processar o dataset e gerar metadados/esquema |

## Exemplo de Saída

### Relatório HTML

O relatório HTML contém várias seções:

1. **Resumo da Análise**: Síntese dos principais insights
2. **Visualizações**: Todas as visualizações geradas com suas análises
3. **Consultas e Análises**: Todas as consultas executadas e suas análises detalhadas
4. **Informações Técnicas**: Detalhes sobre o esquema semântico e metadados

### Logs

O sistema gera vários arquivos de log:
- `openai_analyzer.log`: Log principal com todas as operações
- `sql_queries.log`: Registro de todas as consultas SQL executadas
- `natural_query_engine.log`: Log do motor de consulta

## Tratamento de Erros

O OpenAI Analyzer inclui tratamento robusto de erros:
- Fallback para outros modelos de LLM se OpenAI falhar
- Geração de dados sintéticos quando dados reais não estão disponíveis
- Backups para análises quando a API falha
- Logging detalhado para depuração

## Personalização

### Configuração de LLM

O sistema suporta vários modelos LLM:
- OpenAI (padrão): GPT-4, GPT-3.5-turbo
- Anthropic: Claude (como fallback)
- Hugging Face: Modelos transformers
- Modo Mock: Para testes sem API

### Visualizações Personalizadas

O OpenAI Analyzer suporta vários tipos de visualizações:
- Gráficos de barras para distribuições categóricas
- Histogramas para distribuições numéricas
- Gráficos de linha para tendências temporais
- Boxplots para análises de correlação
- Gráficos de pizza para comparações de proporção

## Requisitos

- Python 3.8+
- pandas
- numpy
- matplotlib
- openai>=1.3.0
- duckdb

## Licença

Este projeto está licenciado sob a licença MIT - veja o arquivo LICENSE para detalhes.

## Contribuição

Contribuições são bem-vindas! Por favor, leia as diretrizes de contribuição antes de enviar um pull request.