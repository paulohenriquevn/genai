# OpenAI Analyzer

O OpenAI Analyzer é um programa completo que utiliza a API do OpenAI para analisar dados, executar consultas em linguagem natural e gerar visualizações e relatórios.

## Funcionalidades Principais

- **Geração automática de perguntas analíticas**: Utiliza a API do OpenAI para gerar perguntas relevantes sobre seus dados.
- **Processamento de consultas em linguagem natural**: Transforma perguntas em análises de dados.
- **Visualizações automáticas**: Gera gráficos e visualizações quando apropriado.
- **Análise de resultados com IA**: Fornece insights e análises sobre os resultados das consultas.
- **Relatórios HTML interativos**: Gera relatórios completos com visualizações e análises.

## Requisitos

- Python 3.8 ou superior
- Bibliotecas Python (instaláveis via `pip install -r requirements.txt`):
  - pandas
  - numpy
  - matplotlib
  - openai>=1.3.0
  - duckdb
  - argparse

## Instalação

1. Clone este repositório:
   ```
   git clone https://github.com/seu-usuario/natural-language-query-system.git
   cd natural-language-query-system
   ```

2. Instale as dependências:
   ```
   pip install -r requirements.txt
   ```

3. Configure a chave de API do OpenAI:
   ```
   export OPENAI_API_KEY=sua-chave-de-api
   ```

## Uso

### Linha de Comando

O OpenAI Analyzer pode ser executado como script Python com várias opções de linha de comando:

```bash
python openai_analyzer.py [opções]
```

#### Opções disponíveis

- `--api-key API_KEY`: Chave de API do OpenAI (opcional se a variável de ambiente estiver configurada)
- `--model MODEL`: Modelo do OpenAI a ser utilizado (padrão: gpt-4)
- `--data-dir DATA_DIR`: Diretório onde os dados estão armazenados
- `--output-dir OUTPUT_DIR`: Diretório para salvar relatórios e visualizações
- `--topic TOPIC`: Tópico para análise automática (ex: 'vendas', 'clientes')
- `--questions QUESTIONS`: Número de perguntas a serem geradas
- `--query QUERY`: Executar uma consulta específica
- `--open-report`: Abrir relatório no navegador após a análise

### Exemplos de Uso

1. Análise básica com configurações padrão:
   ```bash
   python openai_analyzer.py --topic "vendas perdidas"
   ```

2. Análise personalizada com mais perguntas:
   ```bash
   python openai_analyzer.py --topic "vendas" --questions 8 --open-report
   ```

3. Executar uma consulta específica:
   ```bash
   python openai_analyzer.py --query "Mostre um gráfico de barras com o total de vendas perdidas por motivo"
   ```

4. Análise com modelo e diretórios personalizados:
   ```bash
   python openai_analyzer.py --model "gpt-3.5-turbo" --data-dir "/caminho/para/dados" --output-dir "/caminho/para/saída" --topic "clientes"
   ```

### Uso como Biblioteca Python

O OpenAI Analyzer também pode ser utilizado como uma biblioteca Python em seus próprios scripts:

```python
from openai_analyzer import OpenAIAnalyzer

# Inicializa o analisador
analyzer = OpenAIAnalyzer(
    api_key="sua-chave-de-api",  # Opcional se a variável de ambiente estiver configurada
    model="gpt-4",
    data_dir="/caminho/para/dados",  # Opcional
    output_dir="/caminho/para/saída"  # Opcional
)

# Executa uma consulta específica
result, analysis = analyzer.run_query("Qual é o total de vendas por cliente?")
print(analysis)

# Executa uma análise completa sobre um tópico
analysis_results = analyzer.run_analysis("vendas perdidas", num_questions=5)

# Abre o relatório no navegador
analyzer.open_report(analysis_results["report_path"])
```

## Estrutura de Dados

O OpenAI Analyzer funciona com os seguintes datasets (por padrão):

- **clientes.csv**: Dados de clientes
- **vendas.csv**: Dados de vendas realizadas
- **vendas_perdidas.csv**: Dados de oportunidades de vendas perdidas

Se esses arquivos não forem encontrados, o analisador criará dados sintéticos para demonstração.

## Customização

Você pode personalizar o comportamento do analisador modificando os seguintes aspectos:

- **Datasets**: Adicione ou modifique arquivos CSV no diretório de dados
- **Prompt de análise**: Modifique o prompt usado para gerar análises no código-fonte
- **Estilos de relatório**: Altere o HTML e CSS do relatório no método `generate_html_report`

## Testes

Para executar os testes automatizados:

```bash
python test_openai_analyzer.py
```

## Limitações

- Requer uma chave de API válida do OpenAI
- O uso excessivo pode resultar em custos com a API do OpenAI
- A qualidade das análises depende do modelo e dos prompts utilizados
- Datasets muito grandes podem causar problemas de desempenho

## Contribuições

Contribuições são bem-vindas! Para contribuir:

1. Faça um fork do repositório
2. Crie uma branch para sua feature (`git checkout -b feature/nova-funcionalidade`)
3. Faça commit das suas mudanças (`git commit -am 'Adiciona nova funcionalidade'`)
4. Faça push para a branch (`git push origin feature/nova-funcionalidade`)
5. Abra um Pull Request

## Licença

Este projeto está licenciado sob a licença MIT - veja o arquivo LICENSE para detalhes.