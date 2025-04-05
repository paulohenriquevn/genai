# API REST para Sistema de Consulta em Linguagem Natural

Esta API permite interagir com o Sistema de Consulta em Linguagem Natural através de endpoints HTTP, facilitando a integração com aplicações frontend.

## Instalação

1. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure as variáveis de ambiente no arquivo `.env`:
   ```
   OPENAI_API_KEY=sua_chave_aqui
   ANTHROPIC_API_KEY=sua_chave_aqui
   ```

## Execução do Servidor

### Modo Desenvolvimento
```bash
python run_api_server.py --reload
```

### Modo Produção
```bash
python run_api_server.py
```

## Endpoints

### Upload de Arquivo
**Endpoint:** `POST /upload/`

**Parâmetros:**
- `file`: Arquivo (multipart/form-data)
- `description`: Descrição do arquivo (opcional)

**Resposta:**
```json
{
  "file_id": "uuid-identificador-unico",
  "filename": "dados.csv",
  "status": "success"
}
```

### Consulta em Linguagem Natural
**Endpoint:** `POST /query/`

**Parâmetros:**
- `file_id`: ID do arquivo obtido no upload
- `query`: Consulta em linguagem natural

**Resposta:**
```json
{
  "type": "dataframe",
  "query": "Qual o total de vendas por mês?",
  "analysis": "A consulta retornou 12 registros com 2 colunas.",
  "data": [
    {"mes": "2023-01", "total": 1500},
    {"mes": "2023-02", "total": 2000},
    ...
  ]
}
```

### Geração de Visualização
**Endpoint:** `POST /visualization/`

**Parâmetros:**
- `file_id`: ID do arquivo obtido no upload
- `chart_type`: Tipo de gráfico (bar, line, pie, scatter, area, heatmap, radar)
- `x_column`: Coluna para eixo X (opcional)
- `y_column`: Coluna para eixo Y (opcional)
- `title`: Título do gráfico (opcional)

**Resposta:**
```json
{
  "chart": {
    "chart": {"type": "bar"},
    "series": [{"name": "Vendas", "data": [1500, 2000, 1800, ...]}],
    "xaxis": {"categories": ["Jan", "Fev", "Mar", ...]},
    "title": {"text": "Vendas por Mês"}
  },
  "type": "chart",
  "x_column": "mes",
  "y_column": "vendas"
}
```

### Finalizar Sessão
**Endpoint:** `DELETE /session/{file_id}`

**Resposta:**
```json
{
  "status": "success",
  "message": "Sessão encerrada com sucesso"
}
```

## Formatos de Saída

### DataFrame
Para consultas que retornam tabelas de dados:
```json
{
  "type": "dataframe",
  "query": "...",
  "analysis": "...",
  "data": [
    {"coluna1": "valor1", "coluna2": 123},
    {"coluna1": "valor2", "coluna2": 456},
    ...
  ]
}
```

### Gráfico (ApexCharts)
Para consultas que geram visualizações:
```json
{
  "type": "chart",
  "query": "...",
  "analysis": "...",
  "chart": {
    // Configuração ApexCharts
  }
}
```

### Valor Numérico
Para consultas que retornam um número:
```json
{
  "type": "number",
  "query": "...",
  "analysis": "...",
  "data": 12345.67
}
```

### Texto
Para consultas que retornam texto:
```json
{
  "type": "string",
  "query": "...",
  "analysis": "...",
  "data": "Resposta textual aqui"
}
```

## Integração com Frontend

### Exemplo de Upload de Arquivo
```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('description', 'Dados de vendas');

fetch('http://localhost:8000/upload/', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => {
  // Armazena o file_id para consultas futuras
  localStorage.setItem('fileId', data.file_id);
})
.catch(error => console.error('Erro:', error));
```

### Exemplo de Consulta
```javascript
const formData = new FormData();
formData.append('file_id', localStorage.getItem('fileId'));
formData.append('query', 'Mostre o total de vendas por mês');

fetch('http://localhost:8000/query/', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => {
  if (data.type === 'dataframe') {
    // Renderiza tabela
    renderTable(data.data);
  } else if (data.type === 'chart') {
    // Renderiza gráfico ApexCharts
    const chart = new ApexCharts(document.querySelector("#chart"), data.chart);
    chart.render();
  } else {
    // Renderiza outros tipos
    document.getElementById('result').textContent = data.data;
  }
  
  // Mostra análise
  document.getElementById('analysis').textContent = data.analysis;
})
.catch(error => console.error('Erro:', error));
```

### Exemplo de Visualização Direta
```javascript
const formData = new FormData();
formData.append('file_id', localStorage.getItem('fileId'));
formData.append('chart_type', 'bar');
formData.append('x_column', 'mes');
formData.append('y_column', 'vendas');
formData.append('title', 'Vendas Mensais');

fetch('http://localhost:8000/visualization/', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => {
  const chart = new ApexCharts(document.querySelector("#chart"), data.chart);
  chart.render();
})
.catch(error => console.error('Erro:', error));
```

## Notas de Uso

1. Certifique-se de configurar as chaves de API para obter melhores resultados.
2. O tamanho máximo de upload de arquivo padrão é 100MB.
3. Ao finalizar o uso, sempre chame o endpoint `/session/{file_id}` para limpar recursos.
4. As visualizações são geradas em formato ApexCharts, que é compatível com a maioria dos frameworks frontend.

## Formatos de Arquivo Suportados

- CSV (.csv)
- Excel (.xlsx, .xls)
- JSON (.json)
- Parquet (.parquet)
- SQLite (.db, .sqlite)

## Requisitos de Sistema

- Python 3.8 ou superior
- Dependências conforme requirements.txt
- Memória: pelo menos 2GB para processamento de modelos LLM
- Espaço em disco: pelo menos 200MB, mais espaço para armazenar arquivos carregados

## Testes

Execute os testes com:
```bash
pytest tests/test_api.py -v
```