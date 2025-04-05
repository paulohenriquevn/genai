# ApexCharts Integration Guide

Este documento descreve como utilizar a integração com ApexCharts para criar visualizações interativas no Sistema de Consulta em Linguagem Natural.

## Visão Geral

ApexCharts é uma biblioteca JavaScript moderna para criar visualizações interativas e responsivas. Agora o sistema suporta a geração de configurações ApexCharts em formato JSON, que podem ser facilmente integradas em aplicações web.

## Tipos de Gráficos Suportados

A integração suporta os seguintes tipos de gráficos:

- **Line** (Linha)
- **Bar** (Barras) 
- **Pie** (Pizza)
- **Scatter** (Dispersão)
- **Area** (Área)
- **Heatmap** (Mapa de Calor)
- **Radar** (Radar)

## Como Utilizar

### 1. Via API Programática

Para gerar um gráfico no formato ApexCharts programaticamente:

```python
from core.engine.analysis_engine import AnalysisEngine

# Inicializa o engine
engine = AnalysisEngine()

# Carrega dados
engine.load_data("dados/vendas.csv", "vendas")

# Exemplo de gráfico de barras
chart_response = engine.generate_chart(
    data=df,
    chart_type='bar',
    x='categoria',
    y='valor',
    title='Vendas por Categoria',
    chart_format='apex',  # Importante: especifica o formato ApexCharts
    options={  # Opções adicionais de customização
        'colors': ['#33b2df', '#546E7A'],
        'plotOptions': {
            'bar': {
                'borderRadius': 10,
                'columnWidth': '50%'
            }
        }
    }
)

# Obter configuração como dicionário Python
config = chart_response.to_apex_json()

# Salvar em arquivo JSON
with open('chart_config.json', 'w') as f:
    import json
    json.dump(config, f, indent=2)
```

### 2. Via Consulta em Linguagem Natural

Ao fazer uma consulta que solicita uma visualização, o sistema agora gera automaticamente gráficos no formato ApexCharts:

```
"Mostre um gráfico de barras das vendas por mês"
"Apresente um gráfico de linha da evolução de temperatura ao longo do ano"
"Gere um gráfico de pizza com as vendas por categoria"
```

### 3. Integração com Frontend

Para usar a configuração gerada em uma aplicação web:

```javascript
// Exemplo React com ApexCharts
import React from 'react';
import Chart from 'react-apexcharts';

function ApexChartComponent({ chartConfig }) {
  return (
    <div className="chart-container">
      <Chart
        options={chartConfig}
        series={chartConfig.series}
        type={chartConfig.chart.type}
        height={chartConfig.chart.height}
      />
    </div>
  );
}

export default ApexChartComponent;
```

## Opções de Customização

Você pode customizar diversos aspectos dos gráficos usando o parâmetro `options`:

### Cores e Temas

```python
options={
    'colors': ['#008FFB', '#00E396', '#FEB019', '#FF4560', '#775DD0'],
    'theme': {
        'mode': 'light',  # ou 'dark'
        'palette': 'palette1'  # ou palette2, palette3, etc.
    }
}
```

### Legendas e Rótulos

```python
options={
    'legend': {
        'position': 'right',  # 'top', 'right', 'bottom', 'left'
        'fontSize': '14px'
    },
    'dataLabels': {
        'enabled': True,
        'formatter': "function(val) { return val + '%' }"
    }
}
```

### Formatação de Eixos

```python
options={
    'xaxis': {
        'labels': {
            'rotate': -45,
            'style': {
                'fontSize': '12px'
            }
        }
    },
    'yaxis': {
        'labels': {
            'formatter': "function(val) { return '€' + val }"
        }
    }
}
```

### Animações e Interatividade

```python
options={
    'chart': {
        'animations': {
            'enabled': True,
            'easing': 'easeinout',
            'speed': 800
        }
    },
    'tooltip': {
        'enabled': True,
        'shared': True,
        'intersect': False
    }
}
```

## Exemplos por Tipo de Gráfico

### Gráfico de Linha

```python
engine.generate_chart(
    data=df,
    chart_type='line',
    x='mes',
    y='vendas',
    title='Vendas Mensais',
    chart_format='apex',
    options={
        'stroke': {'curve': 'smooth', 'width': 2},
        'markers': {'size': 4}
    }
)
```

### Gráfico de Barras

```python
engine.generate_chart(
    data=df,
    chart_type='bar',
    x='categoria',
    y='valor',
    title='Vendas por Categoria',
    chart_format='apex',
    options={
        'plotOptions': {'bar': {'horizontal': False}}  # True para barras horizontais
    }
)
```

### Gráfico de Pizza/Donut

```python
engine.generate_chart(
    data=df,
    chart_type='pie',
    x='categoria',
    y='valor',
    title='Distribuição de Vendas',
    chart_format='apex',
    options={
        'plotOptions': {'pie': {'donut': {'size': '50%'}}},  # Para gráfico de donut
        'labels': ['A', 'B', 'C', 'D', 'E']  # Opcional: sobrescrever labels
    }
)
```

### Gráfico de Dispersão

```python
engine.generate_chart(
    data=df,
    chart_type='scatter',
    x='x',
    y='y',
    title='Correlação entre X e Y',
    chart_format='apex',
    options={
        'markers': {'size': 6}
    }
)
```

## Formato ApexCharts

O formato JSON gerado segue a estrutura esperada pela biblioteca ApexCharts:

```json
{
  "chart": {
    "type": "bar"
  },
  "series": [{
    "name": "Vendas",
    "data": [10, 20, 30, 40]
  }],
  "xaxis": {
    "categories": ["A", "B", "C", "D"]
  },
  "title": {
    "text": "Título do Gráfico"
  }
}
```

## Limitações Atuais

- Formatadores personalizados (funções JavaScript) devem ser implementados no frontend
- Alguns tipos de gráficos avançados ainda não são suportados (candlestick, treemap, etc.)
- Para visualizações muito complexas, pode ser necessário ajustar a configuração manualmente

## Referências

- [Documentação ApexCharts](https://apexcharts.com/docs/installation/)
- [Exemplos e Demos](https://apexcharts.com/javascript-chart-demos/)
- [React ApexCharts](https://apexcharts.com/docs/react-charts/)