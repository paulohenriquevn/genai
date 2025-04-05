"""
Teste da funcionalidade de geração de gráficos no formato ApexCharts.
"""

import pandas as pd
import json
import os
from core.engine.analysis_engine import AnalysisEngine
from utils.chart_converters import ApexChartsConverter

def create_sample_data():
    """Cria dados de exemplo para os testes."""
    # Dados de vendas mensais
    sales_data = {
        'month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        'sales': [1200, 1800, 2400, 2200, 2600, 3100, 3400, 3200, 2900, 2400, 2100, 2800],
        'expenses': [800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200, 1100, 1050, 1000]
    }
    
    # Dados de categorias
    categories_data = {
        'category': ['Electronics', 'Clothing', 'Food', 'Books', 'Home'],
        'sales': [12500, 8900, 7200, 3500, 6400]
    }
    
    return pd.DataFrame(sales_data), pd.DataFrame(categories_data)

def test_line_chart():
    """Testa a geração de gráfico de linha com ApexCharts."""
    sales_df, _ = create_sample_data()
    
    # Cria o engine
    engine = AnalysisEngine()
    
    # Gera o gráfico de linha
    chart_response = engine.generate_chart(
        data=sales_df,
        chart_type='line',
        x='month',
        y=['sales', 'expenses'],
        title='Monthly Sales and Expenses',
        chart_format='apex',
        options={
            'stroke': {'width': 3},
            'colors': ['#008FFB', '#FF4560']
        }
    )
    
    # Verifica o tipo de resposta
    assert chart_response.type == 'chart'
    assert chart_response.chart_format == 'apex'
    
    # Verifica a estrutura básica do JSON
    config = chart_response.to_apex_json()
    assert 'chart' in config
    assert config['chart']['type'] == 'line'
    assert 'series' in config
    assert len(config['series']) == 2  # sales e expenses
    assert 'xaxis' in config
    assert 'title' in config
    assert config['title']['text'] == 'Monthly Sales and Expenses'
    
    # Salva a configuração em um arquivo para inspeção
    output_dir = 'output/apex_charts'
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f'{output_dir}/line_chart.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Gráfico de linha salvo em: {output_dir}/line_chart.json")
    return chart_response

def test_bar_chart():
    """Testa a geração de gráfico de barras com ApexCharts."""
    sales_df, _ = create_sample_data()
    
    # Cria o engine
    engine = AnalysisEngine()
    
    # Gera o gráfico de barras
    chart_response = engine.generate_chart(
        data=sales_df,
        chart_type='bar',
        x='month',
        y='sales',
        title='Monthly Sales',
        chart_format='apex',
        options={
            'colors': ['#33b2df'],
            'plotOptions': {
                'bar': {
                    'borderRadius': 10,
                    'dataLabels': {'position': 'top'}
                }
            }
        }
    )
    
    # Verifica o tipo de resposta
    assert chart_response.type == 'chart'
    assert chart_response.chart_format == 'apex'
    
    # Verifica a estrutura básica do JSON
    config = chart_response.to_apex_json()
    assert 'chart' in config
    assert config['chart']['type'] == 'bar'
    assert 'series' in config
    assert 'xaxis' in config
    assert 'title' in config
    assert config['title']['text'] == 'Monthly Sales'
    
    # Salva a configuração em um arquivo para inspeção
    output_dir = 'output/apex_charts'
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f'{output_dir}/bar_chart.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Gráfico de barras salvo em: {output_dir}/bar_chart.json")
    return chart_response

def test_pie_chart():
    """Testa a geração de gráfico de pizza com ApexCharts."""
    _, categories_df = create_sample_data()
    
    # Cria o engine
    engine = AnalysisEngine()
    
    # Gera o gráfico de pizza
    chart_response = engine.generate_chart(
        data=categories_df,
        chart_type='pie',
        x='category',
        y='sales',
        title='Sales by Category',
        chart_format='apex',
        options={
            'legend': {'position': 'bottom'},
            'colors': ['#775DD0', '#FF4560', '#FEB019', '#00E396', '#008FFB']
        }
    )
    
    # Verifica o tipo de resposta
    assert chart_response.type == 'chart'
    assert chart_response.chart_format == 'apex'
    
    # Verifica a estrutura básica do JSON
    config = chart_response.to_apex_json()
    assert 'chart' in config
    assert config['chart']['type'] in ['pie', 'donut']  # pode ser pie ou donut
    assert 'series' in config
    assert 'labels' in config
    assert 'title' in config
    assert config['title']['text'] == 'Sales by Category'
    
    # Salva a configuração em um arquivo para inspeção
    output_dir = 'output/apex_charts'
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f'{output_dir}/pie_chart.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Gráfico de pizza salvo em: {output_dir}/pie_chart.json")
    return chart_response

def test_parser_compatibility():
    """Testa a compatibilidade do parser com o formato ApexCharts."""
    from core.response.parser import ResponseParser
    
    # Formato ApexCharts
    result_apex = {
        "type": "chart",
        "value": {
            "format": "apex",
            "config": {
                "chart": {"type": "line"},
                "series": [{"name": "Sales", "data": [1200, 1800, 2400]}],
                "xaxis": {"categories": ["Jan", "Feb", "Mar"]}
            }
        }
    }
    
    # Formato imagem (compatibilidade)
    result_image = {
        "type": "chart",
        "value": "path/to/chart.png"
    }
    
    # Formato antigo (plot)
    result_old = {
        "type": "plot",
        "value": "path/to/plot.png"
    }
    
    parser = ResponseParser()
    
    # Testa formato ApexCharts
    response_apex = parser.parse(result_apex)
    assert response_apex.type == 'chart'
    assert response_apex.chart_format == 'apex'
    assert 'chart' in response_apex.value
    assert 'series' in response_apex.value
    
    # Testa formato imagem
    response_image = parser.parse(result_image)
    assert response_image.type == 'chart'
    assert response_image.chart_format == 'image'
    assert isinstance(response_image.value, str)
    
    # Testa formato antigo
    response_old = parser.parse(result_old)
    assert response_old.type == 'chart'
    assert response_old.chart_format == 'image'
    assert isinstance(response_old.value, str)
    
    print("Todos os testes de compatibilidade do parser passaram!")

if __name__ == "__main__":
    print("\n=== TESTES DE INTEGRAÇÃO COM APEXCHARTS ===\n")
    
    # Executa os testes
    try:
        line_chart = test_line_chart()
        print("✅ Teste de gráfico de linha concluído com sucesso!")
        
        bar_chart = test_bar_chart()
        print("✅ Teste de gráfico de barras concluído com sucesso!")
        
        pie_chart = test_pie_chart()
        print("✅ Teste de gráfico de pizza concluído com sucesso!")
        
        test_parser_compatibility()
        print("✅ Teste de compatibilidade do parser concluído com sucesso!")
        
        print("\nTodos os testes concluídos com sucesso. Arquivos JSON gerados na pasta 'output/apex_charts/'")
    except Exception as e:
        print(f"❌ Erro durante execução dos testes: {str(e)}")