#!/usr/bin/env python3
"""
Exemplo de Integração de ApexCharts
==================================

Este script demonstra como utilizar a integração com ApexCharts
para gerar visualizações interativas a partir de solicitações em
linguagem natural.

Exemplos:
    $ python apex_charts_example.py
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from llm_integration import LLMQueryGenerator, ModelType
from utils.chart_converters import ApexChartsConverter
from core.response.chart import ChartResponse

# Configuração do diretório de saída
OUTPUT_DIR = "output/apex_charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_sample_data():
    """Carrega ou cria dados de exemplo para demonstração."""
    print("📊 Carregando dados de exemplo...")
    
    # Verifica se temos dados reais disponíveis
    if os.path.exists("dados/vendas.csv"):
        print("  ✅ Usando dados reais de vendas.csv")
        vendas_df = pd.read_csv("dados/vendas.csv")
        return vendas_df
    
    # Caso contrário, cria dados sintéticos
    print("  ℹ️ Criando dados sintéticos para demonstração")
    
    # Dados de vendas mensais
    dates = pd.date_range(start="2023-01-01", periods=12, freq='M')
    sales_data = {
        'data': dates,
        'vendas': [1200, 1900, 1500, 1800, 2100, 2400, 2200, 2600, 2300, 2800, 3000, 3200],
        'clientes': [100, 120, 110, 140, 160, 190, 180, 210, 190, 230, 250, 270],
        'ticket_medio': [12, 15.8, 13.6, 12.9, 13.1, 12.6, 12.2, 12.4, 12.1, 12.2, 12.0, 11.9]
    }
    
    # Dados de produtos
    produtos = ['Smartphone', 'Notebook', 'Tablet', 'Monitor', 'Teclado', 'Mouse', 'Headset']
    vendas_por_produto = [45000, 72000, 30000, 24000, 8000, 6000, 15000]
    
    # Criar DataFrames
    vendas_df = pd.DataFrame(sales_data)
    produtos_df = pd.DataFrame({'produto': produtos, 'vendas': vendas_por_produto})
    
    # Salvar para uso futuro
    os.makedirs("dados", exist_ok=True)
    vendas_df.to_csv("dados/vendas_demo.csv", index=False)
    produtos_df.to_csv("dados/produtos_demo.csv", index=False)
    
    return vendas_df

def save_chart(chart, filename):
    """Salva um gráfico no formato adequado (imagem ou JSON)."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    if chart.chart_format == "apex":
        # Salva configuração JSON para ApexCharts
        with open(f"{filepath}.json", 'w', encoding='utf-8') as f:
            if isinstance(chart.value, dict):
                json.dump(chart.value, f, indent=2)
            else:
                f.write(chart.value)
        return f"{filepath}.json"
    else:
        # Salva imagem para formato tradicional
        chart.save(f"{filepath}.png")
        return f"{filepath}.png"

def demonstrate_chart_types(df):
    """Demonstra diferentes tipos de gráficos usando ApexCharts."""
    print("\n📈 Demonstrando diferentes tipos de gráficos ApexCharts...")
    
    examples = []
    
    # 1. Gráfico de Linha
    print("  🔹 Criando gráfico de linha")
    # Preparar dados agrupados por mês
    df['mes'] = pd.to_datetime(df['data']).dt.strftime('%Y-%m')
    vendas_mensais = df.groupby('mes')['valor'].sum().reset_index()
    clientes_mensais = df.groupby('mes')['id_cliente'].nunique().reset_index()
    
    # Juntar em um único DataFrame
    dados_mensais = pd.merge(vendas_mensais, clientes_mensais, on='mes')
    dados_mensais.rename(columns={'valor': 'vendas', 'id_cliente': 'clientes'}, inplace=True)
    
    line_config = ApexChartsConverter.convert_line_chart(
        df=dados_mensais,
        x='mes',
        y=['vendas', 'clientes'],
        title='Vendas e Clientes por Mês (2023)'
    )
    line_chart = ChartResponse(line_config, chart_format="apex")
    line_path = save_chart(line_chart, "1_linha")
    examples.append(("Gráfico de Linha", line_path))
    
    # 2. Gráfico de Barras
    print("  🔹 Criando gráfico de barras")
    bar_config = ApexChartsConverter.convert_bar_chart(
        df=dados_mensais,
        x='mes',
        y='vendas',
        title='Vendas Mensais'
    )
    bar_chart = ChartResponse(bar_config, chart_format="apex")
    bar_path = save_chart(bar_chart, "2_barras")
    examples.append(("Gráfico de Barras", bar_path))
    
    # 3. Gráfico de Área
    print("  🔹 Criando gráfico de área")
    area_config = ApexChartsConverter.convert_area_chart(
        df=dados_mensais,
        x='mes',
        y=['vendas', 'clientes'],
        title='Tendência de Vendas e Clientes',
        stacked=True
    )
    area_chart = ChartResponse(area_config, chart_format="apex")
    area_path = save_chart(area_chart, "3_area")
    examples.append(("Gráfico de Área", area_path))
    
    # 4. Gráfico de Dispersão
    print("  🔹 Criando gráfico de dispersão")
    scatter_config = ApexChartsConverter.convert_scatter_chart(
        df=dados_mensais,
        x='clientes',
        y='vendas',
        title='Relação entre Clientes e Vendas'
    )
    scatter_chart = ChartResponse(scatter_config, chart_format="apex")
    scatter_path = save_chart(scatter_chart, "4_dispersao") 
    examples.append(("Gráfico de Dispersão", scatter_path))
    
    # 5. Gráfico de Pie
    print("  🔹 Criando gráfico de pizza")
    
    # Para o pie chart, vamos criar dados agrupados por cliente
    if not df.empty:
        # Agrupar vendas por cliente
        clientes_df = df.groupby('id_cliente')['valor'].sum().reset_index()
        clientes_df.rename(columns={'id_cliente': 'cliente', 'valor': 'vendas'}, inplace=True)
        
        # Adicionar rótulos para os clientes
        clientes_df['cliente'] = 'Cliente ' + clientes_df['cliente'].astype(str)
        
        pie_config = ApexChartsConverter.convert_pie_chart(
            df=clientes_df,
            labels='cliente',
            values='vendas',
            title='Distribuição de Vendas por Cliente'
        )
        pie_chart = ChartResponse(pie_config, chart_format="apex")
        pie_path = save_chart(pie_chart, "5_pizza")
        examples.append(("Gráfico de Pizza", pie_path))
    
    return examples

def demonstrate_llm_integration():
    """Demonstra a integração com LLM para gerar visualizações."""
    print("\n🤖 Demonstrando integração com LLM para visualizações...")
    
    # Verifica as chaves de API disponíveis
    openai_key = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    
    # Configura o gerador de consultas LLM
    model_type = "mock"  # Padrão
    model_name = None
    api_key = None
    
    if openai_key:
        model_type = ModelType.OPENAI
        model_name = "gpt-3.5-turbo"
        api_key = openai_key
        print("  🔑 Usando OpenAI para geração de código")
    elif anthropic_key:
        model_type = ModelType.ANTHROPIC
        model_name = "claude-3-haiku-20240307"
        api_key = anthropic_key
        print("  🔑 Usando Anthropic para geração de código")
    else:
        print("  ℹ️ Nenhuma chave de API encontrada. Usando modo simulado (mock)")
    
    # Inicializa o gerador de consultas
    llm_generator = LLMQueryGenerator(
        llm_integration=None,
        config_path=None
    )
    
    # Se não estamos no modo mock, configuramos o LLM manualmente
    if model_type != "mock":
        from llm_integration import LLMIntegration
        llm_generator.llm = LLMIntegration(
            model_type=model_type,
            model_name=model_name,
            api_key=api_key
        )
    
    # Consultas de exemplo para visualizações
    visualization_queries = [
        "Gere uma visualização dos dados de vendas por mês",
        "Mostre um gráfico de barras com as vendas de cada mês",
        "Visualize a tendência de vendas ao longo do tempo em um gráfico de linha",
        "Crie um gráfico de pizza mostrando a distribuição de vendas por produto",
        "Gere um gráfico de dispersão comparando o número de clientes e as vendas"
    ]
    
    examples = []
    
    # Processa as consultas e gera visualizações
    for i, query in enumerate(visualization_queries):
        print(f"\n  🔹 Processando consulta: '{query}'")
        
        try:
            # Simula o código gerado para esse exemplo
            # Em um sistema real, este código seria gerado pelo LLM
            if i == 0:  # Visualização genérica
                generated_code = """
import pandas as pd
import matplotlib.pyplot as plt

# Carregar dados
df = pd.read_csv("dados/vendas_demo.csv")
df['data'] = pd.to_datetime(df['data'])

# Preparar dados para visualização
df['mes'] = df['data'].dt.strftime('%Y-%m')

# Definir configuração ApexCharts
result = {
    "type": "chart",
    "value": {
        "format": "apex",
        "config": {
            "chart": {"type": "bar"},
            "series": [{"name": "Vendas", "data": df['vendas'].tolist()}],
            "xaxis": {"categories": df['mes'].tolist()},
            "title": {"text": "Vendas Mensais"}
        }
    }
}
"""
            elif i == 1:  # Gráfico de barras
                generated_code = """
import pandas as pd

# Carregar dados
df = pd.read_csv("dados/vendas_demo.csv")
df['data'] = pd.to_datetime(df['data'])
df['mes'] = df['data'].dt.strftime('%Y-%m')

# Definir configuração ApexCharts
result = {
    "type": "chart",
    "value": {
        "format": "apex",
        "config": {
            "chart": {"type": "bar"},
            "series": [{"name": "Vendas", "data": df['vendas'].tolist()}],
            "xaxis": {"categories": df['mes'].tolist()},
            "title": {"text": "Vendas por Mês"},
            "plotOptions": {
                "bar": {
                    "distributed": true,
                    "dataLabels": {
                        "position": "top"
                    }
                }
            },
            "colors": ["#33b2df", "#546E7A", "#d4526e", "#13d8aa", "#A5978B", "#2b908f", "#f9a3a4", "#90ee7e"]
        }
    }
}
"""
            elif i == 2:  # Gráfico de linha
                generated_code = """
import pandas as pd

# Carregar dados
df = pd.read_csv("dados/vendas_demo.csv")
df['data'] = pd.to_datetime(df['data'])
df['mes'] = df['data'].dt.strftime('%Y-%m')

# Definir configuração ApexCharts
result = {
    "type": "chart",
    "value": {
        "format": "apex",
        "config": {
            "chart": {
                "type": "line",
                "height": 350,
                "zoom": {
                    "enabled": true
                }
            },
            "series": [{"name": "Vendas", "data": df['vendas'].tolist()}],
            "xaxis": {"categories": df['mes'].tolist()},
            "title": {"text": "Tendência de Vendas ao Longo do Tempo"},
            "stroke": {
                "curve": "smooth",
                "width": 3
            },
            "markers": {
                "size": 5
            }
        }
    }
}
"""
            elif i == 3:  # Gráfico de pizza
                generated_code = """
import pandas as pd

# Carregar dados
df = pd.read_csv("dados/produtos_demo.csv")

# Definir configuração ApexCharts
result = {
    "type": "chart",
    "value": {
        "format": "apex",
        "config": {
            "chart": {"type": "pie"},
            "series": df['vendas'].tolist(),
            "labels": df['produto'].tolist(),
            "title": {"text": "Distribuição de Vendas por Produto"},
            "responsive": [{
                "breakpoint": 480,
                "options": {
                    "chart": {"width": 300},
                    "legend": {"position": "bottom"}
                }
            }]
        }
    }
}
"""
            elif i == 4:  # Gráfico de dispersão
                generated_code = """
import pandas as pd

# Carregar dados
df = pd.read_csv("dados/vendas_demo.csv")

# Definir configuração ApexCharts
result = {
    "type": "chart",
    "value": {
        "format": "apex",
        "config": {
            "chart": {
                "type": "scatter",
                "height": 350,
                "zoom": {
                    "type": 'xy'
                }
            },
            "series": [{
                "name": "Vendas vs Clientes",
                "data": [
                    {"x": x, "y": y} for x, y in zip(df['clientes'].tolist(), df['vendas'].tolist())
                ]
            }],
            "xaxis": {
                "title": {"text": "Número de Clientes"},
                "tickAmount": 10
            },
            "yaxis": {
                "title": {"text": "Vendas"}
            },
            "title": {"text": "Relação entre Clientes e Vendas"},
            "markers": {
                "size": 6
            }
        }
    }
}
"""
            
            # Salvar o código gerado
            code_path = os.path.join(OUTPUT_DIR, f"query_{i+1}_codigo.py")
            with open(code_path, 'w', encoding='utf-8') as f:
                f.write(generated_code)
            
            # Em um sistema real, o código seria executado e 
            # o resultado processado. Para este exemplo, vamos criar
            # a estrutura de resultado manualmente.
            
            # Carregar os dados
            if os.path.exists("dados/vendas.csv"):
                df_vendas = pd.read_csv("dados/vendas.csv")
                df_vendas['data'] = pd.to_datetime(df_vendas['data'])
                # Preparar dados de vendas mensais
                df_vendas['mes'] = df_vendas['data'].dt.strftime('%Y-%m')
                vendas_mensais = df_vendas.groupby('mes')['valor'].sum().reset_index()
                vendas_mensais.rename(columns={'valor': 'vendas'}, inplace=True)
                
                # Dados de clientes mensais para os gráficos
                clientes_mensais = df_vendas.groupby('mes')['id_cliente'].nunique().reset_index()
                clientes_mensais.rename(columns={'id_cliente': 'clientes'}, inplace=True)
                
                # Combinando os dados
                df_vendas_processado = pd.merge(vendas_mensais, clientes_mensais, on='mes')
            else:
                # Criar dados de exemplo
                dates = pd.date_range(start="2023-01-01", periods=12, freq='M')
                df_vendas_processado = pd.DataFrame({
                    'mes': [d.strftime('%Y-%m') for d in dates],
                    'vendas': [1200, 1900, 1500, 1800, 2100, 2400, 2200, 2600, 2300, 2800, 3000, 3200],
                    'clientes': [100, 120, 110, 140, 160, 190, 180, 210, 190, 230, 250, 270]
                })
            
            # Criar dados para o gráfico de pizza (vendas por cliente)
            if os.path.exists("dados/vendas.csv"):
                clientes_df = df_vendas.groupby('id_cliente')['valor'].sum().reset_index()
                clientes_df.rename(columns={'id_cliente': 'cliente', 'valor': 'vendas'}, inplace=True)
                clientes_df['cliente'] = 'Cliente ' + clientes_df['cliente'].astype(str)
            else:
                # Dados fictícios de clientes
                clientes = [f'Cliente {i}' for i in range(101, 106)]
                clientes_df = pd.DataFrame({
                    'cliente': clientes,
                    'vendas': [45000, 72000, 30000, 24000, 15000]
                })
            
            # Configurar o gráfico para cada tipo
            if i == 0:  # Visualização genérica
                config = {
                    "chart": {"type": "bar"},
                    "series": [{"name": "Vendas", "data": df_vendas_processado['vendas'].tolist()}],
                    "xaxis": {"categories": df_vendas_processado['mes'].tolist()},
                    "title": {"text": "Vendas Mensais"}
                }
            elif i == 1:  # Gráfico de barras
                config = {
                    "chart": {"type": "bar"},
                    "series": [{"name": "Vendas", "data": df_vendas_processado['vendas'].tolist()}],
                    "xaxis": {"categories": df_vendas_processado['mes'].tolist()},
                    "title": {"text": "Vendas por Mês"},
                    "plotOptions": {
                        "bar": {
                            "distributed": True,
                            "dataLabels": {
                                "position": "top"
                            }
                        }
                    },
                    "colors": ["#33b2df", "#546E7A", "#d4526e", "#13d8aa", "#A5978B", "#2b908f", "#f9a3a4", "#90ee7e"]
                }
            elif i == 2:  # Gráfico de linha
                config = {
                    "chart": {
                        "type": "line",
                        "height": 350,
                        "zoom": {
                            "enabled": True
                        }
                    },
                    "series": [{"name": "Vendas", "data": df_vendas_processado['vendas'].tolist()}],
                    "xaxis": {"categories": df_vendas_processado['mes'].tolist()},
                    "title": {"text": "Tendência de Vendas ao Longo do Tempo"},
                    "stroke": {
                        "curve": "smooth",
                        "width": 3
                    },
                    "markers": {
                        "size": 5
                    }
                }
            elif i == 3:  # Gráfico de pizza
                config = {
                    "chart": {"type": "pie"},
                    "series": clientes_df['vendas'].tolist(),
                    "labels": clientes_df['cliente'].tolist(),
                    "title": {"text": "Distribuição de Vendas por Cliente"},
                    "responsive": [{
                        "breakpoint": 480,
                        "options": {
                            "chart": {"width": 300},
                            "legend": {"position": "bottom"}
                        }
                    }]
                }
            elif i == 4:  # Gráfico de dispersão
                config = {
                    "chart": {
                        "type": "scatter",
                        "height": 350,
                        "zoom": {
                            "type": 'xy'
                        }
                    },
                    "series": [{
                        "name": "Vendas vs Clientes",
                        "data": [
                            {"x": x, "y": y} for x, y in zip(df_vendas_processado['clientes'].tolist(), df_vendas_processado['vendas'].tolist())
                        ]
                    }],
                    "xaxis": {
                        "title": {"text": "Número de Clientes"},
                        "tickAmount": 10
                    },
                    "yaxis": {
                        "title": {"text": "Vendas"}
                    },
                    "title": {"text": "Relação entre Clientes e Vendas"},
                    "markers": {
                        "size": 6
                    }
                }
            
            # Criar objeto ChartResponse
            chart = ChartResponse(config, chart_format="apex")
            
            # Salvar gráfico
            chart_path = save_chart(chart, f"query_{i+1}_resultado")
            examples.append((query, chart_path))
            
            print(f"    ✅ Código e visualização gerados com sucesso")
            
        except Exception as e:
            print(f"    ❌ Erro ao processar consulta: {str(e)}")
    
    return examples

def compare_formats():
    """Compara o formato de imagem tradicional com o ApexCharts."""
    print("\n🔄 Comparando formatos de visualização tradicionais vs ApexCharts...")
    
    # Carregar e preparar dados
    if os.path.exists("dados/vendas.csv"):
        df = pd.read_csv("dados/vendas.csv")
        df['data'] = pd.to_datetime(df['data'])
        # Agrupar por mês
        df['mes'] = df['data'].dt.strftime('%Y-%m')
        vendas_mensais = df.groupby('mes')['valor'].sum().reset_index()
        vendas_mensais.rename(columns={'valor': 'vendas'}, inplace=True)
    else:
        # Criar dados de exemplo
        dates = pd.date_range(start="2023-01-01", periods=12, freq='M')
        vendas_mensais = pd.DataFrame({
            'mes': [d.strftime('%Y-%m') for d in dates],
            'vendas': [1200, 1900, 1500, 1800, 2100, 2400, 2200, 2600, 2300, 2800, 3000, 3200]
        })
    
    # 1. Criar gráfico tradicional com matplotlib
    plt.figure(figsize=(10, 6))
    plt.bar(vendas_mensais['mes'], vendas_mensais['vendas'])
    plt.title('Vendas Mensais')
    plt.xlabel('Mês')
    plt.ylabel('Vendas')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Salvar imagem
    img_path = os.path.join(OUTPUT_DIR, "comparacao_matplotlib.png")
    plt.savefig(img_path)
    plt.close()
    
    # 2. Criar configuração ApexCharts
    apex_config = ApexChartsConverter.convert_bar_chart(
        df=vendas_mensais,
        x='mes',
        y='vendas',
        title='Vendas Mensais'
    )
    
    # Adicionar opções extras para demonstrar recursos
    apex_config.update({
        "theme": {
            "palette": "palette3"
        },
        "annotations": {
            "yaxis": [{
                "y": 2500,
                "borderColor": "#00E396",
                "label": {
                    "borderColor": "#00E396",
                    "style": {
                        "color": "#fff",
                        "background": "#00E396"
                    },
                    "text": "Meta de Vendas"
                }
            }]
        },
        "dataLabels": {
            "enabled": True
        }
    })
    
    # Salvar como JSON
    apex_path = os.path.join(OUTPUT_DIR, "comparacao_apex.json")
    with open(apex_path, 'w', encoding='utf-8') as f:
        json.dump(apex_config, f, indent=2)
    
    return img_path, apex_path

def main():
    """Função principal do script de demonstração."""
    print("🚀 Demonstração de Integração com ApexCharts")
    
    # 1. Carregar ou criar dados de exemplo
    df = load_sample_data()
    
    # 2. Demonstrar diferentes tipos de gráficos
    chart_examples = demonstrate_chart_types(df)
    
    # 3. Demonstrar integração com LLM
    llm_examples = demonstrate_llm_integration()
    
    # 4. Comparar formatos
    img_path, apex_path = compare_formats()
    
    # Exibir resumo
    print("\n✨ Demonstração concluída!")
    print(f"\n📁 Todos os arquivos de exemplo foram salvos em: {OUTPUT_DIR}")
    
    # Resumo dos exemplos de tipos de gráficos
    print("\n📊 Exemplos de tipos de gráficos:")
    for i, (chart_type, path) in enumerate(chart_examples, 1):
        print(f"  {i}. {chart_type}: {path}")
    
    # Resumo dos exemplos de consultas LLM
    print("\n🤖 Exemplos de consultas LLM para visualizações:")
    for i, (query, path) in enumerate(llm_examples, 1):
        print(f"  {i}. Query: \"{query}\"\n     Arquivo: {path}")
    
    # Resumo da comparação
    print("\n🔄 Comparação de formatos:")
    print(f"  Tradicional (imagem): {img_path}")
    print(f"  ApexCharts (JSON): {apex_path}")
    
    print("""
🌐 Para usar as configurações ApexCharts no frontend:

1. Carregue o arquivo JSON da configuração:
   fetch('caminho/para/arquivo.json')
     .then(response => response.json())
     .then(config => {
       var chart = new ApexCharts(
         document.querySelector("#chart"),
         config
       );
       chart.render();
     });

2. Ou use diretamente o objeto de configuração JavaScript:
   var chart = new ApexCharts(
     document.querySelector("#chart"),
     {/* configuração ApexCharts */}
   );
   chart.render();
""")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Demonstração interrompida pelo usuário!")
    except Exception as e:
        print(f"\n❌ Erro: {str(e)}")