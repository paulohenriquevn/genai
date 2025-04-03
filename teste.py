import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots

class PyApexCharts:
    """
    Classe para criar gráficos com estilo similar ao ApexCharts usando Plotly
    """
    
    def __init__(self):
        # Tema padrão similar ao ApexCharts
        self.theme = {
            'background_color': '#fff',
            'grid_color': '#f8f9fa',
            'text_color': '#373d3f',
            'axis_color': '#78909c',
            'title_color': '#263238',
            'colors': ['#008FFB', '#00E396', '#FEB019', '#FF4560', '#775DD0', '#546E7A', '#26a69a', '#D10CE8']
        }
    
    def _apply_theme(self, fig):
        """Aplica o tema no estilo ApexCharts ao gráfico"""
        fig.update_layout(
            plot_bgcolor=self.theme['background_color'],
            paper_bgcolor=self.theme['background_color'],
            font_color=self.theme['text_color'],
            title_font_color=self.theme['title_color'],
            xaxis=dict(
                showgrid=True,
                gridcolor=self.theme['grid_color'],
                tickfont=dict(color=self.theme['axis_color']),
                title_font=dict(color=self.theme['axis_color'])
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor=self.theme['grid_color'],
                tickfont=dict(color=self.theme['axis_color']),
                title_font=dict(color=self.theme['axis_color'])
            ),
            legend=dict(
                font=dict(color=self.theme['text_color'])
            ),
            margin=dict(t=50, b=50, l=50, r=50)
        )
        return fig
    
    def line(self, x, y, title="Gráfico de Linha", names=None, markers=True):
        """Cria um gráfico de linha estilo ApexCharts"""
        fig = go.Figure()
        
        # Se y for uma lista de listas, adiciona múltiplas linhas
        if isinstance(y[0], (list, np.ndarray)):
            for i, y_data in enumerate(y):
                name = names[i] if names and i < len(names) else f'Série {i+1}'
                fig.add_trace(go.Scatter(
                    x=x,
                    y=y_data,
                    mode='lines+markers' if markers else 'lines',
                    name=name,
                    line=dict(color=self.theme['colors'][i % len(self.theme['colors'])], width=3),
                    marker=dict(size=8) if markers else None
                ))
        else:
            # Caso seja apenas uma linha
            name = names[0] if names and len(names) > 0 else 'Série 1'
            fig.add_trace(go.Scatter(
                x=x,
                y=y,
                mode='lines+markers' if markers else 'lines',
                name=name,
                line=dict(color=self.theme['colors'][0], width=3),
                marker=dict(size=8) if markers else None
            ))
        
        fig.update_layout(
            title=title,
            hovermode="x unified",
        )
        
        return self._apply_theme(fig)
    
    def area(self, x, y, title="Gráfico de Área", names=None, stacked=False):
        """Cria um gráfico de área estilo ApexCharts"""
        fig = go.Figure()
        
        # Se y for uma lista de listas, adiciona múltiplas áreas
        if isinstance(y[0], (list, np.ndarray)):
            for i, y_data in enumerate(y):
                name = names[i] if names and i < len(names) else f'Série {i+1}'
                fig.add_trace(go.Scatter(
                    x=x,
                    y=y_data,
                    mode='lines',
                    fill='tonexty' if i > 0 and stacked else 'tozeroy',
                    name=name,
                    line=dict(color=self.theme['colors'][i % len(self.theme['colors'])], width=2),
                    fillcolor=f'rgba({int(self.theme["colors"][i % len(self.theme["colors"])].lstrip("#")[0:2], 16)}, {int(self.theme["colors"][i % len(self.theme["colors"])].lstrip("#")[2:4], 16)}, {int(self.theme["colors"][i % len(self.theme["colors"])].lstrip("#")[4:6], 16)}, 0.25)'
                ))
        else:
            # Caso seja apenas uma área
            name = names[0] if names and len(names) > 0 else 'Série 1'
            fig.add_trace(go.Scatter(
                x=x,
                y=y,
                mode='lines',
                fill='tozeroy',
                name=name,
                line=dict(color=self.theme['colors'][0], width=2),
                fillcolor=f'rgba({int(self.theme["colors"][0].lstrip("#")[0:2], 16)}, {int(self.theme["colors"][0].lstrip("#")[2:4], 16)}, {int(self.theme["colors"][0].lstrip("#")[4:6], 16)}, 0.25)'
            ))
        
        fig.update_layout(
            title=title,
            hovermode="x unified",
        )
        
        # If stacked is True, modify the fill behavior of subsequent traces
        if stacked:
            fig.update_traces(fill='tonexty')
        
        return self._apply_theme(fig)

    def bar(self, x, y, title="Gráfico de Barras", names=None, horizontal=False, stacked=False):
        """Cria um gráfico de barras estilo ApexCharts"""
        fig = go.Figure()
        
        # Se y for uma lista de listas, adiciona múltiplas barras
        if isinstance(y[0], (list, np.ndarray)):
            for i, y_data in enumerate(y):
                name = names[i] if names and i < len(names) else f'Série {i+1}'
                if horizontal:
                    fig.add_trace(go.Bar(
                        y=x,
                        x=y_data,
                        name=name,
                        orientation='h',
                        marker_color=self.theme['colors'][i % len(self.theme['colors'])],
                    ))
                else:
                    fig.add_trace(go.Bar(
                        x=x,
                        y=y_data,
                        name=name,
                        marker_color=self.theme['colors'][i % len(self.theme['colors'])],
                    ))
        else:
            # Caso seja apenas uma série de barras
            name = names[0] if names and len(names) > 0 else 'Série 1'
            if horizontal:
                fig.add_trace(go.Bar(
                    y=x,
                    x=y,
                    name=name,
                    orientation='h',
                    marker_color=self.theme['colors'][0],
                ))
            else:
                fig.add_trace(go.Bar(
                    x=x,
                    y=y,
                    name=name,
                    marker_color=self.theme['colors'][0],
                ))
        
        fig.update_layout(
            title=title,
            barmode='stack' if stacked else 'group',
        )
        
        return self._apply_theme(fig)
    
    def pie(self, labels, values, title="Gráfico de Pizza"):
        """Cria um gráfico de pizza estilo ApexCharts"""
        colors = self.theme['colors'][:len(labels)] if len(labels) <= len(self.theme['colors']) else None
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            textinfo='percent',
            insidetextorientation='radial',
            marker_colors=colors,
            hole=0
        )])
        
        fig.update_layout(
            title=title,
        )
        
        return self._apply_theme(fig)
    
    def donut(self, labels, values, title="Gráfico de Donut"):
        """Cria um gráfico de donut estilo ApexCharts"""
        colors = self.theme['colors'][:len(labels)] if len(labels) <= len(self.theme['colors']) else None
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            textinfo='percent',
            insidetextorientation='radial',
            marker_colors=colors,
            hole=0.5
        )])
        
        fig.update_layout(
            title=title,
        )
        
        return self._apply_theme(fig)
    
    def radar(self, categories, values, title="Gráfico Radar", names=None):
        """Cria um gráfico radar estilo ApexCharts"""
        fig = go.Figure()
        
        # Se values for uma lista de listas, adiciona múltiplas séries
        if isinstance(values[0], (list, np.ndarray)):
            for i, val in enumerate(values):
                name = names[i] if names and i < len(names) else f'Série {i+1}'
                color = self.theme['colors'][i % len(self.theme['colors'])]
                
                # Convert hex to RGB
                rgb = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                
                fig.add_trace(go.Scatterpolar(
                    r=val,
                    theta=categories,
                    fill='toself',
                    name=name,
                    line_color=color,
                    fillcolor=f'rgba{(*rgb, 0.25)}'
                ))
        else:
            # Caso seja apenas uma série
            name = names[0] if names and len(names) > 0 else 'Série 1'
            color = self.theme['colors'][0]
            
            # Convert hex to RGB
            rgb = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=name,
                line_color=color,
                fillcolor=f'rgba{(*rgb, 0.25)}'
            ))
        
        fig.update_layout(
            title=title,
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    gridcolor=self.theme['grid_color']
                )
            ),
        )
        
        return self._apply_theme(fig)
    
    def heatmap(self, x, y, z, title="Heatmap"):
        """Cria um heatmap estilo ApexCharts"""
        fig = go.Figure(data=go.Heatmap(
            z=z,
            x=x,
            y=y,
            colorscale='Blues',
            hoverongaps=False,
            colorbar=dict(
                thickness=20,
                tickfont=dict(color=self.theme['axis_color'])
            )
        ))
        
        fig.update_layout(
            title=title,
        )
        
        return self._apply_theme(fig)
    
    def mixed(self, x, data, title="Gráfico Misto"):
        """
        Cria um gráfico misto combinando diferentes tipos
        data deve ser uma lista de dicionários com:
        - y: valores
        - type: 'bar', 'line', 'area'
        - name: nome da série
        """
        fig = go.Figure()
        
        for i, series in enumerate(data):
            y = series['y']
            series_type = series.get('type', 'line')
            name = series.get('name', f'Série {i+1}')
            color = series.get('color', self.theme['colors'][i % len(self.theme['colors'])])
            
            if series_type == 'bar':
                fig.add_trace(go.Bar(
                    x=x,
                    y=y,
                    name=name,
                    marker_color=color
                ))
            elif series_type == 'line':
                fig.add_trace(go.Scatter(
                    x=x,
                    y=y,
                    mode='lines+markers',
                    name=name,
                    line=dict(color=color, width=3),
                    marker=dict(size=8)
                ))
            elif series_type == 'area':
                fig.add_trace(go.Scatter(
                    x=x,
                    y=y,
                    mode='lines',
                    fill='tozeroy',
                    name=name,
                    line=dict(color=color, width=2),
                    fillcolor=color + '40'
                ))
        
        fig.update_layout(
            title=title,
            hovermode="x unified"
        )
        
        return self._apply_theme(fig)
    
    def candlestick(self, dates, open_data, high_data, low_data, close_data, title="Gráfico de Candlestick"):
        """Cria um gráfico de candlestick estilo ApexCharts"""
        fig = go.Figure(data=[go.Candlestick(
            x=dates,
            open=open_data,
            high=high_data,
            low=low_data,
            close=close_data,
            increasing_line_color=self.theme['colors'][1],  # verde para alta
            decreasing_line_color=self.theme['colors'][3]   # vermelho para baixa
        )])
        
        fig.update_layout(
            title=title,
            xaxis_rangeslider_visible=False
        )
        
        return self._apply_theme(fig)
    
    def spark(self, data, width=200, height=60, color=None, fill=True, title=None):
        """Cria um sparkline (mini gráfico de linha)"""
        color = color or self.theme['colors'][0]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=data,
            mode='lines',
            fill='tozeroy' if fill else None,
            line=dict(color=color, width=2),
            fillcolor=color + '30' if fill else None
        ))
        
        # Configurações específicas para sparkline
        fig.update_layout(
            title=title,
            width=width,
            height=height,
            margin=dict(t=5, b=5, l=5, r=5),
            showlegend=False,
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                showline=False
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                showline=False
            ),
            hovermode=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
    
    def set_theme(self, theme_dict):
        """Atualiza o tema do gráfico com as cores fornecidas"""
        for key, value in theme_dict.items():
            if key in self.theme:
                self.theme[key] = value
                
    def save_as_html(self, fig, file_path):
        """Salva o gráfico como um arquivo HTML interativo"""
        fig.write_html(file_path)
        
    def save_as_image(self, fig, file_path, format='png', width=1000, height=600):
        """Salva o gráfico como uma imagem"""
        fig.write_image(file_path, format=format, width=width, height=height)


# Exemplo de uso
if __name__ == "__main__":
    # Inicializa a classe de gráficos
    charts = PyApexCharts()
    
    # Dados de exemplo
    meses = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
    dados = [45, 52, 38, 24, 33, 26, 21, 20, 6, 8, 15, 10]
    vendas = [35, 41, 62, 42, 13, 18, 29, 37, 36, 51, 32, 35]
    
    # Gráfico de linha
    fig_line = charts.line(
        x=meses, 
        y=[dados, vendas], 
        title="Vendas e Dados por Mês", 
        names=["Dados", "Vendas"]
    )
    fig_line.show()
    
    # Gráfico de área
    fig_area = charts.area(
        x=meses, 
        y=[dados, vendas], 
        title="Área de Vendas e Dados", 
        names=["Dados", "Vendas"],
        stacked=True
    )
    fig_area.show()
    
    # Gráfico de barras
    fig_bar = charts.bar(
        x=meses, 
        y=[dados, vendas], 
        title="Barras de Vendas e Dados", 
        names=["Dados", "Vendas"],
        stacked=True
    )
    fig_bar.show()
    
    # Gráfico de pizza
    categorias = ['Produto A', 'Produto B', 'Produto C', 'Produto D']
    valores = [42, 26, 15, 17]
    fig_pie = charts.pie(
        labels=categorias, 
        values=valores, 
        title="Distribuição de Produtos"
    )
    fig_pie.show()
    
    # Gráfico de donut
    fig_donut = charts.donut(
        labels=categorias, 
        values=valores, 
        title="Distribuição de Produtos (Donut)"
    )
    fig_donut.show()
    
    # Gráfico radar
    metricas = ['Velocidade', 'Confiabilidade', 'Custo', 'Usabilidade', 'Suporte']
    valores_radar = [80, 65, 90, 75, 60]
    valores_radar2 = [70, 85, 40, 80, 90]
    fig_radar = charts.radar(
        categories=metricas, 
        values=[valores_radar, valores_radar2], 
        title="Comparação de Produtos", 
        names=["Produto A", "Produto B"]
    )
    fig_radar.show()
    
    # Gráfico misto
    data_mixed = [
        {'y': dados, 'type': 'bar', 'name': 'Dados'},
        {'y': vendas, 'type': 'line', 'name': 'Vendas'}
    ]
    fig_mixed = charts.mixed(
        x=meses, 
        data=data_mixed, 
        title="Gráfico Misto - Barras e Linhas"
    )
    fig_mixed.show()
    
    # Heatmap
    matriz = [
        [1, 20, 30, 50, 1],
        [20, 1, 60, 80, 30],
        [30, 60, 1, 8, 69],
        [50, 80, 8, 1, 75],
        [1, 30, 69, 75, 1]
    ]
    fig_heatmap = charts.heatmap(
        x=['A', 'B', 'C', 'D', 'E'],
        y=['V', 'W', 'X', 'Y', 'Z'],
        z=matriz,
        title="Matriz de Correlação"
    )
    fig_heatmap.show()
    
    # Exemplo de personalização de tema
    novo_tema = {
        'background_color': '#2b2b2b',
        'grid_color': '#3e3e3e',
        'text_color': '#ffffff',
        'axis_color': '#b0b0b0',
        'title_color': '#ffffff',
        'colors': ['#ff6b6b', '#48dbfb', '#feca57', '#1dd1a1', '#c9d1f6', '#54a0ff']
    }
    
    charts.set_theme(novo_tema)
    
    # Gráfico com o novo tema
    fig_theme = charts.line(
        x=meses, 
        y=dados, 
        title="Gráfico com Tema Personalizado"
    )
    fig_theme.show()