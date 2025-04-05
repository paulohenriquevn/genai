"""
Utilidades para conversão de dados para diferentes formatos de gráficos.
"""
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Optional, Any, Union


class ApexChartsConverter:
    """
    Classe para converter dados de pandas DataFrames para o formato JSON 
    compatível com a biblioteca ApexCharts para visualizações interativas.
    """
    
    @staticmethod
    def convert_line_chart(
        df: pd.DataFrame, 
        x: str, 
        y: Union[str, List[str]], 
        title: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Converte dados para o formato de gráfico de linha do ApexCharts.
        
        Args:
            df: DataFrame com os dados
            x: Nome da coluna para o eixo X
            y: Nome da coluna ou lista de colunas para o eixo Y
            title: Título do gráfico (opcional)
            options: Opções adicionais de customização (opcional)
            
        Returns:
            Configuração JSON para ApexCharts
        """
        # Configuração básica
        config = {
            "chart": {
                "type": "line",
                "height": 380,
                "zoom": {
                    "enabled": True
                }
            },
            "xaxis": {
                "categories": df[x].tolist() if not pd.api.types.is_datetime64_any_dtype(df[x]) else 
                              [str(d) for d in df[x].dt.strftime('%Y-%m-%d').tolist()],
                "title": {
                    "text": x
                }
            },
            "yaxis": {
                "title": {
                    "text": y if isinstance(y, str) else "Valores"
                }
            },
            "tooltip": {
                "enabled": True,
                "shared": True
            },
            "legend": {
                "position": "top"
            },
            "responsive": [
                {
                    "breakpoint": 768,
                    "options": {
                        "chart": {
                            "height": 300
                        }
                    }
                }
            ]
        }
        
        # Adiciona título se fornecido
        if title:
            config["title"] = {"text": title}
        
        # Configura séries de dados
        if isinstance(y, str):
            # Caso de uma única série
            config["series"] = [{
                "name": y,
                "data": df[y].tolist()
            }]
        else:
            # Caso de múltiplas séries
            config["series"] = [
                {
                    "name": col,
                    "data": df[col].tolist()
                } for col in y if col in df.columns
            ]
        
        # Aplica opções personalizadas se fornecidas
        if options:
            ApexChartsConverter._apply_custom_options(config, options)
        
        return config
    
    @staticmethod
    def convert_bar_chart(
        df: pd.DataFrame, 
        x: str, 
        y: Union[str, List[str]], 
        title: Optional[str] = None,
        stacked: bool = False,
        horizontal: bool = False,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Converte dados para o formato de gráfico de barras do ApexCharts.
        
        Args:
            df: DataFrame com os dados
            x: Nome da coluna para o eixo X
            y: Nome da coluna ou lista de colunas para o eixo Y
            title: Título do gráfico (opcional)
            stacked: Se True, empilha as barras (opcional)
            horizontal: Se True, cria um gráfico horizontal (opcional)
            options: Opções adicionais de customização (opcional)
            
        Returns:
            Configuração JSON para ApexCharts
        """
        # Configuração básica
        config = {
            "chart": {
                "type": "bar",
                "height": 380,
                "stacked": stacked
            },
            "plotOptions": {
                "bar": {
                    "horizontal": horizontal,
                    "columnWidth": "70%",
                    "dataLabels": {
                        "position": "top" if not horizontal else "center"
                    }
                }
            },
            "xaxis": {
                "categories": df[x].tolist() if not pd.api.types.is_datetime64_any_dtype(df[x]) else 
                              [str(d) for d in df[x].dt.strftime('%Y-%m-%d').tolist()],
                "title": {
                    "text": x
                }
            },
            "yaxis": {
                "title": {
                    "text": y if isinstance(y, str) else "Valores"
                }
            },
            "dataLabels": {
                "enabled": True
            },
            "tooltip": {
                "enabled": True,
                "shared": True
            },
            "legend": {
                "position": "top"
            }
        }
        
        # Adiciona título se fornecido
        if title:
            config["title"] = {"text": title}
        
        # Configura séries de dados
        if isinstance(y, str):
            # Caso de uma única série
            config["series"] = [{
                "name": y,
                "data": df[y].tolist()
            }]
        else:
            # Caso de múltiplas séries
            config["series"] = [
                {
                    "name": col,
                    "data": df[col].tolist()
                } for col in y if col in df.columns
            ]
        
        # Aplica opções personalizadas se fornecidas
        if options:
            ApexChartsConverter._apply_custom_options(config, options)
        
        return config
    
    @staticmethod
    def convert_pie_chart(
        df: pd.DataFrame, 
        labels: str, 
        values: str, 
        title: Optional[str] = None,
        donut: bool = False,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Converte dados para o formato de gráfico de pizza ou donut do ApexCharts.
        
        Args:
            df: DataFrame com os dados
            labels: Nome da coluna para os rótulos
            values: Nome da coluna para os valores
            title: Título do gráfico (opcional)
            donut: Se True, cria um gráfico de donut ao invés de pizza (opcional)
            options: Opções adicionais de customização (opcional)
            
        Returns:
            Configuração JSON para ApexCharts
        """
        # Preparação dos dados
        # Agrupa por labels se houver duplicatas
        if df[labels].duplicated().any():
            data_df = df.groupby(labels)[values].sum().reset_index()
        else:
            data_df = df[[labels, values]].copy()
        
        # Configuração básica
        config = {
            "chart": {
                "type": "donut" if donut else "pie",
                "height": 380
            },
            "labels": data_df[labels].tolist(),
            "series": data_df[values].tolist(),
            "responsive": [
                {
                    "breakpoint": 480,
                    "options": {
                        "chart": {
                            "width": 300
                        },
                        "legend": {
                            "position": "bottom"
                        }
                    }
                }
            ],
            "legend": {
                "position": "right"
            },
            "dataLabels": {
                "enabled": True
            }
        }
        
        # Configuração específica para donut
        if donut:
            config["plotOptions"] = {
                "pie": {
                    "donut": {
                        "size": "50%",
                        "labels": {
                            "show": True,
                            "name": {
                                "show": True
                            },
                            "value": {
                                "show": True
                            },
                            "total": {
                                "show": True,
                                "label": "Total",
                                "formatter": "function(w) { return w.globals.seriesTotals.reduce((a, b) => a + b, 0) }"
                            }
                        }
                    }
                }
            }
        
        # Adiciona título se fornecido
        if title:
            config["title"] = {"text": title}
        
        # Aplica opções personalizadas se fornecidas
        if options:
            ApexChartsConverter._apply_custom_options(config, options)
        
        return config
    
    @staticmethod
    def convert_scatter_chart(
        df: pd.DataFrame, 
        x: str, 
        y: str, 
        size_col: Optional[str] = None,
        group_col: Optional[str] = None,
        title: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Converte dados para o formato de gráfico de dispersão do ApexCharts.
        
        Args:
            df: DataFrame com os dados
            x: Nome da coluna para o eixo X
            y: Nome da coluna para o eixo Y
            size_col: Nome da coluna para o tamanho dos pontos (opcional)
            group_col: Nome da coluna para agrupar os pontos (opcional)
            title: Título do gráfico (opcional)
            options: Opções adicionais de customização (opcional)
            
        Returns:
            Configuração JSON para ApexCharts
        """
        # Configuração básica
        config = {
            "chart": {
                "type": "scatter",
                "height": 380,
                "zoom": {
                    "enabled": True,
                    "type": 'xy'
                }
            },
            "xaxis": {
                "title": {
                    "text": x
                }
            },
            "yaxis": {
                "title": {
                    "text": y
                }
            },
            "tooltip": {
                "enabled": True
            },
            "markers": {
                "size": [6]
            }
        }
        
        # Adiciona título se fornecido
        if title:
            config["title"] = {"text": title}
        
        # Cria séries de dados
        if group_col:
            # Agrupamento de pontos por categoria
            groups = df[group_col].unique()
            series = []
            
            for group in groups:
                group_df = df[df[group_col] == group]
                
                series_data = []
                for i, row in group_df.iterrows():
                    point = {"x": row[x], "y": row[y]}
                    if size_col:
                        point["z"] = row[size_col]
                    series_data.append(point)
                
                series.append({
                    "name": str(group),
                    "data": series_data
                })
            
            config["series"] = series
            
        else:
            # Sem agrupamento
            series_data = []
            for i, row in df.iterrows():
                point = {"x": row[x], "y": row[y]}
                if size_col:
                    point["z"] = row[size_col]
                series_data.append(point)
            
            config["series"] = [{
                "name": f"{x} vs {y}",
                "data": series_data
            }]
        
        # Configura tamanhos variáveis se especificado
        if size_col:
            config["markers"] = {
                "size": [4, 16],  # min e max tamanhos
                "sizeMode": 'diameter'
            }
            
            # Identificar min/max para normalização
            min_size = df[size_col].min()
            max_size = df[size_col].max()
            
            if min_size != max_size:
                config["bubble"] = {
                    "minBubbleRadius": 4,
                    "maxBubbleRadius": 16
                }
        
        # Aplica opções personalizadas se fornecidas
        if options:
            ApexChartsConverter._apply_custom_options(config, options)
        
        return config
    
    @staticmethod
    def convert_area_chart(
        df: pd.DataFrame, 
        x: str, 
        y: Union[str, List[str]], 
        title: Optional[str] = None,
        stacked: bool = False,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Converte dados para o formato de gráfico de área do ApexCharts.
        
        Args:
            df: DataFrame com os dados
            x: Nome da coluna para o eixo X
            y: Nome da coluna ou lista de colunas para o eixo Y
            title: Título do gráfico (opcional)
            stacked: Se True, empilha as áreas (opcional)
            options: Opções adicionais de customização (opcional)
            
        Returns:
            Configuração JSON para ApexCharts
        """
        # Configuração básica
        config = {
            "chart": {
                "type": "area",
                "height": 380,
                "stacked": stacked
            },
            "dataLabels": {
                "enabled": False
            },
            "fill": {
                "type": "gradient",
                "gradient": {
                    "opacityFrom": 0.6,
                    "opacityTo": 0.1
                }
            },
            "stroke": {
                "curve": "smooth",
                "width": 2
            },
            "xaxis": {
                "categories": df[x].tolist() if not pd.api.types.is_datetime64_any_dtype(df[x]) else 
                              [str(d) for d in df[x].dt.strftime('%Y-%m-%d').tolist()],
                "title": {
                    "text": x
                }
            },
            "yaxis": {
                "title": {
                    "text": y if isinstance(y, str) else "Valores"
                }
            },
            "tooltip": {
                "enabled": True,
                "shared": True
            },
            "legend": {
                "position": "top"
            }
        }
        
        # Adiciona título se fornecido
        if title:
            config["title"] = {"text": title}
        
        # Configura séries de dados
        if isinstance(y, str):
            # Caso de uma única série
            config["series"] = [{
                "name": y,
                "data": df[y].tolist()
            }]
        else:
            # Caso de múltiplas séries
            config["series"] = [
                {
                    "name": col,
                    "data": df[col].tolist()
                } for col in y if col in df.columns
            ]
        
        # Aplica opções personalizadas se fornecidas
        if options:
            ApexChartsConverter._apply_custom_options(config, options)
        
        return config
    
    @staticmethod
    def convert_heatmap(
        df: pd.DataFrame, 
        x: str, 
        y: str, 
        values: str,
        title: Optional[str] = None,
        color_scale: Optional[List[str]] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Converte dados para o formato de mapa de calor do ApexCharts.
        
        Args:
            df: DataFrame com os dados
            x: Nome da coluna para o eixo X
            y: Nome da coluna para o eixo Y
            values: Nome da coluna com os valores para coloração
            title: Título do gráfico (opcional)
            color_scale: Lista de cores para a escala (opcional)
            options: Opções adicionais de customização (opcional)
            
        Returns:
            Configuração JSON para ApexCharts
        """
        # Preparação dos dados
        # Precisamos pivotar o DataFrame para o formato desejado
        pivot_df = df.pivot(index=y, columns=x, values=values).reset_index()
        
        # Lista de categorias das linhas (eixo Y)
        y_categories = pivot_df[y].tolist()
        
        # Lista de categorias das colunas (eixo X)
        x_categories = list(pivot_df.columns[1:])  # todas exceto a primeira (que é o índice pivotado)
        
        # Dados da série
        series_data = []
        for idx, row in pivot_df.iterrows():
            for col in x_categories:
                value = row[col]
                if not pd.isna(value):
                    series_data.append({
                        "x": col,
                        "y": row[y],
                        "value": value
                    })
        
        # Configuração básica
        config = {
            "chart": {
                "type": "heatmap",
                "height": 400
            },
            "dataLabels": {
                "enabled": True
            },
            "colors": color_scale or ["#008FFB", "#00E396", "#FEB019", "#FF4560", "#775DD0"],
            "series": [{
                "name": values,
                "data": series_data
            }],
            "xaxis": {
                "categories": x_categories,
                "title": {
                    "text": x
                }
            },
            "yaxis": {
                "categories": y_categories,
                "title": {
                    "text": y
                }
            },
            "plotOptions": {
                "heatmap": {
                    "radius": 0,
                    "enableShades": True,
                    "shadeIntensity": 0.5,
                    "colorScale": {
                        "ranges": []  # Será preenchido automaticamente pelo ApexCharts
                    }
                }
            }
        }
        
        # Adiciona título se fornecido
        if title:
            config["title"] = {"text": title}
        
        # Aplica opções personalizadas se fornecidas
        if options:
            ApexChartsConverter._apply_custom_options(config, options)
        
        return config
    
    @staticmethod
    def convert_radar_chart(
        df: pd.DataFrame, 
        categories: str, 
        series: Union[str, List[str]], 
        title: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Converte dados para o formato de gráfico de radar do ApexCharts.
        
        Args:
            df: DataFrame com os dados
            categories: Nome da coluna para as categorias (eixos do radar)
            series: Nome da coluna ou lista de colunas para os valores
            title: Título do gráfico (opcional)
            options: Opções adicionais de customização (opcional)
            
        Returns:
            Configuração JSON para ApexCharts
        """
        # Configuração básica
        config = {
            "chart": {
                "type": "radar",
                "height": 400,
                "toolbar": {
                    "show": True
                }
            },
            "xaxis": {
                "categories": df[categories].tolist()
            },
            "markers": {
                "size": 4,
                "hover": {
                    "size": 6
                }
            },
            "tooltip": {
                "enabled": True
            },
            "legend": {
                "position": "top"
            }
        }
        
        # Adiciona título se fornecido
        if title:
            config["title"] = {"text": title}
        
        # Configura séries de dados
        if isinstance(series, str):
            # Caso de uma única série
            config["series"] = [{
                "name": series,
                "data": df[series].tolist()
            }]
        else:
            # Caso de múltiplas séries
            config["series"] = [
                {
                    "name": col,
                    "data": df[col].tolist()
                } for col in series if col in df.columns
            ]
        
        # Aplica opções personalizadas se fornecidas
        if options:
            ApexChartsConverter._apply_custom_options(config, options)
        
        return config
    
    @staticmethod
    def _apply_custom_options(config: Dict[str, Any], options: Dict[str, Any]) -> None:
        """
        Aplica opções personalizadas à configuração do gráfico.
        
        Args:
            config: Configuração base do gráfico
            options: Opções personalizadas a serem aplicadas
        """
        # Função recursiva para mesclar dicionários
        def deep_merge(source, destination):
            for key, value in source.items():
                if key in destination and isinstance(destination[key], dict) and isinstance(value, dict):
                    deep_merge(value, destination[key])
                else:
                    destination[key] = value
        
        # Mescla as opções com o config base
        deep_merge(options, config)