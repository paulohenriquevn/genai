"""
Sistema de análise de dados em linguagem natural.

Este módulo fornece uma interface simples e unificada para o sistema 
de análise de dados usando linguagem natural.
"""

import pandas as pd
from typing import Dict, List, Optional, Any, Union

from core.engine.analysis_engine import AnalysisEngine
from core.response.base import BaseResponse


class NaturalLanguageQuerySystem:
    """
    Interface simplificada para o motor de análise de dados em linguagem natural.
    
    Esta classe fornece métodos fáceis de usar para:
    - Carregar dados
    - Executar consultas em linguagem natural
    - Processar feedback do usuário
    - Executar consultas SQL diretamente
    
    Exemplo de uso:
    ```python
    nlq = NaturalLanguageQuerySystem()
    nlq.load_data("vendas.csv", "vendas")
    result = nlq.ask("Quais foram as 5 maiores vendas do último mês?")
    print(result.get_value())
    ```
    """
    
    def __init__(
        self,
        model_type: str = "openai",
        model_name: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        """
        Inicializa o sistema de consulta em linguagem natural.
        
        Args:
            model_type: Tipo de modelo LLM a ser usado (openai, anthropic, etc.)
            model_name: Nome específico do modelo (opcional)
            api_key: Chave de API para o serviço LLM (opcional)
        """
        self.engine = AnalysisEngine(
            model_type=model_type,
            model_name=model_name,
            api_key=api_key
        )
    
    def load_data(
        self, 
        data: Union[pd.DataFrame, str], 
        name: str, 
        description: Optional[str] = None,
        schema: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Carrega um dataset no sistema.
        
        Args:
            data: DataFrame ou caminho para arquivo de dados
            name: Nome do dataset para referência
            description: Descrição do dataset (opcional)
            schema: Metadados das colunas (opcional)
        """
        self.engine.load_data(data, name, description, schema)
    
    def list_datasets(self) -> List[str]:
        """
        Lista os datasets carregados.
        
        Returns:
            Lista de nomes dos datasets disponíveis
        """
        return self.engine.list_datasets()
    
    def ask(self, query: str) -> BaseResponse:
        """
        Processa uma consulta em linguagem natural.
        
        Args:
            query: Consulta em linguagem natural
            
        Returns:
            Resposta processada (pode ser dataframe, gráfico, texto ou número)
        """
        return self.engine.process_query(query)
    
    def ask_with_feedback(self, query: str, feedback: str) -> BaseResponse:
        """
        Processa uma consulta com feedback adicional do usuário.
        
        Args:
            query: Consulta em linguagem natural
            feedback: Feedback do usuário para melhorar a resposta
            
        Returns:
            Resposta processada
        """
        return self.engine.process_query_with_feedback(query, feedback)
    
    def execute_sql(self, sql_query: str, dataset_name: Optional[str] = None) -> BaseResponse:
        """
        Executa uma consulta SQL diretamente.
        
        Args:
            sql_query: Consulta SQL para executar
            dataset_name: Nome do dataset (opcional se houver apenas um)
            
        Returns:
            Resposta da consulta SQL
        """
        return self.engine.execute_direct_query(sql_query, dataset_name)
    
    def generate_visualization(
        self, 
        data: Union[pd.DataFrame, pd.Series], 
        chart_type: str, 
        x: Optional[str] = None, 
        y: Optional[str] = None,
        title: Optional[str] = None
    ) -> BaseResponse:
        """
        Gera uma visualização a partir de dados.
        
        Args:
            data: DataFrame ou Series com os dados
            chart_type: Tipo de gráfico (bar, line, scatter, hist, pie)
            x: Coluna para eixo x (opcional)
            y: Coluna para eixo y (opcional)
            title: Título do gráfico (opcional)
            
        Returns:
            Resposta contendo o gráfico gerado
        """
        return self.engine.generate_chart(data, chart_type, x, y, title)