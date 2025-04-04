"""
DataFrameWrapper Module
======================

Este módulo fornece uma classe para encapsular DataFrames do pandas
com metadados, informações semânticas e outros atributos.
"""

import pandas as pd
from typing import Dict, List, Any, Optional, Union


class DataFrameWrapper:
    """
    Classe que encapsula um DataFrame do pandas com metadados adicionais.
    
    Esta classe fornece funcionalidades para:
    - Armazenar metadados sobre o DataFrame
    - Rastrear a origem e transformações dos dados
    - Associar informações semânticas às colunas
    """
    
    def __init__(
        self, 
        dataframe: pd.DataFrame, 
        name: str, 
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        source: Optional[str] = None
    ):
        """
        Inicializa um wrapper para DataFrame.
        
        Args:
            dataframe: DataFrame do pandas a ser encapsulado
            name: Nome do DataFrame
            description: Descrição opcional dos dados
            metadata: Metadados adicionais como dicionário
            source: Fonte original dos dados
        """
        self.dataframe = dataframe
        self.name = name
        self.description = description or f"DataFrame {name}"
        self.metadata = metadata or {}
        self.source = source or "unknown"
        
        # Atributos derivados
        self.shape = dataframe.shape
        self.column_types = {col: str(dtype) for col, dtype in dataframe.dtypes.items()}
        
        # Rastreamento de operações
        self.operations_history = []
    
    def __str__(self) -> str:
        """Representação em string do wrapper"""
        return f"DataFrameWrapper: {self.name} ({self.shape[0]} rows, {self.shape[1]} columns)"
    
    def __repr__(self) -> str:
        """Representação detalhada do wrapper"""
        return (f"DataFrameWrapper(name='{self.name}', "
                f"shape={self.shape}, "
                f"source='{self.source}', "
                f"columns={list(self.dataframe.columns)})")
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Retorna um resumo das propriedades do DataFrame.
        
        Returns:
            Dict com resumo dos dados
        """
        return {
            "name": self.name,
            "description": self.description,
            "rows": self.shape[0],
            "columns": list(self.dataframe.columns),
            "column_types": self.column_types,
            "source": self.source,
            "operations": len(self.operations_history)
        }
    
    def get_schema(self) -> Dict[str, Any]:
        """
        Retorna o esquema do DataFrame com tipos de dados.
        
        Returns:
            Dict com o esquema do DataFrame
        """
        schema = {
            "columns": []
        }
        
        for col in self.dataframe.columns:
            col_info = {
                "name": col,
                "type": str(self.dataframe[col].dtype),
                "nullable": self.dataframe[col].isna().any()
            }
            
            # Determina valores únicos se houver poucos
            nunique = self.dataframe[col].nunique()
            if nunique <= 10 and nunique > 0:  # Limite razoável para mostrar únicos
                uniques = self.dataframe[col].dropna().unique().tolist()
                col_info["unique_values"] = uniques
                col_info["nunique"] = nunique
            
            schema["columns"].append(col_info)
        
        return schema
    
    def add_operation(self, operation_name: str, details: Dict[str, Any] = None) -> None:
        """
        Adiciona uma operação ao histórico de operações.
        
        Args:
            operation_name: Nome da operação
            details: Detalhes da operação
        """
        operation = {
            "operation": operation_name,
            "details": details or {}
        }
        self.operations_history.append(operation)
    
    def get_preview(self, rows: int = 5) -> pd.DataFrame:
        """
        Retorna uma prévia do DataFrame.
        
        Args:
            rows: Número de linhas na prévia
            
        Returns:
            DataFrame com a prévia
        """
        return self.dataframe.head(rows)
    
    # Métodos delegados para operações comuns do pandas
    
    def query(self, expr: str) -> 'DataFrameWrapper':
        """
        Filtra o DataFrame usando a sintaxe de query do pandas.
        
        Args:
            expr: Expressão de consulta
            
        Returns:
            Novo DataFrameWrapper com o resultado
        """
        result = self.dataframe.query(expr)
        wrapper = DataFrameWrapper(
            result, 
            f"{self.name}_filtered",
            f"Filtered {self.name} where {expr}",
            self.metadata,
            self.source
        )
        wrapper.add_operation("query", {"expression": expr})
        wrapper.operations_history.extend(self.operations_history)
        return wrapper
    
    def select(self, columns: List[str]) -> 'DataFrameWrapper':
        """
        Seleciona colunas do DataFrame.
        
        Args:
            columns: Lista de colunas a selecionar
            
        Returns:
            Novo DataFrameWrapper com as colunas selecionadas
        """
        result = self.dataframe[columns]
        wrapper = DataFrameWrapper(
            result, 
            f"{self.name}_selected",
            f"Selected columns from {self.name}: {', '.join(columns)}",
            self.metadata,
            self.source
        )
        wrapper.add_operation("select", {"columns": columns})
        wrapper.operations_history.extend(self.operations_history)
        return wrapper
    
    def groupby(self, by: Union[str, List[str]], agg: Dict[str, str]) -> 'DataFrameWrapper':
        """
        Agrupa o DataFrame por colunas específicas.
        
        Args:
            by: Coluna(s) para agrupar
            agg: Dicionário de colunas e funções de agregação
            
        Returns:
            Novo DataFrameWrapper com o resultado agrupado
        """
        by_cols = [by] if isinstance(by, str) else by
        result = self.dataframe.groupby(by).agg(agg).reset_index()
        
        wrapper = DataFrameWrapper(
            result, 
            f"{self.name}_grouped",
            f"Grouped {self.name} by {', '.join(by_cols)}",
            self.metadata,
            self.source
        )
        wrapper.add_operation("groupby", {"by": by_cols, "aggregation": agg})
        wrapper.operations_history.extend(self.operations_history)
        return wrapper