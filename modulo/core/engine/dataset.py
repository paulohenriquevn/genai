"""
Módulo para representação de datasets com metadados e análise automática.
"""

import pandas as pd
from typing import Dict, List, Any, Optional, Union

# Importação do analisador de datasets
from utils.dataset_analyzer import DatasetAnalyzer


class Dataset:
    """
    Representa um dataset com metadados e descrição para uso no motor de análise.
    Inclui análise automática de estrutura e relacionamentos.
    """
    
    def __init__(
        self, 
        dataframe: pd.DataFrame, 
        name: str, 
        description: str = "", 
        schema: Dict[str, str] = None,
        auto_analyze: bool = True
    ):
        """
        Inicializa um objeto Dataset.
        
        Args:
            dataframe: DataFrame Pandas com os dados
            name: Nome do dataset
            description: Descrição do conjunto de dados
            schema: Dicionário de metadados sobre as colunas (opcional)
            auto_analyze: Se True, faz análise automática da estrutura do dataset
        """
        self.dataframe = dataframe
        self.name = name
        self.description = description
        self.schema = schema or {}
        self.analyzed_metadata = {}
        self.primary_key = None
        self.column_types = {}
        self.potential_foreign_keys = []
        
        # Analisar automaticamente se solicitado
        if auto_analyze:
            self._analyze_structure()
    
    def _analyze_structure(self):
        """
        Analisa a estrutura do dataset para detectar metadados importantes.
        """
        # Usa o DatasetAnalyzer para obter metadados detalhados
        analyzer = DatasetAnalyzer()
        analyzer.add_dataset(self.name, self.dataframe)
        analysis_result = analyzer.analyze_all()
        
        # Extrai os metadados do dataset analisado
        if self.name in analysis_result.get("metadata", {}):
            dataset_meta = analysis_result["metadata"][self.name]
            
            # Armazena os metadados completos
            self.analyzed_metadata = dataset_meta
            
            # Extrai informações principais
            self.primary_key = dataset_meta.get("primary_key")
            self.potential_foreign_keys = dataset_meta.get("potential_foreign_keys", [])
            
            # Extrai tipos de dados sugeridos para cada coluna
            self.column_types = {}
            for col_name, col_meta in dataset_meta.get("columns", {}).items():
                self.column_types[col_name] = col_meta.get("suggested_type", "unknown")
    
    def to_json(self) -> Dict[str, Any]:
        """
        Converte o dataset para formato JSON.
        
        Returns:
            Dicionário serializado com metadados do dataset
        """
        # Metadados básicos
        result = {
            "name": self.name,
            "description": self.description,
            "records": len(self.dataframe),
            "columns": [],
            "schema": self.schema
        }
        
        # Adiciona informações de colunas
        for column in self.dataframe.columns:
            col_info = {
                "name": column,
                "type": str(self.dataframe[column].dtype)
            }
            
            # Adiciona tipo deduzido, se disponível
            if column in self.column_types:
                col_info["suggested_type"] = self.column_types[column]
            
            # Adiciona alguns valores de exemplo
            sample_values = self.dataframe[column].head(3).tolist()
            col_info["sample_values"] = [str(v) if pd.notna(v) else None for v in sample_values]
            
            # Estatísticas rápidas
            try:
                if pd.api.types.is_numeric_dtype(self.dataframe[column]):
                    col_info["stats"] = {
                        "min": float(self.dataframe[column].min()),
                        "max": float(self.dataframe[column].max()),
                        "mean": float(self.dataframe[column].mean()),
                        "unique_count": int(self.dataframe[column].nunique())
                    }
                else:
                    col_info["stats"] = {
                        "unique_count": int(self.dataframe[column].nunique()),
                        "most_common": str(self.dataframe[column].value_counts().index[0]) 
                                       if len(self.dataframe[column].value_counts()) > 0 else None
                    }
            except:
                # Em caso de erro, apenas ignora as estatísticas
                pass
            
            # Adiciona à lista de colunas
            result["columns"].append(col_info)
        
        # Adiciona informações da chave primária, se detectada
        if self.primary_key:
            result["primary_key"] = self.primary_key
        
        # Adiciona chaves estrangeiras potenciais
        if self.potential_foreign_keys:
            result["potential_foreign_keys"] = self.potential_foreign_keys
        
        # Adiciona relacionamentos, se disponíveis
        relationships = {}
        if "relationships" in self.analyzed_metadata:
            relationships = self.analyzed_metadata["relationships"]
            
        if relationships:
            result["relationships"] = relationships
            
        return result
        
    def serialize_dataframe(self) -> Dict[str, Any]:
        """
        Serializa o dataframe para uso no prompt template.
        Método requerido pela integração com o template de prompt.
        
        Returns:
            Dict com informações do dataframe
        """
        return {
            "name": self.name,
            "description": self.description,
            "dataframe": self.dataframe
        }