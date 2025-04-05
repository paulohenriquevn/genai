"""
Módulo para análise dinâmica de datasets.

Este módulo fornece funcionalidades para:
- Detectar tipos de dados de colunas
- Identificar chaves primárias/estrangeiras
- Mapear relações entre tabelas
- Gerar metadados para orientar a geração de queries
"""

import pandas as pd
import numpy as np
import json
import re
import logging
from typing import Dict, List, Any, Tuple, Optional, Set
from datetime import datetime

logger = logging.getLogger(__name__)


class DatasetAnalyzer:
    """
    Classe para analisar datasets e detectar metadados importantes.
    
    Esta classe fornece métodos para analisar automaticamente datasets,
    detectar tipos de dados, identificar relações entre tabelas e
    gerar metadados para uso em sistemas de consulta.
    """
    
    # Constantes para detecção de tipos
    DATE_PATTERNS = [
        r'^\d{4}-\d{2}-\d{2}$',  # yyyy-mm-dd
        r'^\d{2}/\d{2}/\d{4}$',  # dd/mm/yyyy
        r'^\d{2}-\d{2}-\d{4}$',  # dd-mm-yyyy
        r'^\d{4}/\d{2}/\d{2}$',  # yyyy/mm/dd
    ]
    
    # Nomes comuns para colunas de id
    ID_COLUMNS = [
        'id', 'key', 'codigo', 'code', 'pk', 'primary'
    ]
    
    # Sufixos comuns para colunas que são chaves estrangeiras
    FK_SUFFIXES = [
        '_id', '_fk', '_key', '_codigo', '_code', '_ref'
    ]
    
    # Prefixos comuns para colunas de data
    DATE_PREFIXES = [
        'dt_', 'data_', 'date_', 'dt', 'data', 'date'
    ]
    
    def __init__(self, sample_size: int = 1000, confidence_threshold: float = 0.8):
        """
        Inicializa o analisador de datasets.
        
        Args:
            sample_size: Tamanho da amostra para análise de datasets muito grandes
            confidence_threshold: Limiar de confiança para detecção de tipos (0.0-1.0)
        """
        self.sample_size = sample_size
        self.confidence_threshold = confidence_threshold
        self.datasets = {}
        self.metadata = {}
        self.relationships = []
    
    def add_dataset(self, name: str, df: pd.DataFrame) -> None:
        """
        Adiciona um dataset para análise.
        
        Args:
            name: Nome único para o dataset
            df: DataFrame Pandas com os dados
        """
        # Armazena uma cópia para evitar modificações externas
        self.datasets[name] = df.copy()
        logger.info(f"Dataset '{name}' adicionado com {len(df)} linhas e {len(df.columns)} colunas")
    
    def analyze_all(self) -> Dict[str, Any]:
        """
        Analisa todos os datasets e gera metadados completos.
        
        Returns:
            Dict com todos os metadados detectados
        """
        start_time = datetime.now()
        logger.info("Iniciando análise completa de todos os datasets")
        
        # Analisa cada dataset individualmente
        for name, df in self.datasets.items():
            self.metadata[name] = self._analyze_dataset(name, df)
        
        # Detecta relações entre datasets
        self._detect_relationships()
        
        # Adiciona as relações aos metadados
        self._add_relationships_to_metadata()
        
        # Cria o resultado final
        result = {
            "metadata": self.metadata,
            "relationships": self.relationships,
            "analysis_summary": {
                "total_datasets": len(self.datasets),
                "total_columns": sum(len(df.columns) for df in self.datasets.values()),
                "total_rows": sum(len(df) for df in self.datasets.values()),
                "total_relationships": len(self.relationships)
            }
        }
        
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Análise completa concluída em {duration:.2f} segundos")
        
        return result
    
    def _analyze_dataset(self, name: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analisa um único dataset para detectar seus metadados.
        
        Args:
            name: Nome do dataset
            df: DataFrame Pandas
            
        Returns:
            Dict com metadados detectados
        """
        logger.info(f"Analisando dataset '{name}'")
        
        # Prepara o resultado
        result = {
            "name": name,
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": {},
            "primary_key": None,
            "primary_key_confidence": 0.0,
            "potential_foreign_keys": []
        }
        
        # Se dataset for muito grande, usa uma amostra
        sample_df = df.sample(min(self.sample_size, len(df))) if len(df) > self.sample_size else df
        
        # Detecta tipos e características de cada coluna
        for column in df.columns:
            col_meta = self._analyze_column(df, column, sample_df)
            result["columns"][column] = col_meta
            
            # Verifica se é um candidato a chave primária
            if col_meta.get("potential_primary_key", False):
                pk_confidence = col_meta.get("uniqueness", 0.0) * col_meta.get("non_null", 0.0)
                
                if pk_confidence > result["primary_key_confidence"]:
                    result["primary_key"] = column
                    result["primary_key_confidence"] = pk_confidence
            
            # Verifica se é um candidato a chave estrangeira
            if col_meta.get("potential_foreign_key", False):
                result["potential_foreign_keys"].append(column)
        
        logger.info(f"Análise do dataset '{name}' concluída")
        return result
    
    def _analyze_column(self, df: pd.DataFrame, column: str, sample_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analisa uma única coluna para detectar tipo e características.
        
        Args:
            df: DataFrame completo
            column: Nome da coluna a ser analisada
            sample_df: DataFrame de amostra para análises mais pesadas
            
        Returns:
            Dict com metadados da coluna
        """
        # Dados básicos
        col_data = df[column]
        sample_data = sample_df[column]
        non_null_count = col_data.count()
        unique_count = col_data.nunique()
        
        # Estatísticas básicas
        stats = {
            "count": len(col_data),
            "non_null": non_null_count / len(col_data) if len(col_data) > 0 else 0,
            "uniqueness": unique_count / non_null_count if non_null_count > 0 else 0,
            "unique_count": unique_count,
            "dtype": str(col_data.dtype)
        }
        
        # Metadados iniciais da coluna
        result = {
            "name": column,
            "suggested_type": self._detect_column_type(col_data, column),
            "nullable": col_data.isna().any(),
            "stats": stats,
            "potential_primary_key": False,
            "potential_foreign_key": False
        }
        
        # Verifica se é uma potencial chave primária
        # Critérios: nome sugestivo, alta unicidade, poucos nulos
        result["potential_primary_key"] = self._is_potential_primary_key(column, stats)
        
        # Verifica se é uma potencial chave estrangeira
        # Critérios: nome sugestivo (_id, _fk, etc), tipo compatível
        result["potential_foreign_key"] = self._is_potential_foreign_key(column, result["suggested_type"])
        
        # Adiciona estatísticas específicas para certos tipos de dados
        if result["suggested_type"] == "numeric":
            if len(sample_data) > 0 and sample_data.notna().any():
                numeric_stats = {
                    "min": float(sample_data.min()) if pd.notna(sample_data.min()) else None,
                    "max": float(sample_data.max()) if pd.notna(sample_data.max()) else None,
                    "mean": float(sample_data.mean()) if pd.notna(sample_data.mean()) else None,
                    "median": float(sample_data.median()) if pd.notna(sample_data.median()) else None
                }
                result["numeric_stats"] = numeric_stats
        
        elif result["suggested_type"] == "date":
            # Tenta detectar o formato de data
            result["date_format"] = self._detect_date_format(col_data)
            
            # Estatísticas temporais se possível
            try:
                date_data = pd.to_datetime(sample_data, errors='coerce')
                if date_data.notna().any():
                    result["temporal_stats"] = {
                        "min_date": date_data.min().strftime('%Y-%m-%d') if pd.notna(date_data.min()) else None,
                        "max_date": date_data.max().strftime('%Y-%m-%d') if pd.notna(date_data.max()) else None,
                        "range_days": (date_data.max() - date_data.min()).days if pd.notna(date_data.min()) and pd.notna(date_data.max()) else None
                    }
            except:
                pass
        
        elif result["suggested_type"] == "categorical":
            # Encontrar os valores mais comuns
            value_counts = sample_data.value_counts(normalize=True)
            top_values = value_counts.head(10).to_dict()
            result["top_values"] = {str(k): float(v) for k, v in top_values.items()}
        
        return result
    
    def _detect_column_type(self, col_data: pd.Series, column_name: str) -> str:
        """
        Detecta o tipo de dados de uma coluna.
        
        Args:
            col_data: Série Pandas com os dados da coluna
            column_name: Nome da coluna
            
        Returns:
            Tipo sugerido (string, numeric, date, boolean, categorical, id)
        """
        # Elimina valores nulos para análise
        non_null_data = col_data.dropna()
        
        # Se não houver dados, não conseguimos determinar
        if len(non_null_data) == 0:
            return "unknown"
        
        # Se já for tipo numérico
        if pd.api.types.is_numeric_dtype(non_null_data):
            # Verifica se parece ser um ID
            if self._is_id_column(column_name, non_null_data):
                return "id"
            # Verifica se parece ser boolean (0/1)
            elif set(non_null_data.unique()).issubset({0, 1}):
                return "boolean"
            else:
                return "numeric"
        
        # Verifica se é datetime ou timedelta
        elif pd.api.types.is_datetime64_any_dtype(non_null_data) or pd.api.types.is_timedelta64_dtype(non_null_data):
            return "date"
        
        # Se for object ou string, precisamos fazer mais análises
        else:
            # Converte para string para análise
            str_data = non_null_data.astype(str)
            
            # Verifica se parece ser uma data
            if self._is_date_column(str_data, column_name):
                return "date"
            
            # Verifica se parece ser boolean (true/false, sim/não)
            elif self._is_boolean_column(str_data):
                return "boolean"
            
            # Verifica se tem poucas categorias únicas
            elif len(str_data.unique()) < 20 and len(str_data.unique()) / len(str_data) < 0.1:
                return "categorical"
            
            # Verifica se é um ID
            elif self._is_id_column(column_name, str_data):
                return "id"
            
            # Caso contrário, é string geral
            else:
                return "string"
    
    def _is_date_column(self, data: pd.Series, column_name: str) -> bool:
        """
        Verifica se uma coluna parece conter datas.
        
        Args:
            data: Série Pandas com os dados
            column_name: Nome da coluna
            
        Returns:
            True se parece ser coluna de data, False caso contrário
        """
        # Verifica nome da coluna primeiro
        for prefix in self.DATE_PREFIXES:
            if column_name.lower().startswith(prefix):
                return True
        
        # Analisa por padrões
        sample = data.sample(min(100, len(data)))
        pattern_matches = 0
        
        for value in sample:
            for pattern in self.DATE_PATTERNS:
                if re.match(pattern, str(value)):
                    pattern_matches += 1
                    break
        
        return pattern_matches / len(sample) >= self.confidence_threshold
    
    def _detect_date_format(self, data: pd.Series) -> str:
        """
        Detecta o formato de data mais provável.
        
        Args:
            data: Série Pandas com os dados
            
        Returns:
            String com o formato de data detectado
        """
        # Formatos comuns para tentar
        formats = [
            '%Y-%m-%d',  # 2023-01-15
            '%d/%m/%Y',  # 15/01/2023
            '%m/%d/%Y',  # 01/15/2023
            '%d-%m-%Y',  # 15-01-2023
            '%Y/%m/%d',  # 2023/01/15
            '%d.%m.%Y',  # 15.01.2023
        ]
        
        # Tenta identificar o formato com base na amostra
        sample = data.dropna().astype(str).sample(min(10, len(data.dropna())))
        
        for date_format in formats:
            success = 0
            for value in sample:
                try:
                    datetime.strptime(value, date_format)
                    success += 1
                except:
                    pass
            
            if success / len(sample) >= 0.8:
                return date_format
        
        # Se não conseguir identificar, retorna formato genérico
        return '%Y-%m-%d'
    
    def _is_boolean_column(self, data: pd.Series) -> bool:
        """
        Verifica se uma coluna parece conter dados booleanos.
        
        Args:
            data: Série Pandas com os dados
            
        Returns:
            True se parece ser coluna booleana, False caso contrário
        """
        # Valores típicos para dados booleanos
        boolean_sets = [
            {'true', 'false'},
            {'t', 'f'},
            {'sim', 'não', 'nao'},
            {'yes', 'no'},
            {'y', 'n'},
            {'s', 'n'},
            {'1', '0'},
            {'ativo', 'inativo'},
            {'active', 'inactive'},
            {'ativado', 'desativado'},
            {'enabled', 'disabled'}
        ]
        
        # Converte para lowercase para comparação
        lower_values = set(data.str.lower())
        
        # Verifica se os valores únicos são um dos conjuntos booleanos
        for boolean_set in boolean_sets:
            if lower_values.issubset(boolean_set):
                return True
        
        return False
    
    def _is_id_column(self, column_name: str, data: pd.Series) -> bool:
        """
        Verifica se uma coluna parece ser um identificador.
        
        Args:
            column_name: Nome da coluna
            data: Série Pandas com os dados
            
        Returns:
            True se parece ser coluna ID, False caso contrário
        """
        # Verifica nome da coluna
        column_lower = column_name.lower()
        
        # Verifica se o nome é exatamente um dos nomes comuns de ID
        for id_name in self.ID_COLUMNS:
            if column_lower == id_name:
                return True
        
        # Verifica se começa com "id_" ou termina com "_id"
        if column_lower.startswith('id_') or column_lower.endswith('_id'):
            return True
        
        # Verifica características dos dados para IDs numéricos
        if pd.api.types.is_numeric_dtype(data):
            # Se for uma série de números inteiros começando em 1 ou 0
            # e incrementando, provavelmente é um ID
            if all(data.sort_values().diff().dropna() == 1):
                # Verifica se começa em 0 ou 1
                return data.min() in [0, 1]
        
        return False
    
    def _is_potential_primary_key(self, column_name: str, stats: Dict[str, Any]) -> bool:
        """
        Verifica se uma coluna é potencial chave primária.
        
        Args:
            column_name: Nome da coluna
            stats: Estatísticas da coluna
            
        Returns:
            True se é potencial chave primária, False caso contrário
        """
        column_lower = column_name.lower()
        
        # Critérios de nome para chave primária
        name_is_pk = any(column_lower == pk_name for pk_name in self.ID_COLUMNS) or column_lower.endswith('_pk')
        
        # Critérios de unicidade e não-nulidade
        high_uniqueness = stats.get("uniqueness", 0) > 0.99
        no_nulls = stats.get("non_null", 0) == 1.0
        
        # Critérios combinados: nome sugestivo ou alta unicidade sem nulos
        return (name_is_pk or (high_uniqueness and no_nulls))
    
    def _is_potential_foreign_key(self, column_name: str, suggested_type: str) -> bool:
        """
        Verifica se uma coluna é potencial chave estrangeira.
        
        Args:
            column_name: Nome da coluna
            suggested_type: Tipo sugerido para a coluna
            
        Returns:
            True se é potencial chave estrangeira, False caso contrário
        """
        column_lower = column_name.lower()
        
        # Verificar sufixos comuns para chaves estrangeiras
        has_fk_suffix = any(column_lower.endswith(suffix) for suffix in self.FK_SUFFIXES)
        
        # Tipo deve ser compatível com IDs (numérico ou string)
        compatible_type = suggested_type in ['id', 'numeric', 'string']
        
        return has_fk_suffix and compatible_type
    
    def _detect_relationships(self) -> None:
        """
        Detecta relacionamentos entre datasets com base em análise de chaves.
        """
        logger.info("Detectando relacionamentos entre datasets")
        
        # Limpa relacionamentos anteriores
        self.relationships = []
        
        # Lista de todas as chaves primárias
        pk_columns = {}
        for ds_name, ds_meta in self.metadata.items():
            if ds_meta.get("primary_key") and ds_meta.get("primary_key_confidence", 0) > 0.9:
                pk_columns[ds_name] = {
                    "column": ds_meta["primary_key"],
                    "dataset": ds_name
                }
        
        # Procura por correspondências entre FKs e PKs
        for source_name, source_meta in self.metadata.items():
            for fk_column in source_meta.get("potential_foreign_keys", []):
                # Nome sem sufixo (_id, etc) da possível tabela referenciada
                for suffix in self.FK_SUFFIXES:
                    if fk_column.lower().endswith(suffix):
                        base_name = fk_column[:-len(suffix)]
                        
                        # Procura datasets com nome semelhante
                        for target_name in self.datasets.keys():
                            # Verifica se o nome do dataset referenciado é similar ao base_name
                            if self._names_match(base_name, target_name):
                                # Verifica se o dataset alvo tem uma chave primária
                                if target_name in pk_columns:
                                    self.relationships.append({
                                        "source_dataset": source_name,
                                        "source_column": fk_column,
                                        "target_dataset": target_name,
                                        "target_column": pk_columns[target_name]["column"],
                                        "relationship_type": "many_to_one",
                                        "confidence": 0.9
                                    })
                                    logger.info(f"Relação detectada: {source_name}.{fk_column} -> {target_name}.{pk_columns[target_name]['column']}")
                                    break
        
        # Verifica sobreposição de valores para relações não detectadas por nome
        self._detect_relationships_by_value_overlap()
    
    def _detect_relationships_by_value_overlap(self, min_overlap: float = 0.8) -> None:
        """
        Detecta relacionamentos baseados na sobreposição de valores.
        
        Args:
            min_overlap: Sobreposição mínima para considerar relacionamento
        """
        # Identificação de colunas candidatas
        for source_name, source_meta in self.metadata.items():
            # Pega colunas potenciais chaves estrangeiras do source
            for source_col in source_meta.get("potential_foreign_keys", []):
                source_col_data = self.datasets[source_name][source_col].dropna()
                source_values = set(source_col_data)
                
                # Pula se não houver valores suficientes
                if len(source_values) < 5:
                    continue
                
                # Compara com chaves primárias de outros datasets
                for target_name, target_meta in self.metadata.items():
                    # Não comparar com o mesmo dataset
                    if source_name == target_name:
                        continue
                    
                    # Verifica a chave primária do dataset alvo
                    target_pk = target_meta.get("primary_key")
                    if not target_pk:
                        continue
                    
                    # Obtém valores da coluna PK do target
                    target_col_data = self.datasets[target_name][target_pk].dropna()
                    target_values = set(target_col_data)
                    
                    # Calcula a sobreposição de valores
                    if len(target_values) > 0:
                        # Quantos valores do source existem no target
                        overlap = len(source_values.intersection(target_values)) / len(source_values)
                        
                        if overlap >= min_overlap:
                            # Verificar se esta relação já foi detectada
                            relationship_exists = False
                            for rel in self.relationships:
                                if (rel["source_dataset"] == source_name and 
                                    rel["source_column"] == source_col and
                                    rel["target_dataset"] == target_name and
                                    rel["target_column"] == target_pk):
                                    relationship_exists = True
                                    break
                            
                            if not relationship_exists:
                                self.relationships.append({
                                    "source_dataset": source_name,
                                    "source_column": source_col,
                                    "target_dataset": target_name,
                                    "target_column": target_pk,
                                    "relationship_type": "many_to_one",
                                    "confidence": overlap,
                                    "detection_method": "value_overlap"
                                })
                                logger.info(f"Relação por valor detectada: {source_name}.{source_col} -> {target_name}.{target_pk} [overlap: {overlap:.2f}]")
    
    def _add_relationships_to_metadata(self) -> None:
        """
        Adiciona as relações detectadas aos metadados dos datasets.
        """
        # Cria um mapeamento reverso para relações
        relations_by_dataset = {}
        
        for rel in self.relationships:
            source = rel["source_dataset"]
            target = rel["target_dataset"]
            
            # Inicializa se necessário
            if source not in relations_by_dataset:
                relations_by_dataset[source] = {"outgoing": [], "incoming": []}
            if target not in relations_by_dataset:
                relations_by_dataset[target] = {"outgoing": [], "incoming": []}
            
            # Adiciona a relação nos dois datasets
            relations_by_dataset[source]["outgoing"].append({
                "name": f"{source}.{rel['source_column']} -> {target}.{rel['target_column']}",
                "target_dataset": target,
                "source_column": rel["source_column"],
                "target_column": rel["target_column"],
                "confidence": rel["confidence"],
                "type": rel["relationship_type"]
            })
            
            relations_by_dataset[target]["incoming"].append({
                "name": f"{source}.{rel['source_column']} -> {target}.{rel['target_column']}",
                "source_dataset": source,
                "source_column": rel["source_column"],
                "target_column": rel["target_column"],
                "confidence": rel["confidence"],
                "type": rel["relationship_type"]
            })
        
        # Adiciona relações aos metadados
        for ds_name, relations in relations_by_dataset.items():
            if ds_name in self.metadata:
                self.metadata[ds_name]["relationships"] = relations
    
    def _names_match(self, name1: str, name2: str) -> bool:
        """
        Verifica se dois nomes têm uma correspondência significativa.
        
        Args:
            name1: Primeiro nome
            name2: Segundo nome
            
        Returns:
            True se os nomes correspondem, False caso contrário
        """
        # Normaliza os nomes
        name1 = name1.lower().strip()
        name2 = name2.lower().strip()
        
        # Verifica plurais e singulares
        if name1 == name2 or name1 + "s" == name2 or name1 == name2 + "s":
            return True
        
        # Compara versões sem underscores e sem espaços
        name1_clean = name1.replace("_", "").replace(" ", "")
        name2_clean = name2.replace("_", "").replace(" ", "")
        
        return name1_clean == name2_clean
    
    def save_metadata(self, output_path: str) -> None:
        """
        Salva todos os metadados em um arquivo JSON.
        
        Args:
            output_path: Caminho para o arquivo de saída
        """
        # Analisa se ainda não fez
        if not self.metadata:
            self.analyze_all()
        
        # Prepara o resultado completo
        result = {
            "metadata": self.metadata,
            "relationships": self.relationships,
            "analysis_info": {
                "timestamp": datetime.now().isoformat(),
                "total_datasets": len(self.datasets),
                "total_relationships": len(self.relationships)
            }
        }
        
        # Salva em formato JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Metadados salvos em {output_path}")
    
    def generate_schema_dict(self) -> Dict[str, Any]:
        """
        Gera um dicionário de esquema para uso em sistemas de consulta.
        
        Returns:
            Dict com esquema estruturado para consultas
        """
        # Analisa se ainda não fez
        if not self.metadata:
            self.analyze_all()
        
        schema = {
            "datasets": {},
            "relationships": []
        }
        
        # Processa cada dataset
        for ds_name, ds_meta in self.metadata.items():
            dataset_info = {
                "name": ds_name,
                "primary_key": ds_meta.get("primary_key"),
                "description": f"Dataset {ds_name} com {ds_meta.get('row_count')} registros",
                "columns": []
            }
            
            # Adiciona informações sobre cada coluna
            for col_name, col_meta in ds_meta.get("columns", {}).items():
                column_info = {
                    "name": col_name,
                    "type": col_meta.get("suggested_type"),
                    "description": self._generate_column_description(col_name, col_meta),
                    "nullable": col_meta.get("nullable", True)
                }
                
                # Adiciona informações específicas do tipo
                if col_meta.get("suggested_type") == "date" and "date_format" in col_meta:
                    column_info["format"] = col_meta["date_format"]
                
                dataset_info["columns"].append(column_info)
            
            schema["datasets"][ds_name] = dataset_info
        
        # Adiciona relacionamentos
        for rel in self.relationships:
            relationship_info = {
                "source_dataset": rel["source_dataset"],
                "source_column": rel["source_column"],
                "target_dataset": rel["target_dataset"],
                "target_column": rel["target_column"],
                "type": rel["relationship_type"],
                "description": f"Relacionamento de {rel['source_dataset']} para {rel['target_dataset']}"
            }
            
            schema["relationships"].append(relationship_info)
        
        return schema
    
    def _generate_column_description(self, col_name: str, col_meta: Dict[str, Any]) -> str:
        """
        Gera uma descrição legível para uma coluna.
        
        Args:
            col_name: Nome da coluna
            col_meta: Metadados da coluna
            
        Returns:
            String com descrição da coluna
        """
        descriptions = []
        
        # Tipo de dados
        type_desc = f"Coluna do tipo {col_meta.get('suggested_type', 'desconhecido')}"
        descriptions.append(type_desc)
        
        # Informação de chave primária
        if col_meta.get("potential_primary_key"):
            descriptions.append("Provável chave primária")
        
        # Informação de chave estrangeira
        if col_meta.get("potential_foreign_key"):
            descriptions.append("Possível referência para outra tabela")
        
        # Para tipos numéricos, adiciona faixa de valores
        if col_meta.get("suggested_type") == "numeric" and "numeric_stats" in col_meta:
            stats = col_meta["numeric_stats"]
            if "min" in stats and "max" in stats and stats["min"] is not None and stats["max"] is not None:
                descriptions.append(f"Valores entre {stats['min']} e {stats['max']}")
        
        # Para datas, adiciona range temporal
        if col_meta.get("suggested_type") == "date" and "temporal_stats" in col_meta:
            stats = col_meta["temporal_stats"]
            if "min_date" in stats and "max_date" in stats and stats["min_date"] and stats["max_date"]:
                descriptions.append(f"Datas entre {stats['min_date']} e {stats['max_date']}")
        
        # Para categóricos, menciona os valores mais comuns
        if col_meta.get("suggested_type") == "categorical" and "top_values" in col_meta:
            top_values = list(col_meta["top_values"].keys())[:3]
            if top_values:
                descriptions.append(f"Valores mais comuns: {', '.join(top_values)}")
        
        return ". ".join(descriptions)


def analyze_datasets_from_dict(datasets: Dict[str, pd.DataFrame], output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Função auxiliar para analisar múltiplos datasets e opcionalmente salvar o resultado.
    
    Args:
        datasets: Dicionário com nomes de datasets e DataFrames
        output_path: Caminho opcional para salvar os metadados
        
    Returns:
        Metadados gerados
    """
    analyzer = DatasetAnalyzer()
    
    # Adiciona os datasets
    for name, df in datasets.items():
        analyzer.add_dataset(name, df)
    
    # Faz a análise
    result = analyzer.analyze_all()
    
    # Salva se for especificado um caminho
    if output_path:
        analyzer.save_metadata(output_path)
    
    return result


def analyze_datasets_from_files(file_paths: Dict[str, str], output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Função auxiliar para analisar datasets a partir de arquivos CSV.
    
    Args:
        file_paths: Dicionário com nomes de datasets e caminhos para arquivos CSV
        output_path: Caminho opcional para salvar os metadados
        
    Returns:
        Metadados gerados
    """
    # Carrega os datasets de arquivos
    datasets = {}
    for name, file_path in file_paths.items():
        try:
            df = pd.read_csv(file_path)
            datasets[name] = df
            logger.info(f"Carregado dataset '{name}' de {file_path}")
        except Exception as e:
            logger.error(f"Erro ao carregar {file_path}: {str(e)}")
    
    # Analisa os datasets carregados
    return analyze_datasets_from_dict(datasets, output_path)


if __name__ == "__main__":
    # Configuração de logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Exemplo para teste com datasets na pasta dados/
    import os
    
    # Diretório de dados
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dados")
    
    # Verifica se existe
    if os.path.exists(data_dir):
        # Lista arquivos CSV
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        
        if csv_files:
            # Cria dicionário com caminhos para análise
            file_paths = {
                os.path.splitext(f)[0]: os.path.join(data_dir, f) 
                for f in csv_files
            }
            
            # Define caminho de saída
            output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema_output.json")
            
            # Faz a análise
            metadata = analyze_datasets_from_files(file_paths, output_path)
            print(f"Análise concluída. Metadados salvos em {output_path}")
        else:
            print(f"Nenhum arquivo CSV encontrado em {data_dir}")
    else:
        print(f"Diretório de dados {data_dir} não encontrado")