"""
Exemplo de uso robusto do módulo de conectores de dados
=======================================================

Este script demonstra como usar o módulo de conectores de forma robusta,
com tratamento adequado de erros e uso das versões corrigidas dos conectores.
"""

import os
import json
import pandas as pd
from typing import Dict, List, Any, Optional
import logging
import matplotlib.pyplot as plt

# Importa o módulo de conectores padrão
from connectors import (
    DataConnector, 
    DataSourceConfig,
    DataConnectionException, 
    DataReadException,
    DuckDBCsvConnector
)

# Importa o módulo de metadados e conectores com suporte a metadados
from metadata_connectors import (
    MetadataEnabledDataConnectorFactory,
    MetadataEnabledDataSourceConfig,
    MetadataRegistry
)

# Importa o módulo de metadados
from column_metadata import (
    DatasetMetadata,
    ColumnMetadata
)

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_connector.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("data_analyzer")

class DataAnalyzer:
    """
    Classe para análise de dados usando o módulo de conectores.
    Implementa tratamento robusto de erros.
    """
    
    def __init__(self, config_file: str, metadata_file: str = None):
        """
        Inicializa o analisador de dados.
        
        Args:
            config_file: Caminho para o arquivo de configuração JSON.
            metadata_file: Caminho para o arquivo de metadados JSON.
        """
        self.config_file = config_file
        self.metadata_file = metadata_file
        self.connectors: Dict[str, DataConnector] = {}
        self.metadata_registry = MetadataRegistry()
        
        # Carrega metadados se disponíveis
        if metadata_file and os.path.exists(metadata_file):
            try:
                self.metadata_registry.register_from_file(metadata_file)
                logger.info(f"Metadados carregados do arquivo: {metadata_file}")
            except Exception as e:
                logger.error(f"Erro ao carregar metadados: {str(e)}")
        
        self.load_connectors()
    
    def load_connectors(self) -> None:
        """
        Carrega os conectores a partir do arquivo de configuração.
        """
        if not os.path.exists(self.config_file):
            logger.error(f"Arquivo de configuração não encontrado: {self.config_file}")
            # Cria uma configuração padrão
            default_config = {
                "data_sources": [
                    {
                        "id": "default_csv",
                        "type": "csv",
                        "path": "dados.csv"
                    }
                ]
            }
            with open(self.config_file, "w") as f:
                json.dump(default_config, f, indent=2)
            logger.info(f"Criado arquivo de configuração padrão: {self.config_file}")
        
        try:
            with open(self.config_file, "r") as f:
                config_json = f.read()
            
            # Modifica o JSON para incluir referências aos metadados
            if self.metadata_file and os.path.exists(self.metadata_file):
                try:
                    config_data = json.loads(config_json)
                    
                    # Adiciona o caminho do arquivo de metadados à configuração
                    if "metadata" not in config_data:
                        config_data["metadata"] = {"files": [self.metadata_file]}
                    elif "files" not in config_data["metadata"]:
                        config_data["metadata"]["files"] = [self.metadata_file]
                    elif self.metadata_file not in config_data["metadata"]["files"]:
                        config_data["metadata"]["files"].append(self.metadata_file)
                    
                    # Atualiza o JSON de configuração
                    config_json = json.dumps(config_data)
                    logger.info(f"Configuração atualizada com referência a metadados: {self.metadata_file}")
                except Exception as e:
                    logger.error(f"Erro ao modificar configuração com metadados: {str(e)}")
            
            # Utiliza a factory com suporte a metadados
            self.connectors = MetadataEnabledDataConnectorFactory.create_from_json(config_json)
            logger.info(f"Carregados {len(self.connectors)} conectores com suporte a metadados")
            
        except Exception as e:
            logger.error(f"Erro ao carregar conectores: {str(e)}")
            # Cria um conector padrão para não quebrar o fluxo
            config = MetadataEnabledDataSourceConfig("default", "csv", path="dados.csv")
            self.connectors = {"default": MetadataEnabledDataConnectorFactory.create_connector(config)}
    
    def get_available_sources(self) -> List[str]:
        """
        Retorna as fontes de dados disponíveis.
        
        Returns:
            List[str]: Lista de IDs das fontes disponíveis.
        """
        return list(self.connectors.keys())
    
    def get_connector(self, source_id: str) -> Optional[DataConnector]:
        """
        Obtém um conector específico, com tratamento de erro.
        
        Args:
            source_id: ID da fonte de dados.
            
        Returns:
            Optional[DataConnector]: O conector solicitado ou None se não encontrado.
        """
        if source_id not in self.connectors:
            logger.warning(f"Fonte não encontrada: {source_id}")
            return None
        return self.connectors[source_id]
    
    def detect_dataset_structure(self, source_id: str) -> Dict[str, Any]:
        """
        Detecta a estrutura do dataset.
        
        Args:
            source_id: ID da fonte de dados.
            
        Returns:
            Dict: Informações sobre a estrutura do dataset.
        """
        connector = self.get_connector(source_id)
        if not connector:
            return {"error": f"Fonte não encontrada: {source_id}"}
        
        try:
            connector.connect()
            
            # Para conectores DuckDB, use o método get_schema
            if hasattr(connector, 'get_schema') and callable(getattr(connector, 'get_schema')):
                schema = connector.get_schema()
                logger.info(f"Esquema obtido para {source_id}: {schema}")
                
                # Se o método sample_data estiver disponível, use-o
                if hasattr(connector, 'sample_data') and callable(getattr(connector, 'sample_data')):
                    sample = connector.sample_data(5)
                else:
                    sample = connector.read_data("SELECT * FROM csv LIMIT 5")
                
                result = {
                    "columns": list(sample.columns),
                    "dtypes": {col: str(sample[col].dtype) for col in sample.columns},
                    "sample": sample.to_dict(orient='records'),
                    "schema": schema.to_dict(orient='records') if not schema.empty else []
                }
            else:
                # Para outros conectores, extraia informações do DataFrame
                sample = connector.read_data()
                sample = sample.head(5) if not sample.empty else sample
                
                result = {
                    "columns": list(sample.columns),
                    "dtypes": {col: str(sample[col].dtype) for col in sample.columns},
                    "sample": sample.to_dict(orient='records')
                }
            
            # Tenta identificar colunas especiais
            numeric_cols = sample.select_dtypes(include=['number']).columns.tolist()
            date_cols = [col for col in sample.columns if 'date' in col.lower() or 'data' in col.lower()]
            id_cols = [col for col in sample.columns if 'id' in col.lower() or 'codigo' in col.lower()]
            
            result.update({
                "numeric_columns": numeric_cols,
                "date_columns": date_cols,
                "id_columns": id_cols
            })
            
            # Adiciona informações de metadados, se disponíveis
            if hasattr(connector, 'config') and hasattr(connector.config, 'metadata') and connector.config.metadata:
                metadata = connector.config.metadata
                meta_info = {
                    "dataset_name": metadata.name,
                    "dataset_description": metadata.description,
                    "columns_metadata": {}
                }
                
                for col_name, col_meta in metadata.columns.items():
                    meta_info["columns_metadata"][col_name] = {
                        "description": col_meta.description,
                        "data_type": col_meta.data_type,
                        "format": col_meta.format,
                        "alias": col_meta.alias,
                        "aggregations": col_meta.aggregations,
                        "tags": col_meta.tags
                    }
                
                result["metadata"] = meta_info
            
            return result
            
        except Exception as e:
            error_msg = f"Erro ao detectar estrutura: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}
        finally:
            try:
                connector.close()
            except Exception:
                pass
    
    def generate_summary(self, source_id: str) -> Dict[str, Any]:
        """
        Gera um resumo estatístico dos dados.
        
        Args:
            source_id: ID da fonte de dados.
            
        Returns:
            Dict: Resumo estatístico.
        """
        connector = self.get_connector(source_id)
        if not connector:
            return {"error": f"Fonte não encontrada: {source_id}"}
        
        try:
            connector.connect()
            
            # Obtém uma amostra dos dados para análise inicial
            if hasattr(connector, 'sample_data'):
                df = connector.sample_data(1000)  # Limite maior para estatísticas mais precisas
            else:
                df = connector.read_data("SELECT * FROM csv LIMIT 1000")
            
            if df.empty:
                return {"warning": "Dataset vazio"}
            
            # Detecta colunas numéricas
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            
            # Estatísticas básicas
            stats = {}
            
            # Para colunas numéricas
            if numeric_cols:
                desc_stats = df[numeric_cols].describe().to_dict()
                stats["numeric"] = desc_stats
            
            # Cardinalidade das colunas (contagem de valores únicos)
            cardinality = {col: df[col].nunique() for col in df.columns}
            stats["cardinality"] = cardinality
            
            # Contagem de nulos
            null_counts = {col: int(df[col].isna().sum()) for col in df.columns}
            stats["null_counts"] = null_counts
            
            # Maiores valores para colunas numéricas
            top_values = {}
            for col in numeric_cols:
                top_values[col] = df[col].nlargest(5).tolist()
            stats["top_values"] = top_values
            
            # Contagem total de registros
            if hasattr(connector, 'count_rows'):
                stats["total_rows"] = connector.count_rows()
            else:
                # Estima o total
                stats["total_rows_estimate"] = "1000+ (amostra limitada)"
            
            # Adiciona informações específicas baseadas em metadados
            if hasattr(connector, 'config') and hasattr(connector.config, 'metadata') and connector.config.metadata:
                metadata = connector.config.metadata
                meta_stats = {"column_recommended_aggregations": {}}
                
                # Para cada coluna numérica, verifica agregações recomendadas
                for col in numeric_cols:
                    col_meta = metadata.get_column_metadata(col)
                    if col_meta and col_meta.aggregations:
                        # Calcula agregações recomendadas
                        agg_results = {}
                        for agg in col_meta.aggregations:
                            if agg == 'sum':
                                agg_results['sum'] = float(df[col].sum())
                            elif agg == 'avg' or agg == 'mean':
                                agg_results['avg'] = float(df[col].mean())
                            elif agg == 'min':
                                agg_results['min'] = float(df[col].min())
                            elif agg == 'max':
                                agg_results['max'] = float(df[col].max())
                        
                        meta_stats["column_recommended_aggregations"][col] = agg_results
                
                stats["metadata_based"] = meta_stats
            
            return {
                "summary": stats,
                "columns": list(df.columns),
                "row_sample": df.head(5).to_dict(orient='records')
            }
            
        except Exception as e:
            error_msg = f"Erro ao gerar resumo: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}
        finally:
            try:
                connector.close()
            except Exception:
                pass
    
    def create_visualization(self, source_id: str, output_path: str) -> Dict[str, Any]:
        """
        Cria visualizações dos dados.
        
        Args:
            source_id: ID da fonte de dados.
            output_path: Diretório para salvar as visualizações.
            
        Returns:
            Dict: Informações sobre as visualizações criadas.
        """
        connector = self.get_connector(source_id)
        if not connector:
            return {"error": f"Fonte não encontrada: {source_id}"}
        
        # Cria o diretório se não existir
        os.makedirs(output_path, exist_ok=True)
        
        try:
            connector.connect()
            
            # Obtém uma amostra dos dados
            if hasattr(connector, 'sample_data'):
                df = connector.sample_data(1000)
            else:
                df = connector.read_data("SELECT * FROM csv LIMIT 1000")
            
            if df.empty:
                return {"warning": "Dataset vazio, não é possível criar visualizações"}
            
            visualizations = []
            
            # Detecta colunas numéricas e categóricas
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Se temos metadados, podemos melhorar a detecção de colunas interessantes
            if hasattr(connector, 'config') and hasattr(connector.config, 'metadata') and connector.config.metadata:
                metadata = connector.config.metadata
                
                # Prioriza colunas com metadados financeiros ou KPIs para visualização
                priority_cols = []
                for col_name in numeric_cols:
                    col_meta = metadata.get_column_metadata(col_name)
                    if col_meta and col_meta.tags:
                        if any(tag in ['financial', 'kpi', 'monetary', 'performance'] for tag in col_meta.tags):
                            priority_cols.append(col_name)
                
                # Se encontrou colunas prioritárias, ajusta a lista de colunas numéricas
                if priority_cols:
                    # Coloca as colunas prioritárias no início da lista
                    for col in priority_cols:
                        if col in numeric_cols:
                            numeric_cols.remove(col)
                    numeric_cols = priority_cols + numeric_cols
            
            # 1. Histograma para cada coluna numérica
            for i, col in enumerate(numeric_cols[:5]):  # Limita a 5 colunas
                plt.figure(figsize=(8, 6))
                plt.hist(df[col].dropna(), bins=20, alpha=0.7)
                plt.title(f'Histograma de {col}')
                plt.xlabel(col)
                plt.ylabel('Frequência')
                plt.grid(True, alpha=0.3)
                
                file_path = os.path.join(output_path, f'histogram_{col}.png')
                plt.savefig(file_path)
                plt.close()
                
                visualizations.append({
                    "type": "histogram",
                    "column": col,
                    "file": file_path
                })
            
            # 2. Gráfico de barras para colunas categóricas (se houver)
            for i, col in enumerate(cat_cols[:3]):  # Limita a 3 colunas
                if df[col].nunique() <= 15:  # Apenas se tiver poucas categorias
                    plt.figure(figsize=(10, 6))
                    df[col].value_counts().plot(kind='bar')
                    plt.title(f'Contagem de {col}')
                    plt.xlabel(col)
                    plt.ylabel('Contagem')
                    plt.xticks(rotation=45)
                    plt.grid(True, alpha=0.3)
                    
                    file_path = os.path.join(output_path, f'barplot_{col}.png')
                    plt.savefig(file_path)
                    plt.close()
                    
                    visualizations.append({
                        "type": "barplot",
                        "column": col,
                        "file": file_path
                    })
            
            # 3. Matriz de correlação (se houver múltiplas colunas numéricas)
            if len(numeric_cols) >= 2:
                plt.figure(figsize=(10, 8))
                corr = df[numeric_cols].corr()
                plt.imshow(corr, cmap='coolwarm', interpolation='none', aspect='auto')
                plt.colorbar()
                plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=90)
                plt.yticks(range(len(numeric_cols)), numeric_cols)
                plt.title('Matriz de Correlação')
                
                # Adiciona os valores no heatmap
                for i in range(len(numeric_cols)):
                    for j in range(len(numeric_cols)):
                        plt.text(j, i, f'{corr.iloc[i, j]:.2f}', 
                                ha='center', va='center', color='white')
                
                file_path = os.path.join(output_path, 'correlation_matrix.png')
                plt.savefig(file_path)
                plt.close()
                
                visualizations.append({
                    "type": "correlation",
                    "columns": numeric_cols,
                    "file": file_path
                })
            
            # 4. Gráfico de barras para Motivo vs ImpactoFinanceiro se temos esses campos
            if 'Motivo' in df.columns and 'ImpactoFinanceiro' in df.columns:
                plt.figure(figsize=(12, 6))
                df.groupby('Motivo')['ImpactoFinanceiro'].sum().sort_values(ascending=False).plot(kind='bar')
                plt.title('Impacto Financeiro por Motivo de Perda')
                plt.xlabel('Motivo')
                plt.ylabel('Impacto Financeiro Total (R$)')
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
                
                file_path = os.path.join(output_path, 'impacto_por_motivo.png')
                plt.savefig(file_path)
                plt.close()
                
                visualizations.append({
                    "type": "barplot",
                    "analysis": "Impacto Financeiro por Motivo",
                    "file": file_path
                })
            
            return {
                "visualizations": visualizations,
                "output_directory": output_path
            }
            
        except Exception as e:
            error_msg = f"Erro ao criar visualizações: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}
        finally:
            try:
                connector.close()
            except Exception:
                pass

# Exemplo de uso
if __name__ == "__main__":
    # Configura o analisador com suporte a metadados
    analyzer = DataAnalyzer("datasources.json", "metadata.json")
    
    # Lista as fontes disponíveis
    sources = analyzer.get_available_sources()
    print(f"Fontes disponíveis: {sources}")
    
    # Para cada fonte, detecta a estrutura e cria visualizações
    for source_id in sources:
        print(f"\n--- Análise da fonte: {source_id} ---")
        
        # Detecta estrutura
        structure = analyzer.detect_dataset_structure(source_id)
        if "error" in structure:
            print(f"Erro ao detectar estrutura: {structure['error']}")
            continue
            
        print(f"Colunas detectadas: {structure['columns']}")
        print(f"Colunas numéricas: {structure.get('numeric_columns', [])}")
        
        # Se temos metadados, exibe informações adicionais
        if "metadata" in structure:
            print("\nInformações de metadados:")
            print(f"  Nome do dataset: {structure['metadata']['dataset_name']}")
            print(f"  Descrição: {structure['metadata']['dataset_description']}")
            print("  Metadados de colunas:")
            for col, meta in structure['metadata']['columns_metadata'].items():
                print(f"    {col}: {meta['description']} ({meta['data_type']})")
        
        # Gera resumo estatístico
        summary = analyzer.generate_summary(source_id)
        if "error" in summary:
            print(f"Erro ao gerar resumo: {summary['error']}")
        else:
            print("\nResumo estatístico:")
            for key, value in summary.get('summary', {}).items():
                if key == 'numeric' and isinstance(value, dict):
                    print(f"  Estatísticas numéricas:")
                    for col, stats in value.items():
                        print(f"    {col}: {stats}")
                elif key == 'metadata_based' and isinstance(value, dict):
                    print(f"  Estatísticas baseadas em metadados:")
                    for agg_type, aggs in value.items():
                        print(f"    {agg_type}:")
                        for col, results in aggs.items():
                            print(f"      {col}: {results}")
                else:
                    print(f"  {key}: {value}")
        
        # Cria visualizações
        viz_result = analyzer.create_visualization(source_id, f"output/{source_id}")
        if "error" in viz_result:
            print(f"Erro ao criar visualizações: {viz_result['error']}")
        else:
            print("\nVisualizações criadas:")
            for viz in viz_result.get('visualizations', []):
                if 'column' in viz:
                    print(f"  {viz['type']} para {viz['column']}")
                elif 'analysis' in viz:
                    print(f"  {viz['type']} para {viz['analysis']}")
                else:
                    print(f"  {viz['type']} para {viz.get('columns', 'múltiplas colunas')}")
            
        print("\n" + "="*50)