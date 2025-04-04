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

# Importa o módulo de conectores
from connectors import (
    DataConnector, 
    DataConnectorFactory, 
    DataSourceConfig,
    DataConnectionException, 
    DataReadException,
    DuckDBCsvConnector
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
    
    def __init__(self, config_file: str):
        """
        Inicializa o analisador de dados.
        
        Args:
            config_file: Caminho para o arquivo de configuração JSON.
        """
        self.config_file = config_file
        self.connectors: Dict[str, DataConnector] = {}
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
            
            self.connectors = DataConnectorFactory.create_from_json(config_json)
            logger.info(f"Carregados {len(self.connectors)} conectores")
            
        except Exception as e:
            logger.error(f"Erro ao carregar conectores: {str(e)}")
            # Cria um conector padrão para não quebrar o fluxo
            config = DataSourceConfig("default", "csv", path="dados.csv")
            self.connectors = {"default": DataConnectorFactory.create_connector(config)}
    
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
            if hasattr(connector, 'get_row_count'):
                stats["total_rows"] = connector.get_row_count()
            else:
                # Estima o total
                stats["total_rows_estimate"] = "1000+ (amostra limitada)"
            
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
    # Configura o analisador
    analyzer = DataAnalyzer("datasources.json")
    
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
                else:
                    print(f"  {key}: {value}")
        
        # Cria visualizações
        viz_result = analyzer.create_visualization(source_id, f"output/{source_id}")
        if "error" in viz_result:
            print(f"Erro ao criar visualizações: {viz_result['error']}")
        else:
            print("\nVisualizações criadas:")
            for viz in viz_result.get('visualizations', []):
                print(f"  {viz['type']} para {viz.get('column', viz.get('columns', 'múltiplas colunas'))}")
            
        print("\n" + "="*50)