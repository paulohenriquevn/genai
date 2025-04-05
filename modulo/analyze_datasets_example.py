#!/usr/bin/env python3
"""
Script de exemplo para demonstrar o uso do analisador de datasets.

Este script carrega os datasets disponíveis na pasta 'dados/',
realiza uma análise completa e gera um arquivo de metadados.
"""

import os
import pandas as pd
import logging
import json
from utils.dataset_analyzer import DatasetAnalyzer, analyze_datasets_from_files

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("analyzer_example")

def main():
    print("====================================================")
    print("🔍 Análise Dinâmica de Datasets")
    print("====================================================")
    
    # Diretório de dados e saída
    dados_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dados")
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils")
    output_file = os.path.join(output_dir, "schema_output.json")
    
    print(f"\n📂 Procurando arquivos CSV em: {dados_dir}")
    
    # Verifica se o diretório existe
    if not os.path.exists(dados_dir):
        print(f"❌ Diretório não encontrado: {dados_dir}")
        return
    
    # Lista arquivos CSV
    csv_files = [f for f in os.listdir(dados_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print("❌ Nenhum arquivo CSV encontrado!")
        return
    
    print(f"✅ Encontrados {len(csv_files)} arquivos CSV:")
    for file in csv_files:
        print(f"  - {file}")
    
    # Cria dicionário com caminhos para análise
    file_paths = {
        os.path.splitext(f)[0]: os.path.join(dados_dir, f) 
        for f in csv_files
    }
    
    print("\n🔄 Iniciando análise dos datasets...")
    
    # Método 1: Usando função auxiliar
    metadata = analyze_datasets_from_files(file_paths, output_file)
    
    # Alternativa: Uso detalhado do DatasetAnalyzer
    """
    # Inicializa o analisador
    analyzer = DatasetAnalyzer()
    
    # Carrega todos os datasets
    for name, path in file_paths.items():
        df = pd.read_csv(path)
        print(f"  📊 Dataset '{name}' carregado: {len(df)} linhas, {len(df.columns)} colunas")
        analyzer.add_dataset(name, df)
    
    # Executa a análise completa
    metadata = analyzer.analyze_all()
    
    # Salva os metadados em arquivo JSON
    analyzer.save_metadata(output_file)
    
    # Gera esquema para consultas
    schema = analyzer.generate_schema_dict()
    """
    
    print(f"\n✅ Análise concluída! Metadados salvos em: {output_file}")
    
    # Exibe resumo da análise
    print("\n📋 Resumo da análise:")
    
    # Total de datasets e colunas
    total_datasets = len(metadata["metadata"])
    total_columns = sum(
        len(ds_meta.get("columns", {})) 
        for ds_meta in metadata["metadata"].values()
    )
    
    print(f"  • Datasets analisados: {total_datasets}")
    print(f"  • Total de colunas: {total_columns}")
    print(f"  • Relacionamentos detectados: {len(metadata['relationships'])}")
    
    # Exibe detalhes de cada dataset
    print("\n📊 Detalhes dos datasets:")
    for ds_name, ds_meta in metadata["metadata"].items():
        pk = ds_meta.get("primary_key", "Não identificada")
        print(f"  • {ds_name}")
        print(f"    - Linhas: {ds_meta.get('row_count', 0)}")
        print(f"    - Colunas: {ds_meta.get('column_count', 0)}")
        print(f"    - Chave primária: {pk}")
        
        # Lista tipos de dados identificados
        col_types = {}
        for col, meta in ds_meta.get("columns", {}).items():
            col_type = meta.get("suggested_type", "desconhecido")
            col_types[col_type] = col_types.get(col_type, 0) + 1
        
        print(f"    - Tipos de dados: {dict(col_types)}")
    
    # Exibe relacionamentos detectados
    if metadata['relationships']:
        print("\n🔗 Relacionamentos detectados:")
        for rel in metadata['relationships']:
            src = f"{rel['source_dataset']}.{rel['source_column']}"
            tgt = f"{rel['target_dataset']}.{rel['target_column']}"
            rel_type = rel['relationship_type']
            confidence = rel.get('confidence', 0) * 100
            
            print(f"  • {src} → {tgt} ({rel_type}, confiança: {confidence:.1f}%)")

    print("\n✅ Exemplo concluído.")
    print("====================================================")

if __name__ == "__main__":
    main()