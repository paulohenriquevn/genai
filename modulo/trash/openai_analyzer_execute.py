import os
import sys
import json
import time
import logging
import argparse
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Type
from datetime import datetime
import base64
from pathlib import Path
import webbrowser
import matplotlib.pyplot as plt
import io
import traceback

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("openai_analyzer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("openai_analyzer")

# Importação da biblioteca OpenAI
try:
    import openai
except ImportError:
    logger.error("Biblioteca OpenAI não encontrada. Instale com: pip install openai>=1.3.0")
    sys.exit(1)

# Importação dos módulos do sistema
from openai_analyzer import OpenAIAnalyzer
from core.response.dataframe import DataFrameResponse
from core.response.number import NumberResponse
from core.response.string import StringResponse
from core.response.chart import ChartResponse


def main():
    """
    Função principal para execução como script.
    """
    # Parse argumentos da linha de comando
    parser = argparse.ArgumentParser(description="Analisador de dados com OpenAI")
    
    parser.add_argument("--api-key", help="Chave de API do OpenAI")
    parser.add_argument("--model", default="gpt-4", help="Modelo OpenAI a ser utilizado")
    parser.add_argument("--data-dir", help="Diretório onde os dados estão armazenados")
    parser.add_argument("--output-dir", help="Diretório para salvar relatórios e visualizações")
    parser.add_argument("--dataset", help="Caminho para um arquivo de dataset específico")
    parser.add_argument("--dataset-name", help="Nome personalizado para o dataset")
    
    # Opções de análise
    parser.add_argument("--topic", default="vendas perdidas", 
                        help="Tópico para análise automática (ex: 'vendas', 'clientes')")
    parser.add_argument("--questions", type=int, default=5,
                        help="Número de perguntas a serem geradas")
    parser.add_argument("--query", help="Executar uma consulta específica")
    parser.add_argument("--open-report", action="store_true", 
                        help="Abrir relatório no navegador após a análise")
    parser.add_argument("--log-level", default="DEBUG", 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Nível de logging")
    
    # Opções de processamento de dataset
    parser.add_argument("--process-dataset-only", action="store_true",
                        help="Apenas processar o dataset e gerar metadados/esquema")
    
    args = parser.parse_args()
    
    try:
        # Inicializa o analisador
        analyzer = OpenAIAnalyzer(
            api_key=args.api_key,
            model=args.model,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            dataset_path=args.dataset,
            dataset_name=args.dataset_name,
            log_level=args.log_level
        )
        
        # Se foi solicitado apenas o processamento do dataset
        if args.process_dataset_only:
            if not args.dataset:
                raise ValueError("--dataset é obrigatório com --process-dataset-only")
            
            print(f"\nProcessando dataset: {args.dataset}")
            schema = analyzer.process_dataset(args.dataset)
            
            print(f"\nEsquema semântico gerado para '{schema.name}':")
            print(f"- {len(schema.columns)} colunas analisadas")
            print(f"- {len(schema.relations)} relações detectadas")
            print(f"- Arquivos de saída salvos em: {args.output_dir or os.path.join(os.getcwd(), 'output')}")
            
            print("\nProcessamento concluído.")
            sys.exit(0)
        
        # Se um dataset específico foi fornecido, processa-o
        if args.dataset:
            print(f"\nProcessando dataset: {args.dataset}")
            analyzer.process_dataset(args.dataset)
        
        if args.query:
            # Executa uma consulta específica
            print(f"\nExecutando consulta: '{args.query}'")
            result, analysis = analyzer.run_query(args.query)
            
            print("\n--- Resultado ---")
            if isinstance(result, DataFrameResponse):
                print("\nDataFrame resultante:")
                print(result.value.head(10))
                if len(result.value) > 10:
                    print(f"...e mais {len(result.value) - 10} linhas.")
            elif isinstance(result, ChartResponse):
                viz_path = analyzer.save_visualization(result, "consulta_especifica")
                print(f"\nVisualização gerada e salva em: {viz_path}")
            elif isinstance(result, NumberResponse):
                print(f"\nResultado numérico: {result.value}")
            elif isinstance(result, StringResponse):
                print(f"\nResultado textual: {result.value}")
            else:
                print(f"\nResultado de tipo {type(result).__name__}: {result.value}")
            
            print("\n--- Análise ---")
            print(analysis)
            
        else:
            # Executa análise completa
            print(f"\nIniciando análise sobre '{args.topic}' com {args.questions} perguntas...")
            
            analysis_results = analyzer.run_analysis(args.topic, args.questions)
            
            print(f"\nAnálise concluída. Resultados salvos em: {analysis_results['report_path']}")
            
            if args.open_report:
                print("Abrindo relatório no navegador...")
                analyzer.open_report(analysis_results['report_path'])
            
        print("\nProcessamento concluído.")
        
    except Exception as e:
        logger.error(f"Erro na execução do analisador: {str(e)}")
        logger.debug(traceback.format_exc())
        print(f"\nErro: {str(e)}")
        print("\nVerifique o arquivo de log para mais detalhes.")
        sys.exit(1)


if __name__ == "__main__":
    main()