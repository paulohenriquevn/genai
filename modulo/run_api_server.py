#!/usr/bin/env python3
"""
Servidor API para o Sistema de Consulta em Linguagem Natural
"""

import os
import uvicorn
import argparse
from dotenv import load_dotenv

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Servidor API para Sistema de Consulta em Linguagem Natural")
    parser.add_argument("--host", default="0.0.0.0", help="Endereço de host (padrão: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Porta (padrão: 8000)")
    parser.add_argument("--reload", action="store_true", help="Ativar recarga automática do código")
    
    args = parser.parse_args()
    
    print(f"Iniciando servidor API em {args.host}:{args.port}")
    print(f"Ambiente: {'Desenvolvimento' if args.reload else 'Produção'}")
    
    # Verifica se as chaves API necessárias estão configuradas
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("AVISO: Nenhuma chave API (OPENAI_API_KEY ou ANTHROPIC_API_KEY) configurada.")
        print("O sistema usará o modo simulado (mock) para LLM, que não produzirá resultados reais.")
    
    # Inicia o servidor
    uvicorn.run(
        "api:app", 
        host=args.host, 
        port=args.port, 
        reload=args.reload
    )

if __name__ == "__main__":
    main()