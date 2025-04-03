import requests
import json
import os
import sys
import logging
import traceback
import argparse
import webbrowser
from typing import Optional, Dict, Any, List
from datetime import datetime

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("genbi.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("GenBI.Client")

# URL base da API GenBI (pode ser sobrescrito com variável de ambiente)
BASE_URL = os.environ.get("GENBI_BASE_URL", "http://localhost:8000")

class GenBIClient:
    """Cliente para interação com a API GenBI"""
    
    def __init__(self, base_url: str = BASE_URL):
        """Inicializa o cliente GenBI com a URL base da API"""
        self.base_url = base_url
        self.session = requests.Session()
        
        # Validar conexão com o servidor
        try:
            self.check_health()
            logger.info(f"Conectado com sucesso ao servidor GenBI em {base_url}")
        except Exception as e:
            logger.error(f"Erro ao conectar ao servidor GenBI: {str(e)}")
            raise
    
    def check_health(self) -> Dict[str, Any]:
        """Verifica a saúde do sistema"""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def natural_language_query(self, question: str, use_cache: bool = True, explain_sql: bool = False) -> Dict[str, Any]:
        """Executa uma consulta em linguagem natural"""
        payload = {
            "question": question,
            "use_cache": use_cache,
            "explain_sql": explain_sql
        }
        
        response = self.session.post(f"{self.base_url}/query/natural_language", json=payload)
        response.raise_for_status()
        return response.json()
    
    def sql_query(self, query: str, params: Optional[Dict[str, Any]] = None, use_cache: bool = True) -> Dict[str, Any]:
        """Executa uma consulta SQL direta"""
        payload = {
            "query": query,
            "params": params,
            "use_cache": use_cache
        }
        
        response = self.session.post(f"{self.base_url}/query/sql", json=payload)
        response.raise_for_status()
        return response.json()
    
    def generate_visualization(self, query_id: str, viz_type: str, options: Optional[Dict[str, Any]] = None) -> str:
        """Gera uma visualização para uma consulta"""
        payload = {
            "query_id": query_id,
            "type": viz_type,
            "options": options or {}
        }
        
        response = self.session.post(f"{self.base_url}/visualization", json=payload)
        response.raise_for_status()
        return response.text
    
    def update_catalog(self, catalog: Dict[str, Any]) -> Dict[str, Any]:
        """Atualiza o catálogo de dados"""
        response = self.session.put(f"{self.base_url}/catalog", json=catalog)
        response.raise_for_status()
        return response.json()
    
    def get_catalog(self) -> Dict[str, Any]:
        """Obtém o catálogo de dados atual"""
        response = self.session.get(f"{self.base_url}/catalog")
        response.raise_for_status()
        return response.json()
    
    def clear_cache(self) -> Dict[str, Any]:
        """Limpa o cache do sistema"""
        response = self.session.delete(f"{self.base_url}/cache")
        response.raise_for_status()
        return response.json()

def test_genbi_system():
    """
    Script de teste para demonstrar as capacidades do sistema GenBI
    """
    print("===== Teste do Sistema GenBI =====")
    
    # Configurar chave da API OpenAI via variável de ambiente
    openai_key = os.environ.get('OPENAI_API_KEY')
    if not openai_key:
        logger.error("Chave da OpenAI não configurada. Configure OPENAI_API_KEY.")
        return
    
    try:
        # Inicializar cliente
        client = GenBIClient()
        
        # 1. Verificar saúde do sistema
        print("\n1. Verificando saúde do sistema...")
        health_info = client.check_health()
        print(f"Status do Sistema: {health_info}")
        
        # 2. Consulta em linguagem natural sobre vendas
        print("\n2. Testando consulta em linguagem natural...")
        nl_result = client.natural_language_query(
            question="Qual o total de vendas por categoria de produto?",
            use_cache=True,
            explain_sql=True
        )
        
        print("Pergunta: Qual o total de vendas por categoria de produto?")
        print("ID da Consulta:", nl_result.get('query_id'))
        print("SQL Gerado:", nl_result.get('sql'))
        print("Resultados:")
        for row in nl_result.get('data', [])[:5]:
            print(row)
        
        if 'explanation' in nl_result:
            print("\nExplicação da consulta:")
            print(nl_result['explanation'])
        
        # 3. Consulta SQL direta
        print("\n3. Testando consulta SQL direta...")
        sql_result = client.sql_query("""
            SELECT 
                p.category AS categoria, 
                SUM(oi.quantity * oi.price) AS receita_total
            FROM order_items oi
            JOIN products p ON oi.product_id = p.product_id
            GROUP BY p.category
            ORDER BY receita_total DESC
        """)
        
        print("Resultados da Consulta SQL:")
        for row in sql_result.get('data', []):
            print(row)
        
        # 4. Gerar visualização
        print("\n4. Gerando visualização...")
        # Usar dados da consulta SQL
        html_viz = client.generate_visualization(
            query_id=sql_result.get('query_id'),
            viz_type="bar",
            options={
                "title": "Receita por Categoria de Produto",
                "x_column": "categoria",
                "y_column": "receita_total"
            }
        )
        
        # Salvar visualização como HTML
        os.makedirs("outputs", exist_ok=True)
        viz_path = os.path.join("outputs", "categoria_vendas.html")
        with open(viz_path, "w") as f:
            f.write(html_viz)
        print(f"Visualização salva como {viz_path}")
        
        # Abrir em navegador (opcional)
        viz_url = f"file://{os.path.abspath(viz_path)}"
        print(f"Abrindo visualização no navegador: {viz_url}")
        webbrowser.open(viz_url)
        
        # 5. Atualizar catálogo de dados
        print("\n5. Atualizando catálogo de dados...")
        catalog_update = client.update_catalog({
            "models": [
                {
                    "name": "sales_summary",
                    "description": "Resumo de vendas por categoria",
                    "source": {
                        "type": "query",
                        "value": """
                        SELECT 
                            p.category AS categoria, 
                            SUM(oi.quantity * oi.price) AS receita_total,
                            SUM(oi.quantity) AS quantidade_vendida
                        FROM order_items oi
                        JOIN products p ON oi.product_id = p.product_id
                        GROUP BY p.category
                        """
                    },
                    "columns": [
                        {"name": "categoria", "type": "string", "semanticType": "category"},
                        {"name": "receita_total", "type": "number", "semanticType": "amount"},
                        {"name": "quantidade_vendida", "type": "number", "semanticType": "count"}
                    ]
                }
            ]
        })
        print("Atualização do Catálogo:", catalog_update)
        
        # 6. Consulta sobre produtos mais vendidos
        print("\n6. Testando consulta sobre produtos mais vendidos...")
        produtos_result = client.natural_language_query(
            question="Quais são os 5 produtos mais vendidos?",
            use_cache=True
        )
        
        print("Pergunta: Quais são os 5 produtos mais vendidos?")
        print("SQL Gerado:", produtos_result.get('sql'))
        print("Resultados:")
        for row in produtos_result.get('data', []):
            print(row)
        
        # Gerar visualização para produtos mais vendidos
        html_viz_produtos = client.generate_visualization(
            query_id=produtos_result.get('query_id'),
            viz_type="bar",
            options={
                "title": "Top 5 Produtos por Quantidade Vendida",
                "x_column": "produto",
                "y_column": "quantidade_vendida"
            }
        )
        
        produtos_viz_path = os.path.join("outputs", "top_produtos_vendidos.html")
        with open(produtos_viz_path, "w") as f:
            f.write(html_viz_produtos)
        print(f"Visualização salva como {produtos_viz_path}")
        
        # Abrir em navegador (opcional)
        produtos_viz_url = f"file://{os.path.abspath(produtos_viz_path)}"
        print(f"Abrindo visualização no navegador: {produtos_viz_url}")
        webbrowser.open(produtos_viz_url)
        
    except Exception as e:
        logger.error(f"Erro ao executar teste: {str(e)}")
        traceback.print_exc()

def interactive_mode():
    """Modo interativo para o usuário fazer perguntas ao GenBI"""
    print("===== GenBI - Modo Interativo =====")
    print("Digite 'sair' para encerrar o programa.")
    
    # Configurar chave da API OpenAI via variável de ambiente
    openai_key = os.environ.get('OPENAI_API_KEY')
    if not openai_key:
        logger.error("Chave da OpenAI não configurada. Configure OPENAI_API_KEY.")
        return
    
    try:
        # Inicializar cliente
        client = GenBIClient()
        print(f"Conectado ao servidor GenBI: {BASE_URL}")
        
        while True:
            # Solicitar pergunta ao usuário
            question = input("\nDigite sua pergunta (ou 'sair' para encerrar): ").strip()
            
            if question.lower() in ['sair', 'exit', 'quit']:
                print("Encerrando o programa...")
                break
                
            if not question:
                continue
            
            # Processar pergunta
            print("\nProcessando pergunta...")
            
            try:
                result = client.natural_language_query(
                    question=question,
                    use_cache=True,
                    explain_sql=True
                )
                
                # Mostrar resultados
                print("\nSQL gerado:")
                print(result.get('sql', 'SQL não disponível'))
                
                if 'explanation' in result:
                    print("\nExplicação:")
                    print(result['explanation'])
                
                print("\nResultados:")
                data = result.get('data', [])
                if data:
                    for row in data[:10]:  # Limitar a 10 linhas para não sobrecarregar o terminal
                        print(row)
                    
                    if len(data) > 10:
                        print(f"...e mais {len(data) - 10} linhas.")
                else:
                    print("Nenhum resultado encontrado.")
                
                # Perguntar se deseja gerar visualização
                viz_choice = input("\nDeseja gerar uma visualização? (s/n): ").strip().lower()
                if viz_choice in ['s', 'sim', 'y', 'yes']:
                    # Determinar colunas disponíveis
                    if data:
                        columns = list(data[0].keys())
                        print("\nColunas disponíveis:", ", ".join(columns))
                        
                        # Perguntar tipo de visualização
                        print("\nTipos de visualização disponíveis: bar, line, pie, scatter, table")
                        viz_type = input("Tipo de visualização: ").strip().lower()
                        
                        if viz_type in ['bar', 'line', 'scatter']:
                            # Para gráficos que precisam de eixos X e Y
                            x_col = input(f"Coluna para eixo X [{columns[0]}]: ").strip() or columns[0]
                            y_col = input(f"Coluna para eixo Y [{columns[1] if len(columns) > 1 else columns[0]}]: ").strip() or (columns[1] if len(columns) > 1 else columns[0])
                            
                            options = {
                                "title": f"Visualização de {question}",
                                "x_column": x_col,
                                "y_column": y_col
                            }
                            
                        elif viz_type == 'pie':
                            # Para gráficos de pizza
                            names_col = input(f"Coluna para nomes [{columns[0]}]: ").strip() or columns[0]
                            values_col = input(f"Coluna para valores [{columns[1] if len(columns) > 1 else columns[0]}]: ").strip() or (columns[1] if len(columns) > 1 else columns[0])
                            
                            options = {
                                "title": f"Visualização de {question}",
                                "names_column": names_col,
                                "values_column": values_col
                            }
                            
                        elif viz_type == 'table':
                            # Tabela simples
                            options = {
                                "title": f"Visualização de {question}"
                            }
                            
                        else:
                            print(f"Tipo de visualização '{viz_type}' não reconhecido. Usando 'table'.")
                            viz_type = 'table'
                            options = {
                                "title": f"Visualização de {question}"
                            }
                        
                        # Gerar visualização
                        print("\nGerando visualização...")
                        try:
                            html_viz = client.generate_visualization(
                                query_id=result.get('query_id'),
                                viz_type=viz_type,
                                options=options
                            )
                            
                            # Criar nome de arquivo baseado na pergunta
                            filename = "_".join(question.lower().split()[:5]).replace("?", "").replace("/", "_")
                            viz_path = os.path.join("outputs", f"{filename}.html")
                            
                            # Salvar e abrir visualização
                            os.makedirs("outputs", exist_ok=True)
                            with open(viz_path, "w") as f:
                                f.write(html_viz)
                                
                            print(f"Visualização salva como {viz_path}")
                            
                            # Perguntar se deseja abrir no navegador
                            open_browser = input("Abrir no navegador? (s/n): ").strip().lower()
                            if open_browser in ['s', 'sim', 'y', 'yes']:
                                viz_url = f"file://{os.path.abspath(viz_path)}"
                                webbrowser.open(viz_url)
                        
                        except Exception as e:
                            print(f"Erro ao gerar visualização: {str(e)}")
                    
                    else:
                        print("Sem dados disponíveis para visualização.")
                
            except Exception as e:
                print(f"Erro ao processar pergunta: {str(e)}")
    
    except Exception as e:
        logger.error(f"Erro no modo interativo: {str(e)}")
        traceback.print_exc()

def main():
    """Função principal que gerencia os diferentes modos de execução"""
    # Parsear argumentos da linha de comando
    parser = argparse.ArgumentParser(description="GenBI - Cliente de Business Intelligence Generativo")
    parser.add_argument("--test", action="store_true", help="Executar testes automáticos")
    parser.add_argument("--interactive", action="store_true", help="Iniciar modo interativo")
    parser.add_argument("--server", action="store_true", help="Executar o servidor API")
    parser.add_argument("--url", help="URL base da API GenBI")
    parser.add_argument("--setup-db", action="store_true", help="Configurar banco de dados de exemplo")
    
    args = parser.parse_args()
    
    # Atualizar URL base se fornecida
    if args.url:
        global BASE_URL
        BASE_URL = args.url
    
    # Configurar banco de dados de exemplo
    if args.setup_db:
        print("Configurando banco de dados de exemplo...")
        try:
            from scripts.setup_sample_database import create_sample_database
            create_sample_database()
            print("Banco de dados de exemplo configurado com sucesso!")
        except Exception as e:
            logger.error(f"Erro ao configurar banco de dados: {str(e)}")
            traceback.print_exc()
    
    # Executar servidor API
    if args.server:
        try:
            print("Iniciando servidor API...")
            import uvicorn
            from app.api_server import app
            uvicorn.run(app, host="0.0.0.0", port=8000)
        except Exception as e:
            logger.error(f"Erro ao iniciar servidor: {str(e)}")
            traceback.print_exc()
    
    # Executar testes
    elif args.test:
        test_genbi_system()
    
    # Modo interativo (padrão)
    elif args.interactive or not (args.test or args.server or args.setup_db):
        interactive_mode()

if __name__ == "__main__":
    main()