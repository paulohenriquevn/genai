#!/usr/bin/env python3
"""
Exemplo do fluxo alternativo quando ocorrem falhas na LLM.

Este script demonstra como o sistema lida com falhas na geração ou execução
de consultas, oferecendo reformulação automática, sugestões de alternativas
e coletando feedback do usuário.
"""

import os
import pandas as pd
import logging
import time
import re
import json
from pprint import pprint

from core_integration import AnalysisEngine, Dataset, StringResponse

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    print("\n" + "="*70)
    print("🔀 Sistema de Consulta com Fluxo Alternativo para Falhas")
    print("="*70)
    
    # Detecta credenciais de API para LLM (OpenAI ou Anthropic)
    openai_key = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    
    print("\n🔑 Inicializando motor de análise...")
    if openai_key:
        model_type = "openai"
        model_name = "gpt-3.5-turbo"
        api_key = openai_key
        print("  🔑 Chave OpenAI encontrada. Usando modelo GPT-3.5.")
    elif anthropic_key:
        model_type = "anthropic"
        model_name = "claude-3-haiku-20240307"
        api_key = anthropic_key
        print("  🔑 Chave Anthropic encontrada. Usando modelo Claude Haiku.")
    else:
        model_type = "mock"
        model_name = None
        api_key = None
        print("  ℹ️ Nenhuma chave de API encontrada. Usando modo simulado.")
    
    # Inicializa o motor de análise
    engine = AnalysisEngine(
        agent_description="Assistente de Análise de Dados com Fluxo Alternativo para Falhas",
        default_output_type="dataframe",
        direct_sql=False,
        model_type=model_type,
        model_name=model_name,
        api_key=api_key
    )
    
    # Carrega datasets de exemplo
    print("\n📊 Carregando datasets...")
    dados_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dados")
    
    # Carrega o dataset de vendas
    vendas_path = os.path.join(dados_dir, "vendas.csv")
    engine.load_data(vendas_path, "vendas", "Registro de vendas com data, valor e cliente")
    print("  ✅ Dataset 'vendas' carregado com sucesso")
    
    # Carrega o dataset de clientes
    clientes_path = os.path.join(dados_dir, "clientes.csv")
    engine.load_data(clientes_path, "clientes", "Cadastro de clientes com nome e localização")
    print("  ✅ Dataset 'clientes' carregado com sucesso")
    
    # Carrega o dataset de vendas perdidas
    vendas_perdidas_path = os.path.join(dados_dir, "vendas_perdidas.csv")
    engine.load_data(vendas_perdidas_path, "vendas_perdidas", "Registro de oportunidades de vendas perdidas e seus motivos")
    print("  ✅ Dataset 'vendas_perdidas' carregado com sucesso")
    
    # Cria diretórios para armazenar feedback e queries bem-sucedidas
    create_storage_directories()
    
    # Demonstração do fluxo alternativo
    print("\n🔄 Demonstrando fluxo alternativo para falhas...")
    
    # 1. Consultas sobre dados inexistentes (detectadas antes da execução)
    test_missing_entity_queries(engine)
    
    # 2. Falhas com reformulação automática
    test_query_rephrasing(engine)
    
    # 3. Coleta de feedback
    test_feedback_collection(engine)
    
    # 4. Sugestões predefinidas após múltiplas falhas
    test_predefined_options(engine)
    
    # Verifica se é um ambiente interativo
    import sys
    if sys.stdin.isatty():
        # Modo interativo com feedback
        interactive_with_feedback(engine)
    else:
        print("\n💡 Modo não interativo detectado. Pulando modo interativo com feedback.")
        print("Para testar o modo interativo, execute este script em um terminal interativo.")

def create_storage_directories():
    """
    Cria diretórios para armazenar feedback e consultas bem-sucedidas.
    """
    # Cria diretório para feedback
    feedback_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_feedback")
    os.makedirs(feedback_dir, exist_ok=True)
    
    # Cria arquivo de feedback se não existir
    feedback_file = os.path.join(feedback_dir, "user_feedback.json")
    if not os.path.exists(feedback_file):
        with open(feedback_file, 'w', encoding='utf-8') as f:
            json.dump([], f)
    
    # Cria diretório para cache de consultas
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "query_cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Cria arquivo de cache se não existir
    cache_file = os.path.join(cache_dir, "successful_queries.json")
    if not os.path.exists(cache_file):
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump({}, f)
    
    print("  ✅ Diretórios de armazenamento criados")

def test_missing_entity_queries(engine):
    """
    Testa a detecção de consultas sobre entidades não existentes.
    """
    print("\n" + "-"*50)
    print("📋 DEMO 1: Detecção de consultas sobre dados inexistentes")
    print("-"*50)
    
    # Lista de consultas para testar
    queries = [
        "Quais são os produtos mais vendidos?",
        "Liste os funcionários do departamento de vendas",
        "Mostre as categorias de produtos disponíveis"
    ]
    
    print("\nEstas consultas são detectadas antes mesmo de chamar a LLM:")
    for query in queries:
        print(f"\n🔍 Consulta: \"{query}\"")
        try:
            response = engine.process_query(query)
            print(f"📝 Resposta: {response.value[:200]}...")
        except Exception as e:
            print(f"❌ Erro: {str(e)}")

def test_query_rephrasing(engine):
    """
    Testa a reformulação automática de consultas.
    """
    print("\n" + "-"*50)
    print("📋 DEMO 2: Reformulação automática de consultas")
    print("-"*50)
    
    # Consultas que precisariam ser reformuladas
    problematic_queries = [
        "Quais vendas tiveram o maior desconto?",  # Coluna inexistente
        "Mostrar vendas por região ordenadas pelo número de itens",  # Coluna inexistente
        "Qual o faturamento por trimestre?",  # Conceito não diretamente mapeado
    ]
    
    print("\nEstas consultas seriam automaticamente reformuladas:")
    for i, query in enumerate(problematic_queries, 1):
        print(f"\n🔍 Consulta original: \"{query}\"")
        
        # Tenta executar a consulta (limitando a apenas uma tentativa de reformulação)
        try:
            response = engine.process_query(query, max_retries=1)
            print(f"✅ Resposta após possível reformulação: {type(response).__name__}")
            if hasattr(response, 'value'):
                if isinstance(response.value, pd.DataFrame):
                    print(f"📊 Resultado ({min(len(response.value), 3)} linhas):")
                    print(response.value.head(3))
                else:
                    print(f"📝 Resultado: {str(response.value)[:150]}...")
        except Exception as e:
            print(f"❌ Erro após tentativa de reformulação: {str(e)}")
            
        # Exemplo do que deveria acontecer internamente
        if i == 1:
            print(f"🔄 Internamente: Reformulação para \"Quais vendas tiveram o maior valor?\"")
        elif i == 2:
            print(f"🔄 Internamente: Reformulação para \"Mostrar vendas por região ordenadas pelo valor total\"")
        else:
            print(f"🔄 Internamente: Reformulação para \"Qual o faturamento agrupado por mês?\"")

def test_feedback_collection(engine):
    """
    Testa a coleta de feedback do usuário.
    """
    print("\n" + "-"*50)
    print("📋 DEMO 3: Coleta de feedback do usuário")
    print("-"*50)
    
    query = "Qual cliente tem o maior ticket médio?"
    feedback = "Também gostaria de ver o número de compras de cada cliente"
    
    print(f"\n🔍 Consulta original: \"{query}\"")
    print(f"💬 Feedback do usuário: \"{feedback}\"")
    
    # Armazena o feedback manualmente para demonstração
    try:
        # Cria diretório para feedback
        feedback_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_feedback")
        os.makedirs(feedback_dir, exist_ok=True)
        
        # Armazena em um arquivo JSON
        feedback_file = os.path.join(feedback_dir, "user_feedback.json")
        
        # Carrega o feedback existente
        existing_feedback = []
        if os.path.exists(feedback_file):
            with open(feedback_file, 'r', encoding='utf-8') as f:
                existing_feedback = json.load(f)
        
        # Adiciona o novo feedback
        existing_feedback.append({
            "timestamp": time.time(),
            "query": query,
            "feedback": feedback
        })
        
        # Salva o feedback atualizado
        with open(feedback_file, 'w', encoding='utf-8') as f:
            json.dump(existing_feedback, f, indent=2, ensure_ascii=False)
        
        print("✅ Feedback armazenado com sucesso para demonstração")
    except Exception as e:
        print(f"❌ Erro ao armazenar feedback: {str(e)}")
    
    # Executa a consulta com o feedback
    try:
        print("\n🔄 Processando a consulta com feedback...")
        response = engine.process_query_with_feedback(query, feedback)
        
        print(f"✅ Resposta com feedback: {type(response).__name__}")
        if hasattr(response, 'value'):
            if isinstance(response.value, pd.DataFrame):
                print(f"📊 Resultado ({min(len(response.value), 3)} linhas):")
                print(response.value.head(3))
            else:
                print(f"📝 Resultado: {str(response.value)[:150]}...")
                
        # Verifica o armazenamento do feedback
        feedback_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_feedback")
        feedback_file = os.path.join(feedback_dir, "user_feedback.json")
        
        if os.path.exists(feedback_file):
            with open(feedback_file, 'r', encoding='utf-8') as f:
                feedback_data = json.load(f)
            
            print(f"\n📝 Feedback armazenado ({len(feedback_data)} registros no total)")
            # Mostra o último feedback adicionado
            if feedback_data:
                latest = feedback_data[-1]
                print(f"  Consulta: \"{latest.get('query', '')}\"")
                print(f"  Feedback: \"{latest.get('feedback', '')}\"")
        
    except Exception as e:
        print(f"❌ Erro ao processar consulta com feedback: {str(e)}")

def test_predefined_options(engine):
    """
    Testa a oferta de opções predefinidas após múltiplas falhas.
    """
    print("\n" + "-"*50)
    print("📋 DEMO 4: Sugestões predefinidas após múltiplas falhas")
    print("-"*50)
    
    query = "Compare o desempenho de vendas entre os diferentes segmentos de mercado"
    
    print(f"\n🔍 Consulta original: \"{query}\"")
    
    # Tenta processar a consulta com várias tentativas até esgotar as reformulações
    try:
        # Força 3 tentativas para demonstrar o comportamento
        print("🔄 Tentando processar a consulta com múltiplas reformulações...")
        response = engine.process_query(query, max_retries=3)
        print(f"✅ Resposta final: {type(response).__name__}")
        
        if hasattr(response, 'value'):
            print(f"📝 Resultado: {str(response.value)[:200]}...")
    except Exception as e:
        print(f"❌ Erro após todas as tentativas: {str(e)}")
    
    # Gera opções predefinidas usando o método interno do engine
    print("\n🎯 Gerando opções predefinidas...")
    try:
        alternatives = engine._generate_alternative_queries()
        
        print("\n🔄 O sistema oferece estas alternativas:")
        for i, alt in enumerate(alternatives[:5], 1):
            print(f"{i}. {alt}")
    except Exception as e:
        print(f"❌ Erro ao gerar alternativas: {str(e)}")
        
        # Mostra exemplos fixos caso o método falhe
        alternatives = [
            "Mostre um resumo do dataset vendas",
            "Compare as vendas entre diferentes clientes",
            "Analise as vendas por região geográfica",
            "Quais são os principais motivos de vendas perdidas?",
            "Qual o valor médio de vendas por cliente?"
        ]
        
        print("\n🔄 Exemplos de alternativas que seriam oferecidas:")
        for i, alt in enumerate(alternatives, 1):
            print(f"{i}. {alt}")

def interactive_with_feedback(engine):
    """
    Modo interativo com feedback do usuário.
    """
    print("\n" + "-"*50)
    print("💬 MODO INTERATIVO COM FEEDBACK")
    print("-"*50)
    
    print("\nEste modo permite testar consultas reais e fornecer feedback para melhorias.")
    print("Para fornecer feedback após uma consulta, digite 'feedback:' seguido do seu comentário.")
    print("Digite 'sair' para encerrar.")
    
    last_query = None
    feedback = None
    
    while True:
        print()
        user_input = input("Digite sua consulta ou feedback: ").strip()
        
        if user_input.lower() in ['sair', 'exit', 'quit']:
            break
            
        # Verifica se é um feedback
        if user_input.lower().startswith('feedback:'):
            if last_query:
                feedback = user_input[len('feedback:'):].strip()
                print(f"✅ Feedback registrado: \"{feedback}\"")
                
                # Executa novamente a última consulta com o feedback
                print("\n🔄 Processando a consulta com o feedback...")
                process_query_safely(engine, last_query, feedback)
            else:
                print("❌ Não há consulta anterior para fornecer feedback.")
        else:
            # É uma nova consulta
            last_query = user_input
            process_query_safely(engine, user_query=last_query, feedback=None)

def process_query_safely(engine, user_query, feedback=None):
    """
    Processa uma consulta com tratamento de exceções para evitar crashes no exemplo.
    
    Args:
        engine: O motor de análise
        user_query: A consulta do usuário
        feedback: Feedback opcional do usuário
    """
    try:
        # Imprime informações sobre a consulta
        print(f"📝 Consulta: {user_query}")
        if feedback:
            print(f"💬 Feedback: {feedback}")
            
        # Processa a consulta (com feedback se disponível)
        if feedback:
            response = engine.process_query_with_feedback(user_query, feedback)
        else:
            response = engine.process_query(user_query)
        
        # Exibe o tipo de resposta
        print(f"🔄 Tipo de resposta: {response.type}")
        
        # Exibe o resultado baseado no tipo
        if response.type == "dataframe":
            # Limita a exibição para datasets grandes
            df = response.value
            max_rows = 5
            print(f"📊 Resultado ({min(len(df), max_rows)} linhas):")
            print(df.head(max_rows))
        elif response.type == "number":
            print(f"🔢 Resultado numérico: {response.value}")
        elif response.type == "string":
            print(f"📝 Resultado textual: {response.value}")
        elif response.type == "plot":
            print(f"📈 Visualização gerada: {response.value}")
        elif response.type == "error":
            print(f"❌ Erro: {response.value}")
            
            # Mostra opções alternativas em caso de erro
            alternatives = engine._generate_alternative_queries()
            
            print("\n🎯 Você pode tentar estas alternativas:")
            for i, alt in enumerate(alternatives[:3], 1):
                print(f"{i}. {alt}")
        
    except Exception as e:
        print(f"❌ Erro ao processar a consulta: {str(e)}")
        
        # Oferece alternativas mesmo em caso de exceção
        try:
            alternatives = engine._generate_alternative_queries()
            
            print("\n🎯 Você pode tentar estas alternativas:")
            for i, alt in enumerate(alternatives[:3], 1):
                print(f"{i}. {alt}")
        except:
            print("\n🎯 Tente uma consulta mais simples, como 'Mostre um resumo dos dados disponíveis'")
        
if __name__ == "__main__":
    main()