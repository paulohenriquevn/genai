#!/usr/bin/env python3
"""
OpenAI Data Analyzer
====================

Este módulo implementa um analisador de dados completo que utiliza a API OpenAI
para gerar perguntas analíticas, executar consultas em linguagem natural,
e produzir visualizações e relatórios de análise.

Principais recursos:
1. Geração de perguntas analíticas usando OpenAI
2. Processamento de consultas em linguagem natural
3. Visualização automatizada de dados
4. Geração de relatórios HTML com análises e visualizações
"""

import os
import sys
import json
import time
import argparse
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import base64
from pathlib import Path
import webbrowser

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
from natural_query_engine import NaturalLanguageQueryEngine
from llm_integration import LLMIntegration, ModelType, LLMQueryGenerator
from core.response.base import BaseResponse
from core.response.dataframe import DataFrameResponse
from core.response.number import NumberResponse
from core.response.string import StringResponse
from core.response.chart import ChartResponse
from core.response.error import ErrorResponse


class OpenAIAnalyzer:
    """
    Analisador de dados que utiliza a API OpenAI para gerar consultas,
    análises e visualizações de dados.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4",
        data_dir: Optional[str] = None,
        output_dir: Optional[str] = None
    ):
        """
        Inicializa o analisador de dados com OpenAI.
        
        Args:
            api_key: Chave de API do OpenAI (opcional, pode usar variável de ambiente)
            model: Modelo OpenAI a ser utilizado
            data_dir: Diretório onde os dados estão armazenados
            output_dir: Diretório para salvar relatórios e visualizações
        """
        # Configura a API do OpenAI
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("Chave de API OpenAI não fornecida. Defina OPENAI_API_KEY ou passe como parâmetro.")
        
        openai.api_key = self.api_key
        self.model = model
        
        # Define diretórios
        self.data_dir = data_dir or os.path.join(os.getcwd(), "dados")
        self.output_dir = output_dir or os.path.join(os.getcwd(), "output")
        
        # Cria diretório de saída se não existir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Inicializa o motor de consulta em linguagem natural
        self.query_engine = None
        self._initialize_query_engine()
        
        # Contador de análises
        self.analysis_count = 0
        
        # Lista para armazenar resultados de análises
        self.analysis_results = []
        
        logger.info(f"Analisador OpenAI inicializado com modelo {model}")
    
    def _initialize_query_engine(self):
        """
        Inicializa o motor de consulta em linguagem natural.
        Carrega dados de exemplo se não houver configuração específica.
        """
        try:
            # Verifica se há um arquivo de configuração
            datasources_path = os.path.join(self.data_dir, "datasources.json")
            
            if os.path.exists(datasources_path):
                # Inicializa com arquivo de configuração
                self.query_engine = NaturalLanguageQueryEngine(
                    data_config_path=datasources_path,
                    base_data_path=self.data_dir
                )
            else:
                # Inicializa sem arquivo de configuração
                self.query_engine = NaturalLanguageQueryEngine(
                    base_data_path=self.data_dir
                )
                
                # Carrega dados de exemplo manualmente
                self._load_sample_data()
            
            logger.info(f"Motor de consulta inicializado com {len(self.query_engine.dataframes)} datasets")
        except Exception as e:
            logger.error(f"Erro ao inicializar o motor de consulta: {str(e)}")
            raise
    
    def _load_sample_data(self):
        """
        Carrega dados de exemplo para análise, incluindo clientes, vendas e vendas perdidas.
        """
        try:
            # Caminhos para os arquivos CSV
            clientes_path = os.path.join(self.data_dir, "clientes.csv")
            vendas_path = os.path.join(self.data_dir, "vendas.csv")
            perdidas_path = os.path.join(self.data_dir, "vendas_perdidas.csv")
            
            # Carrega os dados se existirem
            if os.path.exists(clientes_path):
                clientes_df = pd.read_csv(clientes_path)
                self.query_engine.load_data(
                    data=clientes_df,
                    name="clientes",
                    description="Dados de clientes"
                )
                logger.info(f"Dados de clientes carregados: {len(clientes_df)} registros")
            
            if os.path.exists(vendas_path):
                vendas_df = pd.read_csv(vendas_path)
                
                # Converte coluna de data para datetime, se existir
                if "data" in vendas_df.columns:
                    vendas_df["data"] = pd.to_datetime(vendas_df["data"])
                
                self.query_engine.load_data(
                    data=vendas_df,
                    name="vendas",
                    description="Dados de vendas realizadas"
                )
                logger.info(f"Dados de vendas carregados: {len(vendas_df)} registros")
            
            if os.path.exists(perdidas_path):
                perdidas_df = pd.read_csv(perdidas_path)
                self.query_engine.load_data(
                    data=perdidas_df,
                    name="vendas_perdidas",
                    description="Dados de vendas perdidas"
                )
                logger.info(f"Dados de vendas perdidas carregados: {len(perdidas_df)} registros")
            
            # Se nenhum dado foi carregado, gera dados de exemplo
            if not self.query_engine.dataframes:
                logger.warning("Nenhum dado encontrado. Criando dados sintéticos para exemplo.")
                self._create_synthetic_data()
        
        except Exception as e:
            logger.error(f"Erro ao carregar dados de exemplo: {str(e)}")
            raise
    
    def _create_synthetic_data(self):
        """
        Cria dados sintéticos para demonstração quando não há dados disponíveis.
        """
        # Define seed para reprodutibilidade
        np.random.seed(42)
        
        # Cria DataFrame de clientes
        clientes_df = pd.DataFrame({
            'id_cliente': range(1, 11),
            'nome': [f'Cliente {i}' for i in range(1, 11)],
            'cidade': np.random.choice(['São Paulo', 'Rio de Janeiro', 'Belo Horizonte', 'Curitiba', 'Brasília'], 10),
            'segmento': np.random.choice(['Varejo', 'Corporativo', 'Governo'], 10)
        })
        
        # Cria DataFrame de vendas
        vendas_df = pd.DataFrame({
            'id_venda': range(1, 101),
            'data': pd.date_range(start='2023-01-01', periods=100),
            'valor': np.random.uniform(100, 1000, 100).round(2),
            'id_cliente': np.random.randint(1, 11, 100),
            'id_produto': np.random.randint(1, 6, 100)
        })
        
        # Cria DataFrame de vendas perdidas
        vendas_perdidas_df = pd.DataFrame({
            'OportunidadeID': [f'PERDA-{i:03d}' for i in range(1, 51)],
            'Motivo': np.random.choice(['Preço', 'Concorrência', 'Atendimento'], 50),
            'EstagioPerda': np.random.choice(['Proposta', 'Negociação'], 50),
            'ImpactoFinanceiro': np.random.uniform(3000, 20000, 50).round(2),
            'FeedbackCliente': [f'Feedback {i}' for i in range(1, 51)]
        })
        
        # Carrega os DataFrames no motor de consulta
        self.query_engine.load_data(
            data=clientes_df,
            name="clientes",
            description="Dados de clientes"
        )
        
        self.query_engine.load_data(
            data=vendas_df,
            name="vendas",
            description="Dados de vendas realizadas"
        )
        
        self.query_engine.load_data(
            data=vendas_perdidas_df,
            name="vendas_perdidas",
            description="Dados de vendas perdidas"
        )
        
        logger.info("Dados sintéticos criados com sucesso")
    
    def generate_analysis_questions(self, topic: str, num_questions: int = 5) -> List[str]:
        """
        Gera perguntas analíticas usando a API do OpenAI.
        
        Args:
            topic: Tópico para análise (ex: "vendas", "clientes", "vendas perdidas")
            num_questions: Número de perguntas a serem geradas
            
        Returns:
            Lista de perguntas analíticas
        """
        logger.info(f"Gerando {num_questions} perguntas sobre {topic}")
        
        # Cria descrição dos dados disponíveis
        dfs_info = "\n".join([
            f"- {name}: {df.description} ({len(df.dataframe)} registros, colunas: {', '.join(df.dataframe.columns)})" 
            for name, df in self.query_engine.dataframes.items()
        ])
        
        # Monta o prompt
        prompt = f"""
        Você é um especialista em análise de dados. Gere {num_questions} perguntas analíticas sobre {topic} 
        que podem ser respondidas usando os seguintes dados:
        
        {dfs_info}
        
        As perguntas devem ser diversificadas e incluir análises:
        1. Descritivas (contagens, somas, médias)
        2. Comparativas (comparações entre grupos)
        3. Visuais (solicitar gráficos, visualizações)
        4. Tendências (análises temporais, se aplicável)
        5. Correlações (relações entre variáveis)
        
        Cada pergunta deve ser clara, específica e possível de responder com os dados disponíveis.
        Forneça apenas as perguntas, uma por linha, sem numeração ou explicações adicionais.
        """
        
        try:
            # Chama a API do OpenAI
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Você é um assistente especializado em análise de dados."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            # Processa a resposta
            questions_text = response.choices[0].message.content.strip()
            
            # Separa as perguntas linha por linha
            questions = [q.strip() for q in questions_text.split('\n') if q.strip()]
            
            # Limita ao número solicitado
            questions = questions[:num_questions]
            
            logger.info(f"Geradas {len(questions)} perguntas analíticas sobre {topic}")
            return questions
            
        except Exception as e:
            logger.error(f"Erro ao gerar perguntas analíticas: {str(e)}")
            # Retorna algumas perguntas básicas como fallback
            return [
                f"Qual é o total de {topic}?",
                f"Quais são os principais {topic} por valor?",
                f"Mostre um gráfico de {topic} por categoria",
                f"Qual a média de {topic}?",
                f"Como os {topic} estão distribuídos?"
            ][:num_questions]
    
    def analyze_query_result(self, query: str, result: BaseResponse) -> str:
        """
        Analisa o resultado de uma consulta usando OpenAI para gerar insights.
        
        Args:
            query: A consulta original
            result: O resultado da consulta (BaseResponse ou subclasse)
            
        Returns:
            Análise do resultado em texto
        """
        try:
            # Prepara a descrição do resultado para o OpenAI
            result_description = self._format_result_for_analysis(result)
            
            # Determina se é uma consulta sobre vendas perdidas
            is_perdas_query = "perdidas" in query.lower() or "perda" in query.lower()
            
            # Define o contexto específico para vendas perdidas
            context = ""
            if is_perdas_query:
                context = """
                Contexto sobre Vendas Perdidas:
                - As vendas perdidas representam oportunidades de negócio que não se concretizaram
                - O 'Motivo' indica a razão principal pela qual a venda foi perdida (ex: Preço, Concorrência, Atendimento)
                - O 'EstagioPerda' indica em qual fase do processo de venda a oportunidade foi perdida (ex: Proposta, Negociação)
                - O 'ImpactoFinanceiro' representa o valor que seria obtido se a venda tivesse sido concretizada
                - O 'FeedbackCliente' contém comentários dos clientes sobre por que não seguiram com a compra
                
                Foco de análise:
                - Identificar padrões de perda de vendas por motivo e estágio
                - Quantificar o impacto financeiro das perdas
                - Correlacionar estágio da perda com impacto financeiro
                - Avaliar se perdas em estágios mais avançados têm maior impacto financeiro
                - Extrair insights sobre como reduzir vendas perdidas e aumentar conversão
                """
            
            # Monta o prompt
            prompt = f"""
            Analise o resultado da seguinte consulta:
            
            CONSULTA: "{query}"
            
            {context}
            
            RESULTADO:
            {result_description}
            
            Forneça uma análise concisa e detalhada sobre esses dados, incluindo:
            1. Principais padrões ou tendências claramente visíveis nos dados
            2. Insights específicos e acionáveis que podem ser extraídos desses resultados
            3. Possíveis explicações baseadas em fatos para os resultados observados
            4. Recomendações concretas e específicas baseadas nos dados
            
            IMPORTANTE:
            - Sua análise deve ser baseada exclusivamente nos dados apresentados
            - Cite números e estatísticas específicas presentes nos dados
            - Evite generalizações vagas
            - Seja específico e factual (ex: "O motivo X representa 35% das vendas perdidas" em vez de "Alguns motivos são mais frequentes")
            - Se o resultado for uma visualização, descreva os padrões visíveis no gráfico
            - Estruture sua resposta em parágrafos claros e concisos
            
            Sua análise deve ser objetiva e ter no máximo 4 parágrafos.
            """
            
            # Chama a API do OpenAI
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Você é um analista de dados especializado em gerar insights valiosos e acionáveis a partir de dados de negócios."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,  # Temperatura mais baixa para respostas mais factuais
                max_tokens=750
            )
            
            # Extrai a análise
            analysis = response.choices[0].message.content.strip()
            
            logger.info(f"Análise gerada para a consulta: '{query[:50]}...'")
            return analysis
            
        except Exception as e:
            logger.error(f"Erro ao analisar resultado da consulta: {str(e)}")
            return f"Não foi possível gerar uma análise detalhada devido a um erro: {str(e)}"
    
    def _format_result_for_analysis(self, result: BaseResponse) -> str:
        """
        Formata o resultado de uma consulta para análise.
        
        Args:
            result: O resultado da consulta
            
        Returns:
            Descrição formatada do resultado
        """
        if isinstance(result, DataFrameResponse):
            # Formata DataFrame para análise
            df = result.value
            
            # Extrai informações estatísticas relevantes
            try:
                # Identifica colunas numéricas
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                
                # Gera estatísticas para colunas numéricas
                stats = ""
                if numeric_cols:
                    stats_df = df[numeric_cols].describe().round(2)
                    stats = f"\nEstatísticas das colunas numéricas:\n{stats_df.to_string()}\n"
                
                # Contagem de valores para colunas categóricas (primeiras 2 colunas)
                categorical_stats = ""
                categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()[:2]
                for col in categorical_cols:
                    if len(df[col].unique()) < 15:  # Só para colunas com poucos valores únicos
                        value_counts = df[col].value_counts().head(10)
                        categorical_stats += f"\nDistribuição de valores em '{col}':\n{value_counts.to_string()}\n"
                
                # Formata o resultado completo
                if len(df) > 20:
                    description = f"DataFrame com {len(df)} registros. Primeiras 10 linhas:\n{df.head(10).to_string()}\n"
                    description += stats
                    description += categorical_stats
                else:
                    description = f"DataFrame com {len(df)} registros:\n{df.to_string()}\n"
                    description += stats
                    description += categorical_stats
                
            except Exception as e:
                # Em caso de erro, usa formato simples
                if len(df) > 20:
                    description = f"DataFrame com {len(df)} registros. Primeiras 10 linhas:\n{df.head(10).to_string()}\n"
                else:
                    description = f"DataFrame com {len(df)} registros:\n{df.to_string()}"
            
            return description
            
        elif isinstance(result, NumberResponse):
            # Formata valor numérico
            return f"Valor numérico: {result.value}"
            
        elif isinstance(result, StringResponse):
            # Formata valor de texto
            return f"Texto: {result.value}"
            
        elif isinstance(result, ChartResponse):
            # Para visualizações, fornecer contexto mais rico sobre o que provavelmente está representado
            chart_context = "Visualização gerada como resultado da consulta."
            
            # Analisa o último código executado, se disponível
            if hasattr(result, 'last_code_executed') and result.last_code_executed:
                chart_context += f"\n\nCódigo que gerou a visualização:\n{result.last_code_executed}"
            
            # Tentar inferir o tipo de visualização a partir da última consulta executada
            chart_type = "gráfico"
            
            # Descreve os possíveis tipos de dados visualizados
            if hasattr(self.query_engine, 'last_query') and "vendas perdidas" in self.query_engine.last_query.lower():
                query = self.query_engine.last_query.lower()
                
                if "motivo" in query:
                    chart_context += "\n\nEsta visualização provavelmente mostra dados relacionados aos motivos de vendas perdidas."
                    
                    if "barras" in query:
                        chart_context += " É um gráfico de barras que mostra a quantidade ou impacto financeiro por motivo de perda."
                    elif "pizza" in query:
                        chart_context += " É um gráfico de pizza que mostra a distribuição percentual dos motivos de perda."
                
                elif "estágio" in query or "estagio" in query:
                    chart_context += "\n\nEsta visualização provavelmente mostra dados relacionados aos estágios de vendas perdidas."
                    
                    if "barras" in query:
                        chart_context += " É um gráfico de barras que mostra a quantidade de vendas perdidas por estágio."
                
                elif "impacto" in query:
                    chart_context += "\n\nEsta visualização provavelmente mostra dados relacionados ao impacto financeiro das vendas perdidas."
                    
                    if "histograma" in query:
                        chart_context += " É um histograma que mostra a distribuição dos valores de impacto financeiro."
                
                elif "tempo" in query or "tendência" in query or "tendencia" in query or "evolução" in query or "linha" in query:
                    chart_context += "\n\nEsta visualização provavelmente mostra uma tendência temporal do impacto financeiro das vendas perdidas."
                    chart_context += " É um gráfico de linha que mostra a evolução das perdas ao longo do tempo."
                
                elif "correlação" in query or "correlacao" in query or "relação" in query:
                    chart_context += "\n\nEsta visualização provavelmente mostra a relação entre estágio de perda e impacto financeiro."
                    chart_context += " É um boxplot ou gráfico de dispersão que mostra a distribuição de impacto financeiro por estágio."
            
            chart_context += "\n\nAnalise os padrões visíveis no gráfico, tendências, outliers e distribuições para extrair insights sobre os dados visualizados."
            
            return chart_context
            
        elif isinstance(result, ErrorResponse):
            # Formata mensagem de erro
            return f"Erro: {result.value}"
            
        else:
            # Tipo de resposta desconhecido
            return f"Resultado de tipo {type(result).__name__} com valor: {str(result.value)}"
    
    def run_query(self, query: str) -> Tuple[BaseResponse, str]:
        """
        Executa uma consulta e gera análise do resultado.
        
        Args:
            query: Consulta em linguagem natural
            
        Returns:
            Tupla com (resultado da consulta, análise do resultado)
        """
        logger.info(f"Executando consulta: {query}")
        
        try:
            # Executa a consulta
            result = self.query_engine.execute_query(query)
            
            # Gera análise do resultado
            analysis = self.analyze_query_result(query, result)
            
            return result, analysis
            
        except Exception as e:
            logger.error(f"Erro ao executar consulta: {str(e)}")
            error_response = ErrorResponse(f"Erro ao processar consulta: {str(e)}")
            return error_response, f"Não foi possível analisar o resultado devido a um erro: {str(e)}"
    
    def save_visualization(self, chart_response: ChartResponse, filename: str) -> str:
        """
        Salva uma visualização em arquivo.
        
        Args:
            chart_response: Resposta contendo a visualização
            filename: Nome base do arquivo (sem extensão)
            
        Returns:
            Caminho completo do arquivo salvo
        """
        # Cria pasta para visualizações se não existir
        visualizations_dir = os.path.join(self.output_dir, "visualizations")
        os.makedirs(visualizations_dir, exist_ok=True)
        
        # Constrói o caminho completo
        output_path = os.path.join(visualizations_dir, f"{filename}.png")
        
        # Salva a visualização
        chart_response.save(output_path)
        
        logger.info(f"Visualização salva em: {output_path}")
        return output_path
    
    def run_analysis(self, topic: str, num_questions: int = 5) -> dict:
        """
        Executa uma análise completa com base em um tópico.
        
        Args:
            topic: Tópico para análise (ex: "vendas", "clientes", "vendas_perdidas")
            num_questions: Número de perguntas a serem geradas
            
        Returns:
            Dicionário com os resultados da análise
        """
        # Define o ID e a data da análise
        analysis_id = f"analysis_{self.analysis_count}"
        self.analysis_count += 1
        
        analysis_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Estrutura para armazenar resultados
        analysis_results = {
            "id": analysis_id,
            "topic": topic,
            "timestamp": analysis_time,
            "queries": [],
            "summary": "",
            "visualizations": []
        }
        
        # Gera perguntas analíticas
        questions = self.generate_analysis_questions(topic, num_questions)
        
        # Executa cada pergunta
        for i, question in enumerate(questions):
            try:
                # Executa a consulta
                result, analysis = self.run_query(question)
                
                # Registra resultado e análise
                query_result = {
                    "query": question,
                    "result_type": type(result).__name__,
                    "analysis": analysis,
                    "visualization_path": None
                }
                
                # Se o resultado for uma visualização, salva
                if isinstance(result, ChartResponse):
                    viz_filename = f"{analysis_id}_viz_{i}"
                    viz_path = self.save_visualization(result, viz_filename)
                    query_result["visualization_path"] = viz_path
                    analysis_results["visualizations"].append({
                        "path": viz_path,
                        "query": question,
                        "analysis": analysis
                    })
                
                analysis_results["queries"].append(query_result)
                
                # Pausa para evitar muitas chamadas à API em sequência
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Erro ao processar pergunta '{question}': {str(e)}")
                analysis_results["queries"].append({
                    "query": question,
                    "result_type": "Error",
                    "analysis": f"Erro ao processar esta consulta: {str(e)}",
                    "visualization_path": None
                })
        
        # Gera um resumo geral da análise
        summary = self._generate_analysis_summary(analysis_results)
        analysis_results["summary"] = summary
        
        # Salva os resultados
        self._save_analysis_results(analysis_results)
        
        # Gera relatório HTML
        report_path = self.generate_html_report(analysis_results)
        analysis_results["report_path"] = report_path
        
        # Armazena os resultados
        self.analysis_results.append(analysis_results)
        
        return analysis_results
    
    def _generate_analysis_summary(self, analysis_results: dict) -> str:
        """
        Gera um resumo geral da análise usando OpenAI.
        
        Args:
            analysis_results: Resultados da análise
            
        Returns:
            Resumo da análise
        """
        try:
            # Compila as análises individuais
            analyses = [f"Consulta: {q['query']}\nAnálise: {q['analysis']}" 
                       for q in analysis_results["queries"]]
            
            all_analyses = "\n\n".join(analyses)
            
            # Monta o prompt
            prompt = f"""
            Com base nas seguintes análises individuais sobre "{analysis_results['topic']}", 
            crie um resumo geral conciso que sintetize os principais insights e conclusões:
            
            {all_analyses}
            
            O resumo deve:
            1. Destacar padrões e tendências consistentes entre as análises
            2. Identificar os insights mais importantes e relevantes
            3. Fornecer uma visão consolidada dos dados analisados
            4. Sugerir possíveis ações baseadas nas descobertas
            
            Limite o resumo a no máximo 3 parágrafos.
            """
            
            # Chama a API do OpenAI
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Você é um especialista em síntese de análises de dados."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=700
            )
            
            # Extrai o resumo
            summary = response.choices[0].message.content.strip()
            
            logger.info(f"Resumo geral gerado para análise sobre '{analysis_results['topic']}'")
            return summary
            
        except Exception as e:
            logger.error(f"Erro ao gerar resumo da análise: {str(e)}")
            return "Não foi possível gerar um resumo consolidado da análise."
    
    def _save_analysis_results(self, analysis_results: dict) -> str:
        """
        Salva os resultados da análise em formato JSON.
        
        Args:
            analysis_results: Resultados da análise
            
        Returns:
            Caminho do arquivo JSON
        """
        # Cria pasta para resultados se não existir
        results_dir = os.path.join(self.output_dir, "results")
        os.makedirs(results_dir, exist_ok=True)
        
        # Constrói o caminho do arquivo
        filename = f"{analysis_results['id']}_{analysis_results['topic'].replace(' ', '_')}.json"
        output_path = os.path.join(results_dir, filename)
        
        # Cria uma cópia dos resultados para serialização
        results_copy = analysis_results.copy()
        
        # Serializa o resultado
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_copy, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Resultados da análise salvos em: {output_path}")
        return output_path
    
    def generate_html_report(self, analysis_results: dict) -> str:
        """
        Gera um relatório HTML com os resultados da análise.
        
        Args:
            analysis_results: Resultados da análise
            
        Returns:
            Caminho do arquivo HTML
        """
        # Cria pasta para relatórios se não existir
        reports_dir = os.path.join(self.output_dir, "reports")
        os.makedirs(reports_dir, exist_ok=True)
        
        # Constrói o caminho do arquivo
        filename = f"{analysis_results['id']}_{analysis_results['topic'].replace(' ', '_')}.html"
        output_path = os.path.join(reports_dir, filename)
        
        # Prepara as seções do relatório
        visualizations_html = ""
        for viz in analysis_results["visualizations"]:
            # Converte a imagem para base64 para incluir no HTML
            try:
                with open(viz["path"], "rb") as img_file:
                    img_data = base64.b64encode(img_file.read()).decode('utf-8')
                    
                visualizations_html += f"""
                <div class="visualization-card">
                    <h3>Consulta: {viz["query"]}</h3>
                    <div class="viz-container">
                        <img src="data:image/png;base64,{img_data}" alt="Visualização" />
                    </div>
                    <div class="analysis-container">
                        <h4>Análise:</h4>
                        <p>{viz["analysis"]}</p>
                    </div>
                </div>
                """
            except Exception as e:
                logger.error(f"Erro ao processar visualização para relatório: {str(e)}")
                visualizations_html += f"""
                <div class="visualization-card error">
                    <h3>Consulta: {viz["query"]}</h3>
                    <p class="error-message">Erro ao processar visualização: {str(e)}</p>
                </div>
                """
        
        # Prepara a lista de consultas e análises
        queries_html = ""
        for query in analysis_results["queries"]:
            viz_img = ""
            if query.get("visualization_path"):
                # Se tiver visualização, não mostrar novamente (já está na seção de visualizações)
                viz_img = "<p><em>Visualização disponível na seção de gráficos.</em></p>"
            
            queries_html += f"""
            <div class="query-card">
                <h3>Consulta: {query["query"]}</h3>
                <p><strong>Tipo de resultado:</strong> {query["result_type"]}</p>
                {viz_img}
                <div class="analysis-container">
                    <h4>Análise:</h4>
                    <p>{query["analysis"]}</p>
                </div>
            </div>
            """
        
        # Template HTML
        html_template = f"""<!DOCTYPE html>
        <html lang="pt-br">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Análise de {analysis_results["topic"]} - Relatório</title>
            <style>
                :root {{
                    --primary-color: #2c3e50;
                    --secondary-color: #3498db;
                    --accent-color: #e74c3c;
                    --background-color: #f9f9f9;
                    --card-bg-color: #ffffff;
                    --text-color: #333333;
                }}
                
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    color: var(--text-color);
                    background-color: var(--background-color);
                    margin: 0;
                    padding: 0;
                }}
                
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                
                header {{
                    background-color: var(--primary-color);
                    color: white;
                    padding: 20px;
                    text-align: center;
                    border-radius: 5px 5px 0 0;
                }}
                
                .content {{
                    padding: 20px;
                }}
                
                h1, h2, h3, h4 {{
                    color: var(--primary-color);
                }}
                
                .summary-card {{
                    background-color: var(--card-bg-color);
                    border-radius: 5px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    padding: 20px;
                    margin-bottom: 20px;
                }}
                
                .visualization-section, .queries-section {{
                    margin-top: 40px;
                }}
                
                .visualization-card, .query-card {{
                    background-color: var(--card-bg-color);
                    border-radius: 5px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    padding: 20px;
                    margin-bottom: 20px;
                }}
                
                .visualization-card h3, .query-card h3 {{
                    color: var(--secondary-color);
                    border-bottom: 1px solid #eee;
                    padding-bottom: 10px;
                }}
                
                .viz-container {{
                    text-align: center;
                    margin: 20px 0;
                }}
                
                .viz-container img {{
                    max-width: 100%;
                    border-radius: 5px;
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                }}
                
                .analysis-container {{
                    background-color: #f5f8fa;
                    border-radius: 5px;
                    padding: 15px;
                    margin-top: 15px;
                }}
                
                .analysis-container h4 {{
                    margin-top: 0;
                    color: var(--secondary-color);
                }}
                
                .error-message {{
                    color: var(--accent-color);
                    font-style: italic;
                }}
                
                footer {{
                    text-align: center;
                    padding: 20px;
                    margin-top: 40px;
                    background-color: var(--primary-color);
                    color: white;
                    border-radius: 0 0 5px 5px;
                }}
                
                @media (max-width: 768px) {{
                    .container {{
                        padding: 10px;
                    }}
                    
                    header, .content {{
                        padding: 15px;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <header>
                    <h1>Análise de {analysis_results["topic"]}</h1>
                    <p>Gerado em: {analysis_results["timestamp"]}</p>
                </header>
                
                <div class="content">
                    <div class="summary-card">
                        <h2>Resumo da Análise</h2>
                        <p>{analysis_results["summary"]}</p>
                    </div>
                    
                    <div class="visualization-section">
                        <h2>Visualizações</h2>
                        {visualizations_html if visualizations_html else "<p>Nenhuma visualização foi gerada nesta análise.</p>"}
                    </div>
                    
                    <div class="queries-section">
                        <h2>Consultas e Análises</h2>
                        {queries_html}
                    </div>
                </div>
                
                <footer>
                    <p>Relatório gerado pelo OpenAI Analyzer</p>
                    <p>© {datetime.now().year} - Sistema de Análise em Linguagem Natural</p>
                </footer>
            </div>
        </body>
        </html>
        """
        
        # Salva o arquivo HTML
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_template)
        
        logger.info(f"Relatório HTML gerado em: {output_path}")
        return output_path
    
    def open_report(self, report_path: str) -> None:
        """
        Abre o relatório HTML no navegador padrão.
        
        Args:
            report_path: Caminho para o arquivo HTML
        """
        try:
            # Converte para URL de arquivo
            file_url = f"file://{os.path.abspath(report_path)}"
            webbrowser.open(file_url)
            logger.info(f"Relatório aberto no navegador: {file_url}")
        except Exception as e:
            logger.error(f"Erro ao abrir relatório: {str(e)}")
            print(f"Não foi possível abrir o relatório automaticamente. Acesse manualmente: {report_path}")


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
    
    # Opções de análise
    parser.add_argument("--topic", default="vendas perdidas", 
                        help="Tópico para análise automática (ex: 'vendas', 'clientes')")
    parser.add_argument("--questions", type=int, default=5,
                        help="Número de perguntas a serem geradas")
    parser.add_argument("--query", help="Executar uma consulta específica")
    parser.add_argument("--open-report", action="store_true", 
                        help="Abrir relatório no navegador após a análise")
    
    args = parser.parse_args()
    
    try:
        # Inicializa o analisador
        analyzer = OpenAIAnalyzer(
            api_key=args.api_key,
            model=args.model,
            data_dir=args.data_dir,
            output_dir=args.output_dir
        )
        
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
        print(f"\nErro: {str(e)}")
        print("\nVerifique o arquivo de log para mais detalhes.")
        sys.exit(1)


if __name__ == "__main__":
    main()