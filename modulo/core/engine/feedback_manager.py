"""
Módulo para gerenciamento de feedback do usuário e histórico de consultas.
"""

import os
import json
import time
import logging
from typing import Dict, List, Any, Optional

# Configura logger
logger = logging.getLogger("feedback_manager")

class FeedbackManager:
    """
    Gerencia o feedback do usuário e armazena consultas bem-sucedidas.
    Responsável por:
    - Armazenar feedback do usuário para melhorias futuras
    - Armazenar consultas bem-sucedidas para uso em cache e sugestões
    - Usar feedback para melhorar respostas em consultas subsequentes
    """
    
    def __init__(self, base_dir: Optional[str] = None):
        """
        Inicializa o gerenciador de feedback.
        
        Args:
            base_dir: Diretório base para armazenamento. Se não fornecido,
                     usa o diretório do arquivo atual.
        """
        # Define o diretório base para armazenamento
        if base_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            # Vai para o diretório pai do diretório engine
            base_dir = os.path.dirname(os.path.dirname(base_dir))
        
        self.base_dir = base_dir
        
        # Configura diretórios para armazenamento
        self.feedback_dir = os.path.join(self.base_dir, "user_feedback")
        self.cache_dir = os.path.join(self.base_dir, "query_cache")
        
        # Cria diretórios se não existirem
        os.makedirs(self.feedback_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Caminhos para arquivos de armazenamento
        self.feedback_file = os.path.join(self.feedback_dir, "user_feedback.json")
        self.cache_file = os.path.join(self.cache_dir, "successful_queries.json")
        
        logger.info(f"FeedbackManager inicializado. Diretório base: {self.base_dir}")
    
    def store_user_feedback(self, query: str, feedback: str) -> None:
        """
        Armazena feedback do usuário para melhorias futuras.
        
        Args:
            query: Consulta relacionada ao feedback
            feedback: Texto do feedback
        """
        try:
            # Carrega o feedback existente
            existing_feedback = []
            if os.path.exists(self.feedback_file):
                with open(self.feedback_file, 'r', encoding='utf-8') as f:
                    existing_feedback = json.load(f)
            
            # Adiciona o novo feedback
            existing_feedback.append({
                "timestamp": time.time(),
                "query": query,
                "feedback": feedback
            })
            
            # Salva o feedback atualizado
            with open(self.feedback_file, 'w', encoding='utf-8') as f:
                json.dump(existing_feedback, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Feedback armazenado para a consulta: '{query}'")
                
        except Exception as e:
            logger.error(f"Erro ao armazenar feedback do usuário: {str(e)}")
    
    def store_successful_query(self, query: str, code: str) -> None:
        """
        Armazena consultas bem-sucedidas para uso futuro em sugestões.
        
        Args:
            query: Consulta que foi bem-sucedida
            code: Código gerado para a consulta
        """
        try:
            # Carrega o cache existente
            existing_cache = {}
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    existing_cache = json.load(f)
            
            # Adiciona a nova consulta
            cleaned_query = query.strip().lower()
            existing_cache[cleaned_query] = {
                "timestamp": time.time(),
                "original_query": query,
                "code": code
            }
            
            # Salva o cache atualizado
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(existing_cache, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Consulta bem-sucedida armazenada: '{query}'")
                
        except Exception as e:
            logger.error(f"Erro ao armazenar consulta bem-sucedida: {str(e)}")
    
    def get_feedback_for_query(self, query: str) -> List[Dict[str, Any]]:
        """
        Recupera feedback histórico para consultas similares.
        
        Args:
            query: Consulta atual
            
        Returns:
            Lista de feedbacks relevantes
        """
        try:
            if not os.path.exists(self.feedback_file):
                return []
                
            # Carrega feedbacks existentes
            with open(self.feedback_file, 'r', encoding='utf-8') as f:
                feedbacks = json.load(f)
            
            # Implementação simples: retorna feedbacks para consultas similares
            # Em uma implementação real, usaríamos embeddings ou busca semântica
            query_terms = set(query.lower().split())
            relevant_feedbacks = []
            
            for fb in feedbacks:
                stored_query = fb["query"].lower()
                stored_terms = set(stored_query.split())
                
                # Verifica sobreposição de termos
                common_terms = query_terms.intersection(stored_terms)
                if len(common_terms) > 2 or any(term in stored_query for term in query_terms):
                    relevant_feedbacks.append(fb)
            
            # Ordena por timestamp (mais recentes primeiro)
            relevant_feedbacks.sort(key=lambda x: x["timestamp"], reverse=True)
            
            return relevant_feedbacks[:5]  # Retorna até 5 feedbacks relevantes
                
        except Exception as e:
            logger.error(f"Erro ao recuperar feedbacks: {str(e)}")
            return []
    
    def get_similar_successful_queries(self, query: str, max_results: int = 3) -> List[Dict[str, Any]]:
        """
        Recupera consultas bem-sucedidas similares à consulta atual.
        
        Args:
            query: Consulta atual
            max_results: Número máximo de resultados a retornar
            
        Returns:
            Lista de consultas similares bem-sucedidas
        """
        try:
            if not os.path.exists(self.cache_file):
                return []
                
            # Carrega cache de consultas
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                cached_queries = json.load(f)
            
            # Implementação simples de busca por similaridade
            # Em uma implementação real, usaríamos embeddings para melhor similaridade
            query_terms = set(query.lower().split())
            similar_queries = []
            
            for stored_query, details in cached_queries.items():
                stored_terms = set(stored_query.split())
                
                # Calcula similaridade baseada em termos comuns
                common_terms = query_terms.intersection(stored_terms)
                similarity = len(common_terms) / max(len(query_terms), len(stored_terms))
                
                if similarity > 0.3 or any(term in stored_query for term in query_terms):
                    similar_queries.append({
                        "query": details["original_query"],
                        "code": details["code"],
                        "timestamp": details["timestamp"],
                        "similarity": similarity
                    })
            
            # Ordena por similaridade
            similar_queries.sort(key=lambda x: x["similarity"], reverse=True)
            
            return similar_queries[:max_results]
                
        except Exception as e:
            logger.error(f"Erro ao recuperar consultas similares: {str(e)}")
            return []
    
    def process_query_with_feedback(self, query: str, feedback: Optional[str] = None) -> Dict[str, Any]:
        """
        Processa uma consulta incorporando feedback do usuário.
        
        Args:
            query: Consulta do usuário
            feedback: Feedback opcional para esta consulta
            
        Returns:
            Dicionário com consulta enriquecida e contexto adicional
        """
        result = {
            "query": query,
            "has_feedback": False,
            "similar_queries": [],
            "feedback_context": []
        }
        
        # Armazena feedback se fornecido
        if feedback:
            self.store_user_feedback(query, feedback)
            result["has_feedback"] = True
            result["feedback"] = feedback
        
        # Busca consultas similares do cache
        similar_queries = self.get_similar_successful_queries(query)
        if similar_queries:
            result["similar_queries"] = similar_queries
        
        # Busca feedbacks relevantes
        relevant_feedbacks = self.get_feedback_for_query(query)
        if relevant_feedbacks:
            result["feedback_context"] = relevant_feedbacks
        
        return result
    
    def cleanup_old_records(self, max_age_days: int = 30) -> None:
        """
        Remove registros antigos para economizar espaço.
        
        Args:
            max_age_days: Idade máxima em dias para manter registros
        """
        try:
            # Tempo atual em segundos
            current_time = time.time()
            # Tempo máximo em segundos (dias * 24h * 60min * 60s)
            max_age_seconds = max_age_days * 24 * 60 * 60
            
            # Limpa cache de consultas
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    cached_queries = json.load(f)
                
                # Filtra por idade
                filtered_queries = {}
                for query, details in cached_queries.items():
                    if current_time - details["timestamp"] < max_age_seconds:
                        filtered_queries[query] = details
                
                # Salva versão filtrada
                with open(self.cache_file, 'w', encoding='utf-8') as f:
                    json.dump(filtered_queries, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Limpeza do cache de consultas: {len(cached_queries) - len(filtered_queries)} registros removidos")
            
            # Limpa feedbacks antigos
            if os.path.exists(self.feedback_file):
                with open(self.feedback_file, 'r', encoding='utf-8') as f:
                    feedbacks = json.load(f)
                
                # Filtra por idade
                filtered_feedbacks = [
                    fb for fb in feedbacks 
                    if current_time - fb["timestamp"] < max_age_seconds
                ]
                
                # Salva versão filtrada
                with open(self.feedback_file, 'w', encoding='utf-8') as f:
                    json.dump(filtered_feedbacks, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Limpeza de feedbacks: {len(feedbacks) - len(filtered_feedbacks)} registros removidos")
                
        except Exception as e:
            logger.error(f"Erro durante limpeza de registros antigos: {str(e)}")