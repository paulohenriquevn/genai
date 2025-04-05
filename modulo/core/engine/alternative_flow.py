"""
Sistema de fluxo alternativo quando ocorrem falhas na LLM.
"""

import re
import os
import logging
from typing import List, Dict, Any, Optional, Set, Union

from core.engine.dataset import Dataset
from core.response.string import StringResponse

# Configura o logger
logger = logging.getLogger("core_integration")


class AlternativeFlow:
    """
    Implementa o fluxo alternativo para falhas na geração ou execução de consultas.
    
    Fornece:
    - Detecção de entidades inexistentes
    - Reformulação automática de consultas
    - Geração de consultas alternativas
    - Tratamento amigável de erros
    """
    
    def __init__(self, datasets: Dict[str, Dataset], llm_generator=None):
        """
        Inicializa o fluxo alternativo.
        
        Args:
            datasets: Dicionário de datasets disponíveis (nome -> objeto Dataset)
            llm_generator: Gerador de código LLM para reformulações
        """
        self.datasets = datasets
        self.llm_generator = llm_generator
        self._setup_entity_keywords()
    
    def _setup_entity_keywords(self):
        """Configura as palavras-chave para detecção de entidades."""
        # Lista de palavras-chave que indicam consultas sobre entidades não existentes
        self.missing_entity_keywords = {
            'produtos': ['produtos', 'produto', 'estoque', 'inventário', 'item', 'itens', 'mercadoria'],
            'funcionários': ['funcionários', 'funcionário', 'funcionario', 'funcionarios', 'colaborador', 'colaboradores', 'empregado', 'empregados', 'staff', 'equipe'],
            'departamentos': ['departamento', 'departamentos', 'setor', 'setores', 'área', 'áreas', 'divisão', 'divisões'],
            'categorias': ['categoria', 'categorias', 'classe', 'classes', 'tipo de produto', 'tipos de produto']
        }
    
    def check_missing_entities(self, query: str) -> Optional[StringResponse]:
        """
        Verifica se a consulta menciona entidades que não existem.
        
        Args:
            query: Consulta em linguagem natural
            
        Returns:
            StringResponse se detectar entidades inexistentes, None caso contrário
        """
        # Verifica se a consulta menciona entidades não existentes
        for entity_type, keywords in self.missing_entity_keywords.items():
            entity_exists = any(entity_type in ds.name.lower() for ds in self.datasets.values())
            
            # Se a entidade não existe mas é mencionada na consulta
            if not entity_exists and any(keyword in query.lower() for keyword in keywords):
                # Gera sugestões de consultas alternativas baseadas nos dados disponíveis
                alternative_queries = self.generate_alternative_queries()
                datasets_desc = ", ".join([f"{name}" for name, _ in self.datasets.items()])
                
                return self.create_missing_entity_response(
                    entity_type, 
                    datasets_desc, 
                    alternative_queries
                )
                
        return None
    
    def create_missing_entity_response(self, entity_type: str, datasets_desc: str, alternative_queries: List[str]) -> StringResponse:
        """
        Cria uma resposta amigável para consultas sobre entidades não existentes.
        
        Args:
            entity_type: Tipo de entidade não encontrada
            datasets_desc: Descrição dos datasets disponíveis
            alternative_queries: Consultas alternativas sugeridas
            
        Returns:
            StringResponse com mensagem e sugestões
        """
        message = f"Não há dados sobre {entity_type} disponíveis. Os datasets disponíveis são: {datasets_desc}."
        
        if alternative_queries:
            message += "\n\nVocê pode tentar estas consultas alternativas:\n"
            for i, query in enumerate(alternative_queries[:3], 1):
                message += f"{i}. {query}\n"
            
        message += "\nPor favor, reformule sua consulta para usar os dados disponíveis."
        
        return StringResponse(message)
        
    def handle_missing_table_error(self, error_msg: str) -> StringResponse:
        """
        Gera uma resposta amigável para erros de tabela não encontrada.
        
        Args:
            error_msg: Mensagem de erro original
            
        Returns:
            StringResponse com informações úteis
        """
        # Extrai o nome da tabela da mensagem de erro, se possível
        import re
        table_match = re.search(r"tabela '(\w+)'", error_msg)
        missing_table = table_match.group(1) if table_match else "mencionada"
        
        # Lista de datasets disponíveis com suas colunas
        datasets_info = []
        for name, ds in self.datasets.items():
            cols = ", ".join(ds.dataframe.columns[:5]) + ("..." if len(ds.dataframe.columns) > 5 else "")
            datasets_info.append(f"• {name}: {cols}")
        
        datasets_desc = "\n".join(datasets_info)
        
        message = f"""Não foi possível encontrar a tabela '{missing_table}' nos dados disponíveis.

Os datasets disponíveis são:
{datasets_desc}

Por favor, reformule sua consulta para usar apenas os datasets listados acima."""
        
        return StringResponse(message)
    
    def rephrase_query(self, original_query: str, error_info: str) -> str:
        """
        Usa o LLM para reformular a consulta original baseado no erro encontrado.
        
        Args:
            original_query: Consulta original que falhou
            error_info: Informação sobre o erro
            
        Returns:
            Consulta reformulada
        """
        if not self.llm_generator:
            # Se não temos gerador LLM, usa simplificação básica
            return self.simplify_query(original_query)
            
        # Lista de datasets disponíveis
        available_datasets = ', '.join(self.datasets.keys())
        
        # Cria um prompt para o LLM reformular a consulta
        rephrase_prompt = f'''Por favor, reformule a seguinte consulta para que ela funcione com os datasets disponíveis.

CONSULTA ORIGINAL: "{original_query}"

ERRO ENCONTRADO: {error_info}

DATASETS DISPONÍVEIS: {available_datasets}

COLUNAS DISPONÍVEIS:
'''
        
        # Adiciona informações sobre as colunas disponíveis
        for name, dataset in self.datasets.items():
            rephrase_prompt += f"\n{name}: {', '.join(dataset.dataframe.columns)}"
        
        rephrase_prompt += '''

Sua tarefa é reformular a consulta original para que ela:
1. Use apenas os datasets e colunas listados acima
2. Mantenha a intenção original da consulta
3. Evite os mesmos erros
4. Seja clara e direta

Por favor, forneça APENAS a consulta reformulada, sem explicações adicionais.'''

        try:
            # Tenta reformular a consulta usando o LLM
            rephrased_query = self.llm_generator.generate_code(rephrase_prompt)
            
            # Limpa a resposta, pegando apenas a primeira linha não vazia
            cleaned_query = re.sub(r'^[\s\'"]*|[\s\'"]*$', '', rephrased_query.split('\n')[0])
            
            # Se a limpeza resultar em string vazia, use uma linha subsequente
            if not cleaned_query:
                for line in rephrased_query.split('\n'):
                    line = re.sub(r'^[\s\'"]*|[\s\'"]*$', '', line)
                    if line:
                        cleaned_query = line
                        break
            
            # Garante que a consulta reformulada não seja o código Python gerado
            # (às vezes o LLM pode ignorar as instruções)
            if "import" in cleaned_query or "def " in cleaned_query or "result =" in cleaned_query:
                # Fallback para uma simplificação da consulta original
                return self.simplify_query(original_query)
            
            return cleaned_query if cleaned_query else original_query
            
        except Exception as e:
            logger.error(f"Erro ao reformular consulta: {str(e)}")
            # Em caso de erro, tenta uma simplificação básica
            return self.simplify_query(original_query)
            
    def simplify_query(self, query: str) -> str:
        """
        Simplifica a consulta original para torná-la mais genérica.
        
        Args:
            query: Consulta original
            
        Returns:
            Consulta simplificada
        """
        # Substitui termos específicos por termos mais genéricos
        simplifications = [
            (r'produto[s]?', 'dados'),
            (r'funcionario[s]?|colaborador[es]?|empregado[s]?', 'pessoas'),
            (r'departamento[s]?|setor[es]?', 'grupos'),
            (r'categorias?', 'tipos'),
            (r'estoque', 'quantidade'),
            (r'inventário', 'dados'),
        ]
        
        simplified = query
        for pattern, replacement in simplifications:
            simplified = re.sub(pattern, replacement, simplified, flags=re.IGNORECASE)
        
        # Se a consulta foi modificada, retorna a versão simplificada
        if simplified != query:
            return simplified
        
        # Se não conseguiu simplificar, retorna uma consulta ainda mais genérica
        keywords = ['mostre', 'liste', 'exiba', 'apresente', 'qual', 'quais', 'como', 'onde', 'quando']
        for keyword in keywords:
            if keyword in query.lower():
                # Extrai apenas a parte após a palavra-chave
                parts = re.split(rf'{keyword}\s+', query.lower(), flags=re.IGNORECASE, maxsplit=1)
                if len(parts) > 1:
                    return f"{keyword} os dados disponíveis sobre {parts[1]}"
        
        # Último recurso: consulta completamente genérica
        return "Mostre um resumo dos dados disponíveis"
    
    def generate_alternative_queries(self) -> List[str]:
        """
        Gera consultas alternativas baseadas nos datasets disponíveis.
        
        Returns:
            Lista de consultas sugeridas
        """
        alternatives = []
        
        # Consultas básicas para cada dataset
        for name in self.datasets.keys():
            alternatives.append(f"Mostre um resumo do dataset {name}")
            alternatives.append(f"Quais são as principais informações em {name}?")
        
        # Consultas mais específicas baseadas nos metadados dos datasets
        for name, dataset in self.datasets.items():
            # Consultas baseadas em tipos de colunas
            if hasattr(dataset, 'column_types'):
                # Identifica colunas numéricas para agregações
                numeric_cols = [col for col, type in dataset.column_types.items() 
                                if type in ['numeric', 'number', 'int', 'float']]
                
                # Identifica colunas categóricas para agrupamentos
                cat_cols = [col for col, type in dataset.column_types.items() 
                            if type in ['categorical', 'string', 'object']]
                
                # Identifica colunas de data para análises temporais
                date_cols = [col for col, type in dataset.column_types.items() 
                            if type in ['date', 'datetime']]
                
                # Gera consultas para agregações
                if numeric_cols and cat_cols:
                    alternatives.append(f"Qual é o total de {numeric_cols[0]} por {cat_cols[0]} em {name}?")
                
                # Gera consultas para ordenações
                if numeric_cols:
                    alternatives.append(f"Quais são os maiores valores de {numeric_cols[0]} em {name}?")
                
                # Gera consultas para análises temporais
                if date_cols:
                    alternatives.append(f"Como os dados de {name} variam ao longo do tempo?")
                    alternatives.append(f"Mostre os dados de {name} agrupados por mês")
            
            # Consultas baseadas em relacionamentos
            if hasattr(dataset, 'analyzed_metadata') and dataset.analyzed_metadata:
                if 'relationships' in dataset.analyzed_metadata:
                    rel_info = dataset.analyzed_metadata.get('relationships', {})
                    
                    if 'outgoing' in rel_info and rel_info['outgoing']:
                        for rel in rel_info['outgoing'][:2]:  # Limita a 2 relacionamentos
                            target = rel.get('target_dataset')
                            alternatives.append(f"Mostre dados de {name} relacionados com {target}")
        
        # Remove duplicatas e limita a 10 alternativas
        unique_alternatives = list(set(alternatives))
        return unique_alternatives[:10]
    
    def offer_predefined_options(self, query: str, error_msg: str) -> StringResponse:
        """
        Oferece opções predefinidas de consultas quando todas as tentativas falharam.
        
        Args:
            query: Consulta original
            error_msg: Mensagem de erro
            
        Returns:
            StringResponse com opções predefinidas
        """
        # Gera alternativas
        alternatives = self.generate_alternative_queries()
        
        # Prepara a mensagem de resposta
        message = f"""Não foi possível processar a consulta: "{query}"

Erro: {error_msg}

Aqui estão algumas consultas alternativas que você pode tentar:

"""
        # Adiciona as alternativas numeradas
        for i, alt in enumerate(alternatives[:5], 1):
            message += f"{i}. {alt}\n"
            
        message += """
Você também pode:
• Simplificar sua consulta
• Especificar exatamente quais datasets quer consultar
• Fornecer feedback para melhorarmos o sistema
"""
        
        return StringResponse(message)