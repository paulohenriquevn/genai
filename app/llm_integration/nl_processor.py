import re
import logging
from typing import Dict, Any, Optional
from enum import Enum

logger = logging.getLogger("GenBI.NLProcessor")

class LLMProvider(Enum):
    """Provedores de LLM suportados"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    CUSTOM = "custom"

class LLMMessage:
    """Mensagem para o LLM"""
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content
        
    def to_dict(self):
        return {"role": self.role, "content": self.content}

class LLMResponse:
    """Resposta do LLM"""
    def __init__(self, content: str, model: str, usage: Dict[str, int], created_at: float, finish_reason: Optional[str] = None):
        self.content = content
        self.model = model
        self.usage = usage
        self.created_at = created_at
        self.finish_reason = finish_reason

class NLtoSQLProcessor:
    """Processador que converte perguntas em linguagem natural para SQL"""
    
    def __init__(self, llm_client, catalog_info: Dict[str, Any], prompt_templates: Optional[Dict[str, str]] = None):
        """
        Inicializa o processador NL-to-SQL
        
        Args:
            llm_client: Cliente LLM a usar
            catalog_info: Informações sobre o catálogo de dados (modelos, relações, etc.)
            prompt_templates: Templates de prompts para o LLM (opcional)
        """
        self.llm_client = llm_client
        self.catalog_info = catalog_info
        
        # Templates padrão se não fornecidos
        self.prompt_templates = prompt_templates or {
            "system_context": """
            Você é um assistente especializado em converter perguntas em linguagem natural para consultas SQL.
            Você deve analisar a pergunta do usuário e criar uma consulta SQL válida considerando o schema do banco de dados fornecido.
            
            Regras:
            1. Gere apenas SQL padrão e bem formatado
            2. Se o SQL envolver várias tabelas, use JOINs explícitos
            3. Inclua aliases para facilitar a leitura
            4. Use colunas e tabelas exatamente como descritas no schema
            5. Substitua comparações de texto por LIKE ou ILIKE quando apropriado
            6. Se a pergunta exigir agregações, use GROUP BY adequadamente
            7. Use HAVING para filtros em colunas agregadas
            8. Use LIMIT para limitar resultados quando apropriado
            9. Dê preferência ao formato mais simples da consulta que atenda à pergunta
            10. Retorne apenas a consulta SQL sem comentários
            """,
            
            "schema_template": """
            SCHEMA DO BANCO DE DADOS:
            {schema_description}
            """,
            
            "sql_generation": """
            Converta a seguinte pergunta em uma consulta SQL utilizando o schema acima:
            
            PERGUNTA: {question}
            
            SQL:
            """
        }
    
    def _format_schema_description(self) -> str:
        """Formata a descrição do schema para incluir no prompt"""
        description = []
        
        # Formatar modelos (tabelas)
        if 'models' in self.catalog_info:
            for model in self.catalog_info['models']:
                table_desc = f"Tabela: {model['name']}"
                if 'description' in model and model['description']:
                    table_desc += f" - {model['description']}"
                description.append(table_desc)
                
                # Listar colunas
                description.append("Colunas:")
                for col in model['columns']:
                    col_desc = f"  - {col['name']} ({col['type']})"
                    if 'description' in col and col['description']:
                        col_desc += f": {col['description']}"
                    description.append(col_desc)
                
                # Adicionar chave primária se existir
                if 'primaryKey' in model and model['primaryKey']:
                    pk = model['primaryKey']
                    pk_str = pk if isinstance(pk, str) else ', '.join(pk)
                    description.append(f"  Chave primária: {pk_str}")
                
                description.append("")  # Linha em branco
        
        # Formatar relacionamentos
        if 'relationships' in self.catalog_info:
            description.append("Relacionamentos:")
            for rel in self.catalog_info['relationships']:
                rel_desc = f"  - {rel['name']} ({rel['type']}): {' -> '.join(rel['models'])}"
                if 'condition' in rel:
                    rel_desc += f" ON {rel['condition']}"
                description.append(rel_desc)
            
            description.append("")  # Linha em branco
            
        return "\n".join(description)
    
    def generate_sql(self, question: str) -> str:
        """
        Gera consulta SQL a partir de uma pergunta em linguagem natural
        
        Args:
            question: Pergunta em linguagem natural
            
        Returns:
            str: Consulta SQL gerada
        """
        # Preparar descrição do schema
        schema_description = self._format_schema_description()
        
        # Construir prompt completo
        system_prompt = self.prompt_templates["system_context"]
        schema_prompt = self.prompt_templates["schema_template"].format(
            schema_description=schema_description
        )
        
        question_prompt = self.prompt_templates["sql_generation"].format(
            question=question
        )
        
        # Criar mensagens para o LLM
        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=f"{schema_prompt}\n\n{question_prompt}")
        ]
        
        # Chamar LLM para gerar SQL
        response = self.llm_client.complete(messages, temperature=0.0)
        
        # Extrair SQL da resposta
        sql = response.content.strip()
        
        # Limpar SQL (remover comentários, aspas extras, etc.)
        sql = self._clean_sql(sql)
        
        logger.info(f"SQL gerado: {sql}")
        return sql
    
    def _clean_sql(self, sql: str) -> str:
        """Limpa a consulta SQL (remove blocos de código, comentários, etc.)"""
        # Remover blocos de código markdown, se presentes
        sql = re.sub(r'```sql\s*', '', sql)
        sql = re.sub(r'```\s*', '', sql)
        
        # Remover comentários SQL
        sql = re.sub(r'--.*$', '', sql, flags=re.MULTILINE)
        
        # Remover espaços em branco extras
        sql = sql.strip()
        
        return sql
    
    def explain_sql(self, sql: str, question: str) -> str:
        """
        Gera uma explicação em linguagem natural de uma consulta SQL
        
        Args:
            sql: Consulta SQL a explicar
            question: Pergunta original em linguagem natural
            
        Returns:
            str: Explicação da consulta
        """
        # Criar prompt para explicação
        prompt = f"""
        Por favor, explique a seguinte consulta SQL em linguagem natural simples:
        
        Pergunta original: {question}
        
        Consulta SQL:
        ```sql
        {sql}
        ```
        
        Explique em detalhes o que essa consulta faz, incluindo:
        1. Quais tabelas são usadas
        2. Como os dados são filtrados
        3. Quais transformações são aplicadas
        4. O que os resultados representam
        """
        
        # Criar mensagens para o LLM
        messages = [
            LLMMessage(role="system", content="Você é um especialista em explicar consultas SQL de forma clara e compreensível para não especialistas."),
            LLMMessage(role="user", content=prompt)
        ]
        
        # Chamar LLM para gerar explicação
        response = self.llm_client.complete(messages, temperature=0.1)
        
        return response.content.strip()