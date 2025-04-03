import re
import time
import logging
from typing import Dict, Any, Optional, List
from .llm_client import LLMMessage, LLMResponse

logger = logging.getLogger("GenBI.NLProcessor")

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
            Você é um assistente especializado em converter perguntas em linguagem natural para consultas SQL precisas.
            
            Diretrizes para geração de SQL:
            1. Use apenas as tabelas e colunas disponíveis no schema fornecido
            2. Gere SQL padrão e bem formatado
            3. Use JOINs quando necessário para relacionar tabelas
            4. Aplique agregações (SUM, COUNT, AVG) conforme apropriado
            5. Use GROUP BY para consultas agregadas
            6. Filtre e ordene resultados conforme necessário
            7. Limite resultados para grandes conjuntos de dados
            8. Seja conciso e direto
            9. Retorne apenas a consulta SQL sem comentários ou explicações
            """,
            
            "schema_template": """
            SCHEMA DO BANCO DE DADOS:
            {schema_description}
            
            INSTRUÇÕES:
            - Use apenas as tabelas e colunas descritas
            - Se a pergunta exigir dados não disponíveis, adapte ou explique a limitação
            """,
            
            "sql_generation": """
            Converta a seguinte pergunta em uma consulta SQL precisa usando o schema acima:
            
            PERGUNTA: {question}
            
            CONSULTA SQL:
            """
        }
    
    def _format_schema_description(self) -> str:
        """Formata a descrição do schema para incluir no prompt"""
        description = []
        
        # Verifica se catalog_info existe e tem o conteúdo esperado
        if not self.catalog_info:
            # Fallback para o schema padrão de amostra para SQLite
            description = [
                "Tabela: customers - Clientes da empresa",
                "Colunas:",
                "  - customer_id (number): ID único do cliente",
                "  - name (string): Nome completo do cliente",
                "  - email (string): Endereço de e-mail do cliente",
                "  - city (string): Cidade do cliente",
                "  - country (string): País do cliente",
                "  - created_at (datetime): Data de cadastro do cliente",
                "",
                "Tabela: products - Catálogo de produtos",
                "Colunas:",
                "  - product_id (number): ID único do produto",
                "  - name (string): Nome do produto",
                "  - category (string): Categoria do produto",
                "  - price (number): Preço do produto em reais",
                "  - cost (number): Custo do produto em reais",
                "",
                "Tabela: orders - Pedidos realizados pelos clientes",
                "Colunas:",
                "  - order_id (number): ID único do pedido",
                "  - customer_id (number): ID do cliente que fez o pedido",
                "  - order_date (date): Data em que o pedido foi realizado",
                "  - status (string): Status atual do pedido",
                "",
                "Tabela: order_items - Itens incluídos em cada pedido",
                "Colunas:",
                "  - order_item_id (number): ID único do item do pedido",
                "  - order_id (number): ID do pedido ao qual o item pertence",
                "  - product_id (number): ID do produto",
                "  - quantity (number): Quantidade do produto",
                "  - price (number): Preço unitário do produto no momento da compra",
                "",
                "Relacionamentos:",
                "  - order_to_customer (MANY_TO_ONE): orders -> customers ON orders.customer_id = customers.customer_id",
                "  - item_to_order (MANY_TO_ONE): order_items -> orders ON order_items.order_id = orders.order_id",
                "  - item_to_product (MANY_TO_ONE): order_items -> products ON order_items.product_id = products.product_id",
                ""
            ]
            return "\n".join(description)
        
        # Formatar modelos (tabelas)
        if 'models' in self.catalog_info and isinstance(self.catalog_info['models'], list):
            for model in self.catalog_info['models']:
                if not isinstance(model, dict):
                    continue
                    
                table_desc = f"Tabela: {model.get('name', 'unknown')}"
                if 'description' in model and model['description']:
                    table_desc += f" - {model['description']}"
                description.append(table_desc)
                
                # Listar colunas
                description.append("Colunas:")
                if 'columns' in model and isinstance(model['columns'], list):
                    for col in model['columns']:
                        if not isinstance(col, dict):
                            continue
                        col_desc = f"  - {col.get('name', 'unknown')} ({col.get('type', 'unknown')})"
                        if 'description' in col and col['description']:
                            col_desc += f": {col['description']}"
                        description.append(col_desc)
                
                description.append("")  # Linha em branco
        
        # Formatar relacionamentos
        if 'relationships' in self.catalog_info and isinstance(self.catalog_info['relationships'], list):
            description.append("Relacionamentos:")
            for rel in self.catalog_info['relationships']:
                if not isinstance(rel, dict):
                    continue
                    
                rel_name = rel.get('name', 'unknown')
                rel_type = rel.get('type', 'unknown')
                rel_models = rel.get('models', [])
                
                if isinstance(rel_models, list) and len(rel_models) > 1:
                    rel_desc = f"  - {rel_name} ({rel_type}): {' -> '.join(rel_models)}"
                    if 'condition' in rel:
                        rel_desc += f" ON {rel['condition']}"
                    description.append(rel_desc)
            
            description.append("")  # Linha em branco
        
        # Se não há descrição, retorne um schema padrão de fallback
        if not description:
            return self._get_fallback_schema()
            
        return "\n".join(description)
        
    def _get_fallback_schema(self) -> str:
        """Retorna schema padrão de amostra para SQLite"""
        return """
        Tabela: customers - Clientes da empresa
        Colunas:
          - customer_id (number): ID único do cliente
          - name (string): Nome completo do cliente
          - email (string): Endereço de e-mail do cliente
          - city (string): Cidade do cliente
          - country (string): País do cliente
          - created_at (datetime): Data de cadastro do cliente

        Tabela: products - Catálogo de produtos
        Colunas:
          - product_id (number): ID único do produto
          - name (string): Nome do produto
          - category (string): Categoria do produto
          - price (number): Preço do produto em reais
          - cost (number): Custo do produto em reais

        Tabela: orders - Pedidos realizados pelos clientes
        Colunas:
          - order_id (number): ID único do pedido
          - customer_id (number): ID do cliente que fez o pedido
          - order_date (date): Data em que o pedido foi realizado
          - status (string): Status atual do pedido

        Tabela: order_items - Itens incluídos em cada pedido
        Colunas:
          - order_item_id (number): ID único do item do pedido
          - order_id (number): ID do pedido ao qual o item pertence
          - product_id (number): ID do produto
          - quantity (number): Quantidade do produto
          - price (number): Preço unitário do produto no momento da compra

        Relacionamentos:
          - order_to_customer (MANY_TO_ONE): orders -> customers ON orders.customer_id = customers.customer_id
          - item_to_order (MANY_TO_ONE): order_items -> orders ON order_items.order_id = orders.order_id
          - item_to_product (MANY_TO_ONE): order_items -> products ON order_items.product_id = products.product_id
        """
    
    def generate_sql(self, question: str) -> str:
        """
        Gera consulta SQL a partir de uma pergunta em linguagem natural
        
        Args:
            question: Pergunta em linguagem natural
            
        Returns:
            str: Consulta SQL gerada
        """
        try:
            # Validar entrada
            if not question or not isinstance(question, str):
                raise ValueError("A pergunta deve ser uma string não vazia")
            
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
            
            # Tentar gerar SQL com retry em caso de erro
            max_retries = 3
            backoff_factor = 1.5  # Tempo de espera aumenta exponencialmente
            last_error = None
            
            for attempt in range(max_retries):
                try:
                    # Chamar LLM para gerar SQL com temperatura reduzida para estabilidade
                    response = self.llm_client.complete(messages, temperature=0.0)
                    
                    # Extrair SQL da resposta
                    if not response or not response.content:
                        logger.warning(f"Tentativa {attempt+1}: Nenhuma resposta válida gerada pelo LLM")
                        continue
                    
                    sql = response.content.strip()
                    
                    # Limpar SQL (remover comentários, aspas extras, etc.)
                    sql = self._clean_sql(sql)
                    
                    # Validar SQL básica
                    if not sql.upper().startswith(('SELECT', 'WITH')):
                        logger.warning(f"Tentativa {attempt+1}: SQL gerado não parece ser uma consulta válida: {sql}")
                        continue
                    
                    # Verificar expressões maliciosas ou arriscadas
                    unsafe_patterns = [
                        "DROP ", "DELETE ", "TRUNCATE ", "UPDATE ", "INSERT ", 
                        "ALTER ", "CREATE ", "GRANT ", "REVOKE ", "PRAGMA ", 
                        "ATTACH ", "DETACH ", "sys.", "xp_", "sp_", "exec("
                    ]
                    
                    if any(pattern.upper() in sql.upper() for pattern in unsafe_patterns):
                        logger.warning(f"Tentativa {attempt+1}: SQL gerado contém comandos potencialmente perigosos")
                        continue
                    
                    logger.info(f"SQL gerado na tentativa {attempt+1}: {sql}")
                    return sql
                    
                except Exception as e:
                    last_error = e
                    logger.warning(f"Erro na tentativa {attempt+1} de gerar SQL: {str(e)}")
                    # Esperar antes de tentar novamente
                    if attempt < max_retries - 1:  # Não espera após a última tentativa
                        import time
                        wait_time = backoff_factor ** attempt
                        logger.info(f"Aguardando {wait_time:.2f}s antes da próxima tentativa...")
                        time.sleep(wait_time)
            
            # Se chegou aqui, todas as tentativas falharam
            if last_error:
                logger.error(f"Todas as tentativas de gerar SQL falharam. Último erro: {str(last_error)}")
            else:
                logger.error("Todas as tentativas de gerar SQL falharam com respostas inválidas")
            
            # Deduzir uma consulta padrão baseada em palavras-chave na pergunta
            question_lower = question.lower()
            
            if any(term in question_lower for term in ['venda', 'vendas', 'receita', 'faturamento']):
                # Consulta para vendas/receita
                time_pattern = None
                if any(term in question_lower for term in ['mês', 'mensal', 'mensais']):
                    time_pattern = "strftime('%Y-%m', orders.order_date)"
                elif any(term in question_lower for term in ['ano', 'anual', 'anuais']):
                    time_pattern = "strftime('%Y', orders.order_date)"
                elif any(term in question_lower for term in ['dia', 'diário', 'diária']):
                    time_pattern = "orders.order_date"
                
                if time_pattern:
                    return f"""
                    SELECT 
                        {time_pattern} AS periodo, 
                        SUM(oi.quantity * oi.price) AS receita_total
                    FROM order_items oi
                    JOIN orders ON oi.order_id = orders.order_id
                    GROUP BY periodo
                    ORDER BY periodo
                    """
                elif any(term in question_lower for term in ['categoria', 'categorias']):
                    return """
                    SELECT 
                        p.category AS categoria, 
                        SUM(oi.quantity * oi.price) AS receita_total
                    FROM order_items oi
                    JOIN products p ON oi.product_id = p.product_id
                    GROUP BY p.category
                    ORDER BY receita_total DESC
                    """
                elif any(term in question_lower for term in ['cliente', 'clientes']):
                    return """
                    SELECT 
                        c.name AS cliente, 
                        SUM(oi.quantity * oi.price) AS receita_total
                    FROM order_items oi
                    JOIN orders o ON oi.order_id = o.order_id
                    JOIN customers c ON o.customer_id = c.customer_id
                    GROUP BY c.customer_id
                    ORDER BY receita_total DESC
                    """
                else:
                    return """
                    SELECT 
                        SUM(oi.quantity * oi.price) AS receita_total
                    FROM order_items oi
                    """
            
            elif any(term in question_lower for term in ['produto', 'produtos', 'vendido', 'vendidos']):
                limit = 10
                if 'top' in question_lower or 'mais' in question_lower:
                    # Tentar extrair número após "top" ou "mais"
                    import re
                    matches = re.findall(r'\b(?:top|mais)\s+(\d+)\b', question_lower)
                    if matches:
                        try:
                            limit = int(matches[0])
                        except ValueError:
                            pass
                
                return f"""
                SELECT 
                    p.name AS produto, 
                    SUM(oi.quantity) AS quantidade_vendida,
                    SUM(oi.quantity * oi.price) AS receita_total
                FROM order_items oi
                JOIN products p ON oi.product_id = p.product_id
                GROUP BY p.name
                ORDER BY quantidade_vendida DESC
                LIMIT {limit}
                """
            
            elif any(term in question_lower for term in ['cliente', 'clientes']):
                return """
                SELECT 
                    c.name AS cliente,
                    c.city AS cidade,
                    c.country AS pais,
                    COUNT(o.order_id) AS total_pedidos
                FROM customers c
                LEFT JOIN orders o ON c.customer_id = o.customer_id
                GROUP BY c.customer_id
                ORDER BY total_pedidos DESC
                """
            
            elif any(term in question_lower for term in ['pedido', 'pedidos', 'order', 'orders']):
                return """
                SELECT 
                    o.order_id AS pedido,
                    c.name AS cliente,
                    o.order_date AS data,
                    o.status AS status,
                    SUM(oi.quantity * oi.price) AS valor_total
                FROM orders o
                JOIN customers c ON o.customer_id = c.customer_id
                JOIN order_items oi ON o.order_id = oi.order_id
                GROUP BY o.order_id
                ORDER BY o.order_date DESC
                """
            
            # Consulta padrão se nenhum padrão foi identificado
            return """
            SELECT 
                'Por favor, especifique o que deseja consultar sobre vendas, produtos, clientes ou pedidos' AS mensagem
            """
            
        except Exception as e:
            logger.error(f"Erro ao gerar SQL: {str(e)}")
            # Retornar consulta padrão indicando erro
            return """
            SELECT 
                'Não foi possível gerar a consulta' AS error_message
            """
    
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
        
        try:
            # Chamar LLM para gerar explicação
            response = self.llm_client.complete(messages, temperature=0.1)
            return response.content.strip()
        except Exception as e:
            logger.error(f"Erro ao gerar explicação: {str(e)}")
            return "Não foi possível gerar uma explicação para a consulta."