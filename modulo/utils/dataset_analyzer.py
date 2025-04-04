import pandas as pd
import json
import re
from typing import List, Dict, Any, Union

class DatasetAnalyzer:
    """
    Módulo para analisar datasets, identificar campos, tipos de dados e gerar aliases.
    """
    
    # Mapeamento de aliases comuns para diferentes tipos de campos
    ALIAS_MAPPINGS = {
        # Campos de cliente
        "cliente": ["client", "customer", "user", "usuario", "pessoa"],
        "nome_cliente": ["customer_name", "client_name", "nome", "name", "nome_completo", "full_name"],
        "id_cliente": ["customer_id", "client_id", "user_id", "codigo_cliente", "cliente_codigo"],
        "email": ["email_address", "endereco_email", "e_mail", "correio_eletronico"],
        "telefone": ["phone", "celular", "contato", "mobile", "tel", "fone"],
        
        # Campos de produto
        "produto": ["product", "item", "mercadoria"],
        "id_produto": ["product_id", "product_code", "item_id", "codigo_produto", "sku"],
        "nome_produto": ["product_name", "item_name", "nome_item", "descricao_produto", "product_description"],
        "preco": ["price", "valor", "custo", "preco_unitario", "unit_price", "amount"],
        
        # Campos de transação
        "pedido": ["order", "compra", "transaction", "transacao"],
        "id_pedido": ["order_id", "order_number", "pedido_numero", "transaction_id"],
        "data_pedido": ["order_date", "date", "data", "data_compra", "purchase_date"],
        "status": ["status_pedido", "order_status", "situacao", "state"],
        
        # Campos de feedback
        "feedback": ["comentario", "observacao", "avaliacao", "review", "opinion", "opiniao", "nota"],
        "classificacao": ["rating", "estrelas", "stars", "avaliacao", "nota"],
        
        # Campos de localização
        "endereco": ["address", "local", "location", "localizacao"],
        "cidade": ["city", "municipio", "town"],
        "estado": ["state", "uf", "provincia", "province"],
        "pais": ["country", "nacao", "nation"],
        "cep": ["zip", "zip_code", "codigo_postal", "postal_code"],
        
        # Campos gerais
        "data": ["date", "data_registro", "data_criacao", "created_at", "creation_date"],
        "quantidade": ["quantity", "qtd", "qtde", "quant", "amount", "count"],
        "valor": ["value", "amount", "total", "price", "preco"],
        "descricao": ["description", "desc", "detalhes", "details", "info"]
    }
    
    # Mapeamento de tipos de dados
    TYPE_MAPPINGS = {
        "int64": "integer",
        "float64": "number",
        "object": "string",
        "bool": "boolean",
        "datetime64": "date",
        "datetime64[ns]": "date",
        "category": "string",
        "timedelta64[ns]": "duration"
    }
    
    def __init__(self):
        pass
    
    def read_dataset(self, file_path: str) -> pd.DataFrame:
        """
        Lê o dataset a partir do caminho do arquivo.
        Suporta formatos CSV, Excel, JSON, etc.
        
        Args:
            file_path: Caminho para o arquivo de dataset
            
        Returns:
            DataFrame com os dados carregados
        """
        try:
            if file_path.endswith('.csv'):
                return pd.read_csv(file_path)
            elif file_path.endswith(('.xls', '.xlsx')):
                return pd.read_excel(file_path)
            elif file_path.endswith('.json'):
                return pd.read_json(file_path)
            elif file_path.endswith('.parquet'):
                return pd.read_parquet(file_path)
            else:
                raise ValueError(f"Formato de arquivo não suportado: {file_path}")
        except Exception as e:
            raise Exception(f"Erro ao ler o dataset: {str(e)}")
    
    def generate_aliases(self, field_name: str) -> List[str]:
        """
        Gera aliases para um campo com base no nome.
        
        Args:
            field_name: Nome do campo
            
        Returns:
            Lista de aliases possíveis
        """
        # Normaliza o nome do campo (remove underscores, converte para minúsculas)
        normalized_name = field_name.lower().replace("_", "")
        
        # Verifica se o campo está no mapeamento de aliases predefinidos
        for key, aliases in self.ALIAS_MAPPINGS.items():
            # Checa se o nome do campo corresponde à chave ou a um dos aliases
            if normalized_name == key.lower().replace("_", "") or any(
                normalized_name == alias.lower().replace("_", "") for alias in aliases
            ):
                # Retorna os aliases menos o nome do próprio campo (se estiver na lista)
                result_aliases = [a for a in aliases if a.lower() != field_name.lower()]
                if key.lower() != field_name.lower():
                    result_aliases.append(key)
                return result_aliases[:3]  # Limita a 3 aliases
        
        # Se não encontrou no mapeamento, gera aliases baseados em padrões comuns
        result_aliases = []
        
        # Remove prefixos comuns (por exemplo, "ds_", "vl_", "nr_", etc.)
        clean_name = re.sub(r'^(ds_|vl_|nr_|id_|cd_|qt_)', '', field_name)
        if clean_name != field_name:
            result_aliases.append(clean_name)
        
        # Se o campo tiver underscores, cria variantes
        if "_" in field_name:
            parts = field_name.split("_")
            # Versão camelCase
            camel_case = parts[0] + "".join(p.capitalize() for p in parts[1:])
            if camel_case != field_name:
                result_aliases.append(camel_case)
            
            # Versão sem underscore
            no_underscore = "".join(parts)
            if no_underscore != field_name and no_underscore not in result_aliases:
                result_aliases.append(no_underscore)
        
        # Se o nome começa com uma letra e termina com um número, gera um alias sem o número
        match = re.match(r'([a-zA-Z_]+)(\d+)$', field_name)
        if match and match.group(1) not in result_aliases:
            result_aliases.append(match.group(1))
        
        # Limita a 3 aliases
        return result_aliases[:3] if result_aliases else ["campo" + str(hash(field_name) % 1000)]
    
    def infer_field_description(self, field_name: str) -> str:
        """
        Infere uma descrição para o campo com base no nome.
        
        Args:
            field_name: Nome do campo
            
        Returns:
            Descrição inferida
        """
        # Dicionário de descrições comuns
        descriptions = {
            "id": "Identificador único",
            "nome": "Nome completo",
            "email": "Endereço de e-mail",
            "telefone": "Número de telefone",
            "endereco": "Endereço completo",
            "cidade": "Nome da cidade",
            "estado": "Estado ou província",
            "pais": "Nome do país",
            "cep": "Código postal",
            "data": "Data do registro",
            "data_nascimento": "Data de nascimento",
            "idade": "Idade em anos",
            "sexo": "Sexo ou gênero",
            "cpf": "Número de CPF",
            "cnpj": "Número de CNPJ",
            "rg": "Número de RG",
            "preco": "Valor monetário",
            "valor": "Valor numérico",
            "quantidade": "Quantidade numérica",
            "status": "Status atual",
            "observacao": "Observações adicionais",
            "feedback": "Feedback do cliente",
            "comentario": "Comentário do usuário",
            "produto": "Nome do produto",
            "categoria": "Categoria do item",
            "descricao": "Descrição detalhada",
            "cliente": "Dados do cliente",
            "pedido": "Informações do pedido",
            "pagamento": "Dados de pagamento"
        }
        
        # Procura correspondências exatas
        for key, desc in descriptions.items():
            if field_name.lower() == key.lower():
                return desc
        
        # Procura correspondências parciais
        for key, desc in descriptions.items():
            if key.lower() in field_name.lower():
                parts = field_name.split('_')
                if len(parts) > 1:
                    # Se o campo tem formato "algo_chave", adapta a descrição
                    prefix = parts[0].capitalize()
                    return f"{desc} relacionado a {prefix}"
                return desc
        
        # Formata o nome do campo para uma descrição genérica
        formatted_name = field_name.replace("_", " ").capitalize()
        return f"Campo que representa {formatted_name}"
    
    def analyze_dataset(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Analisa o dataset e retorna informações sobre cada campo.
        
        Args:
            df: DataFrame pandas com os dados
            
        Returns:
            Lista de dicionários com informações de cada campo
        """
        fields_info = []
        
        for column in df.columns:
            # Determina o tipo de dados
            pandas_dtype = str(df[column].dtype)
            
            # Converte tipo pandas para um tipo mais genérico
            if pandas_dtype in self.TYPE_MAPPINGS:
                data_type = self.TYPE_MAPPINGS[pandas_dtype]
            else:
                # Tenta inferir o tipo baseado em heurísticas
                if df[column].dtype == 'object':
                    # Verifica se parece uma data
                    try:
                        pd.to_datetime(df[column].dropna().iloc[0])
                        data_type = "date"
                    except (ValueError, TypeError, IndexError):
                        # Verifica se é principalmente numérico
                        if df[column].dropna().apply(lambda x: str(x).replace('.', '').isdigit()).mean() > 0.8:
                            data_type = "number"
                        else:
                            data_type = "string"
                else:
                    data_type = "unknown"
            
            # Gera aliases para o campo
            aliases = self.generate_aliases(column)
            
            # Infere uma descrição para o campo
            description = self.infer_field_description(column)
            
            # Adiciona informações do campo à lista
            field_info = {
                "name": column,
                "description": description,
                "data_type": data_type,
                "alias": aliases
            }
            
            fields_info.append(field_info)
        
        return fields_info
    
    def process_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Processa um arquivo de dataset e retorna o payload completo.
        
        Args:
            file_path: Caminho para o arquivo de dataset
            
        Returns:
            Lista de dicionários com informações de cada campo
        """
        # Lê o dataset
        df = self.read_dataset(file_path)
        
        # Analisa o dataset
        fields_info = self.analyze_dataset(df)
        
        return fields_info
    
    def save_to_json(self, fields_info: List[Dict[str, Any]], output_path: str) -> None:
        """
        Salva o payload em um arquivo JSON.
        
        Args:
            fields_info: Lista de informações dos campos
            output_path: Caminho para salvar o arquivo JSON
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(fields_info, f, ensure_ascii=False, indent=4)
        print(f"Payload salvo em {output_path}")

# Exemplo de uso
if __name__ == "__main__":
    analyzer = DatasetAnalyzer()
    
    # Exemplo com um arquivo CSV
    file_path = "../dados/vendas_perdidas.csv"
    try:
        fields_info = analyzer.process_file(file_path)
        print(json.dumps(fields_info, indent=4, ensure_ascii=False))
        
        schema = {
            "name": "vendas_perdidas",
            "description": "Descricao",
            "columns": fields_info
        }
        
        # Opcional: salvar em arquivo
        analyzer.save_to_json(schema, "schema_output.json")
    except Exception as e:
        print(f"Erro ao processar arquivo: {str(e)}")