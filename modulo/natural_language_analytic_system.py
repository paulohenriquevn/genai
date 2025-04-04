"""
Sistema Integrado de Consulta em Linguagem Natural
==================================================

Este módulo integra todos os componentes desenvolvidos em um sistema
completo de análise de dados por linguagem natural:

1. Conectores de dados
2. Motor de consulta
3. Modelos de linguagem (LLM)
4. Execução segura de código
5. API REST

O objetivo é fornecer uma interface completa e fácil de usar para análise
de dados usando comandos em linguagem natural, sem necessidade de
conhecimentos em programação ou SQL.
"""

import os
import sys
import json
import logging
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Union, Tuple

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("sistema_integrado.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("sistema_integrado")

# Importação dos módulos próprios
from natural_query_engine import NaturalLanguageQueryEngine
from llm_integration import LLMIntegration, LLMQueryGenerator, ModelType

# Classe principal do sistema integrado
class NaturalLanguageAnalyticSystem:
    """
    Sistema integrado de análise de dados por linguagem natural.
    
    Esta classe integra todos os componentes do sistema e fornece
    uma interface unificada e de alto nível para consultas em
    linguagem natural sobre dados estruturados.
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        data_dir: Optional[str] = None,
        llm_config_path: Optional[str] = None,
        output_dir: Optional[str] = None
    ):
        """
        Inicializa o sistema integrado de análise.
        
        Args:
            config_path: Caminho para configuração do sistema
            data_dir: Diretório de dados
            llm_config_path: Caminho para configuração do LLM
            output_dir: Diretório para saídas
        """
        # Configura caminhos
        self.config_path = config_path or os.path.join(os.getcwd(), "config.json")
        self.data_dir = data_dir or os.path.join(os.getcwd(), "dados")
        self.output_dir = output_dir or os.path.join(os.getcwd(), "output")
        
        # Cria diretórios se não existirem
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Carrega configurações
        self.config = self._load_config()
        
        # Inicializa o gerador de consultas com modelo de linguagem
        self.llm_generator = LLMQueryGenerator(config_path=llm_config_path)
        
        # Inicializa o motor de consulta
        self.query_engine = self._init_query_engine()
        
        # Registra o gerador de consultas no motor
        self.query_engine._call_language_model = self.llm_generator.generate_code
        
        logger.info("Sistema integrado de análise inicializado com sucesso")
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Carrega a configuração do sistema.
        
        Returns:
            Dict[str, Any]: Configuração do sistema
        """
        # Configuração padrão
        default_config = {
            "data_sources": {
                "files": [
                    {"name": "vendas", "path": os.path.join(self.data_dir, "vendas.csv"), "type": "csv"},
                    {"name": "clientes", "path": os.path.join(self.data_dir, "clientes.csv"), "type": "csv"},
                    {"name": "vendas_perdidas", "path": os.path.join(self.data_dir, "vendas_perdidas.csv"), "type": "csv"}
                ],
                "connections": []
            },
            "output_types": ["string", "number", "dataframe", "plot"],
            "api": {
                "enabled": True,
                "host": "0.0.0.0",
                "port": 8000
            }
        }
        
        # Tenta carregar do arquivo
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    file_config = json.load(f)
                    # Mescla com os valores padrão
                    self._deep_merge(default_config, file_config)
            except Exception as e:
                logger.error(f"Erro ao carregar configurações: {str(e)}")
        else:
            # Salva a configuração padrão
            logger.info(f"Arquivo de configuração não encontrado. Criando em {self.config_path}")
            with open(self.config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
        
        return default_config
    
    def _deep_merge(self, base: Dict, update: Dict) -> None:
        """
        Mescla dois dicionários de forma recursiva.
        
        Args:
            base: Dicionário base a ser atualizado
            update: Dicionário com atualizações
        """
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def _init_query_engine(self) -> NaturalLanguageQueryEngine:
        """
        Inicializa o motor de consulta com base nas configurações.
        
        Returns:
            NaturalLanguageQueryEngine: Motor de consulta configurado
        """
        # Cria uma lista de arquivos para o datasources.json
        data_sources = []
        
        for file_config in self.config["data_sources"]["files"]:
            source = {
                "id": file_config["name"],
                "type": file_config["type"],
                "path": file_config["path"],
                "encoding": file_config.get("encoding", "utf-8"),
                "delimiter": file_config.get("delimiter", ",")
            }
            data_sources.append(source)
        
        # Adiciona conexões de banco de dados
        for conn_config in self.config["data_sources"]["connections"]:
            source = {
                "id": conn_config["name"],
                "type": conn_config["type"]
            }
            # Adiciona outros parâmetros de conexão
            for key, value in conn_config.items():
                if key != "name" and key != "type":
                    source[key] = value
            
            data_sources.append(source)
        
        # Cria o arquivo de configuração datasources.json
        datasources_config = {"data_sources": data_sources}
        datasources_path = os.path.join(os.getcwd(), "datasources.json")
        
        with open(datasources_path, 'w') as f:
            json.dump(datasources_config, f, indent=2)
        
        # Inicializa o motor de consulta
        query_engine = NaturalLanguageQueryEngine(
            data_config_path=datasources_path,
            metadata_config_path=os.path.join(os.getcwd(), "metadata.json"),
            output_types=self.config["output_types"],
            base_data_path=self.data_dir
        )
        
        return query_engine
    
    def process_query(self, query: str, output_type: Optional[str] = None) -> Tuple[Any, str]:
        """
        Processa uma consulta em linguagem natural.
        
        Args:
            query: Consulta em linguagem natural
            output_type: Tipo de saída esperado
            
        Returns:
            Tuple[Any, str]: Resultado e tipo de resultado
        """
        # Define o tipo de saída esperado (se especificado)
        if output_type:
            self.query_engine.agent_state.output_type = output_type
        
        # Processa a consulta
        response = self.query_engine.execute_query(query)
        
        # Retorna o resultado e o tipo
        return response.value, response.type
    
    def load_data_from_file(
        self,
        file_path: str,
        name: Optional[str] = None,
        file_type: Optional[str] = None
    ) -> bool:
        """
        Carrega dados de um arquivo.
        
        Args:
            file_path: Caminho para o arquivo
            name: Nome para a fonte de dados
            file_type: Tipo do arquivo (csv, excel, json)
            
        Returns:
            bool: True se os dados foram carregados com sucesso
        """
        try:
            # Determina o nome da fonte
            if not name:
                name = os.path.splitext(os.path.basename(file_path))[0]
            
            # Determina o tipo do arquivo
            if not file_type:
                if file_path.endswith(".csv"):
                    file_type = "csv"
                elif file_path.endswith((".xls", ".xlsx")):
                    file_type = "excel"
                elif file_path.endswith(".json"):
                    file_type = "json"
                else:
                    file_type = "csv"  # Padrão
            
            # Carrega o arquivo
            if file_type == "csv":
                df = pd.read_csv(file_path)
            elif file_type == "excel":
                df = pd.read_excel(file_path)
            elif file_type == "json":
                df = pd.read_json(file_path)
            else:
                raise ValueError(f"Tipo de arquivo não suportado: {file_type}")
            
            # Adiciona ao motor de consulta
            from core.dataframe import DataFrameWrapper
            wrapper = DataFrameWrapper(df, name)
            self.query_engine.dataframes[name] = wrapper
            self.query_engine.agent_state.dfs = list(self.query_engine.dataframes.values())
            
            # Adiciona à configuração
            self.config["data_sources"]["files"].append({
                "name": name,
                "path": file_path,
                "type": file_type
            })
            
            # Salva a configuração atualizada
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            logger.info(f"Dados carregados com sucesso: {name} ({len(df)} registros)")
            
            return True
        
        except Exception as e:
            logger.error(f"Erro ao carregar arquivo {file_path}: {str(e)}")
            return False
    
    def get_available_datasources(self) -> List[str]:
        """
        Retorna a lista de fontes de dados disponíveis.
        
        Returns:
            List[str]: Lista de nomes das fontes de dados
        """
        return list(self.query_engine.dataframes.keys())
    
    def get_datasource_info(self, name: str) -> Dict[str, Any]:
        """
        Retorna informações sobre uma fonte de dados.
        
        Args:
            name: Nome da fonte de dados
            
        Returns:
            Dict[str, Any]: Informações sobre a fonte de dados
        """
        if name not in self.query_engine.dataframes:
            raise ValueError(f"Fonte de dados não encontrada: {name}")
        
        # Obtém o DataFrame
        df = self.query_engine.dataframes[name].dataframe
        
        # Prepara informações básicas
        info = {
            "name": name,
            "rows": len(df),
            "columns": list(df.columns),
            "dtypes": {col: str(df[col].dtype) for col in df.columns},
            "sample": df.head(5).to_dict(orient='records'),
            "null_counts": {col: int(df[col].isna().sum()) for col in df.columns}
        }
        
        # Adiciona estatísticas para colunas numéricas
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if numeric_cols:
            info["numeric_stats"] = df[numeric_cols].describe().to_dict()
        
        return info
    
    def execute_sql_query(self, sql_query: str) -> pd.DataFrame:
        """
        Executa uma consulta SQL diretamente.
        
        Args:
            sql_query: Consulta SQL
            
        Returns:
            DataFrame com os resultados
        """
        return self.query_engine.execute_sql_query(sql_query)
    
    def start_api_server(self, host: Optional[str] = None, port: Optional[int] = None) -> None:
        """
        Inicia o servidor API.
        
        Args:
            host: Host para o servidor (default: valor na configuração)
            port: Porta para o servidor (default: valor na configuração)
        """
        if not self.config["api"]["enabled"]:
            logger.warning("API não está habilitada na configuração")
            return
        
        try:
            # Importa os módulos necessários
            import uvicorn
            from api_service import app
            
            # Configura a referência ao sistema integrado na API
            import api_service
            api_service.engine = self.query_engine
            
            # Define host e porta
            api_host = host or self.config["api"]["host"]
            api_port = port or self.config["api"]["port"]
            
            logger.info(f"Iniciando servidor API em {api_host}:{api_port}")
            
            # Inicia o servidor
            uvicorn.run(app, host=api_host, port=api_port)
            
        except ImportError as e:
            logger.error(f"Erro ao importar dependências da API: {str(e)}")
            logger.error("Instale com: pip install fastapi uvicorn")
        except Exception as e:
            logger.error(f"Erro ao iniciar servidor API: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Retorna estatísticas do sistema.
        
        Returns:
            Dict[str, Any]: Estatísticas do sistema
        """
        # Obtém estatísticas do motor de consulta
        query_stats = self.query_engine.get_stats()
        
        # Obtém estatísticas do gerador LLM
        llm_stats = self.llm_generator.get_stats()
        
        # Combina as estatísticas
        stats = {
            "query_engine": query_stats,
            "llm": llm_stats,
            "datasources": {
                "count": len(self.query_engine.dataframes),
                "names": list(self.query_engine.dataframes.keys()),
                "total_records": sum(len(df.dataframe) for df in self.query_engine.dataframes.values())
            },
            "system": {
                "version": "1.0.0",
                "python_version": sys.version,
                "memory_usage_mb": self._get_memory_usage(),
                "uptime_seconds": self._get_uptime()
            }
        }
        
        return stats
    
    def _get_memory_usage(self) -> float:
        """
        Retorna o uso de memória em MB.
        
        Returns:
            float: Uso de memória em MB
        """
        try:
            import os
            import psutil
            
            # Obtém o processo atual
            process = psutil.Process(os.getpid())
            
            # Retorna o uso de memória em MB
            return process.memory_info().rss / (1024 * 1024)
        except:
            return 0.0
    
    def _get_uptime(self) -> float:
        """
        Retorna o tempo de execução em segundos.
        
        Returns:
            float: Tempo de execução em segundos
        """
        import time
        
        # Valor padrão para tempo de início
        if not hasattr(self, "_start_time"):
            self._start_time = time.time()
        
        return time.time() - self._start_time


# Interface interativa de linha de comando
def interactive_cli():
    """Interface interativa de linha de comando para o sistema"""
    print("\n===== Sistema Integrado de Análise por Linguagem Natural =====\n")
    
    # Inicializa o sistema
    system = NaturalLanguageAnalyticSystem()
    
    # Exibe fontes de dados disponíveis
    datasources = system.get_available_datasources()
    print(f"Fontes de dados disponíveis: {', '.join(datasources)}\n")
    
    print("Digite 'ajuda' para ver comandos disponíveis ou 'sair' para encerrar.\n")
    
    while True:
        # Solicita um comando
        command = input("\nComando > ").strip()
        
        if not command:
            continue
        
        # Processa comandos especiais
        if command.lower() in ['sair', 'exit', 'quit']:
            break
        elif command.lower() in ['ajuda', 'help']:
            _show_help()
            continue
        elif command.lower().startswith('carregar '):
            # Carregar um arquivo de dados
            parts = command.split(' ', 1)
            if len(parts) > 1:
                file_path = parts[1].strip()
                print(f"\nCarregando arquivo: {file_path}")
                if system.load_data_from_file(file_path):
                    print("\nArquivo carregado com sucesso!")
                else:
                    print("\nErro ao carregar o arquivo.")
            continue
        elif command.lower().startswith('info '):
            # Exibir informações sobre uma fonte de dados
            parts = command.split(' ', 1)
            if len(parts) > 1:
                name = parts[1].strip()
                try:
                    info = system.get_datasource_info(name)
                    print(f"\nInformações sobre {name}:")
                    print(f"  Registros: {info['rows']}")
                    print(f"  Colunas: {', '.join(info['columns'])}")
                    print("\nAmostra:")
                    for i, row in enumerate(info['sample']):
                        print(f"  {i+1}. {row}")
                except Exception as e:
                    print(f"\nErro: {str(e)}")
            continue
        elif command.lower() == 'sql':
            # Executar uma consulta SQL direta
            sql = input("\nDigite a consulta SQL > ")
            if sql.strip():
                try:
                    result = system.execute_sql_query(sql)
                    print("\nResultado:")
                    print(result.head(10))
                    print(f"\n[{len(result)} registros no total]")
                except Exception as e:
                    print(f"\nErro: {str(e)}")
            continue
        elif command.lower() == 'stats':
            # Exibir estatísticas do sistema
            stats = system.get_stats()
            print("\nEstatísticas do Sistema:")
            print(f"  Total de consultas: {stats['query_engine']['total_queries']}")
            print(f"  Taxa de sucesso: {stats['query_engine']['success_rate']:.1f}%")
            print(f"  Modelo utilizado: {stats['llm']['model_type']} ({stats['llm']['model_name']})")
            print(f"  Tempo médio de geração: {stats['llm']['avg_generation_time']:.2f}s")
            print(f"  Fontes de dados: {', '.join(stats['datasources']['names'])}")
            print(f"  Total de registros: {stats['datasources']['total_records']}")
            print(f"  Uso de memória: {stats['system']['memory_usage_mb']:.1f} MB")
            continue
        elif command.lower() == 'api':
            # Iniciar o servidor API
            print("\nIniciando servidor API (pressione Ctrl+C para parar)...")
            try:
                system.start_api_server()
            except KeyboardInterrupt:
                print("\nServidor API interrompido.")
            continue
        
        # Trata como uma consulta em linguagem natural
        try:
            print("\nProcessando consulta...")
            result, result_type = system.process_query(command)
            
            print("\n----- Resposta -----\n")
            
            if result_type == "dataframe":
                print(result.head(10))
                print(f"\n[{len(result)} registros no total]")
            elif result_type == "plot":
                print("[Visualização gerada]")
                # Salva a visualização
                viz_path = os.path.join(system.output_dir, "ultima_visualizacao.png")
                import base64
                from PIL import Image
                import io
                
                # Extrai a imagem do base64
                if isinstance(result, str) and result.startswith("data:image/png;base64,"):
                    img_data = result.split(",")[1]
                    img = Image.open(io.BytesIO(base64.b64decode(img_data)))
                    img.save(viz_path)
                    print(f"\nVisualização salva em: {viz_path}")
            else:
                print(result)
            
            print("\n-------------------\n")
            
        except Exception as e:
            print(f"\nErro ao processar consulta: {str(e)}\n")
    
    print("\nObrigado por usar o Sistema Integrado de Análise por Linguagem Natural!")

def _show_help():
    """Exibe ajuda sobre os comandos disponíveis"""
    print("\nComandos disponíveis:")
    print("  - [pergunta em linguagem natural]  Processa uma consulta sobre os dados")
    print("  - carregar [caminho do arquivo]    Carrega um arquivo de dados")
    print("  - info [nome da fonte]             Exibe informações sobre uma fonte de dados")
    print("  - sql                              Executa uma consulta SQL direta")
    print("  - stats                            Exibe estatísticas do sistema")
    print("  - api                              Inicia o servidor API web")
    print("  - ajuda                            Exibe esta mensagem de ajuda")
    print("  - sair                             Encerra o programa\n")


# Função principal que permite iniciar o sistema de várias formas
def main():
    """Função principal para iniciar o sistema"""
    import argparse
    
    # Configura o parser de argumentos
    parser = argparse.ArgumentParser(description="Sistema Integrado de Análise por Linguagem Natural")
    parser.add_argument("--api", action="store_true", help="Inicia apenas o servidor API")
    parser.add_argument("--config", type=str, help="Caminho para o arquivo de configuração")
    parser.add_argument("--data-dir", type=str, help="Diretório de dados")
    parser.add_argument("--llm-config", type=str, help="Caminho para configuração do LLM")
    parser.add_argument("--output-dir", type=str, help="Diretório para saídas")
    parser.add_argument("--query", type=str, help="Executa uma única consulta e sai")
    
    # Parse os argumentos
    args = parser.parse_args()
    
    # Inicializa o sistema
    system = NaturalLanguageAnalyticSystem(
        config_path=args.config,
        data_dir=args.data_dir,
        llm_config_path=args.llm_config,
        output_dir=args.output_dir
    )
    
    # Executa de acordo com o modo escolhido
    if args.api:
        # Inicia apenas o servidor API
        print("Iniciando servidor API...")
        system.start_api_server()
    elif args.query:
        # Executa uma única consulta e sai
        print(f"Executando consulta: {args.query}")
        result, result_type = system.process_query(args.query)
        
        if result_type == "dataframe":
            print(result.head(10))
            print(f"\n[{len(result)} registros no total]")
        elif result_type == "plot":
            # Salva a visualização
            viz_path = os.path.join(system.output_dir, "consulta_resultado.png")
            import base64
            from PIL import Image
            import io
            
            # Extrai a imagem do base64
            if isinstance(result, str) and result.startswith("data:image/png;base64,"):
                img_data = result.split(",")[1]
                img = Image.open(io.BytesIO(base64.b64decode(img_data)))
                img.save(viz_path)
                print(f"Visualização salva em: {viz_path}")
        else:
            print(result)
    else:
        # Inicia a interface interativa
        interactive_cli()


if __name__ == "__main__":
    main()