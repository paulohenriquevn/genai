import ast
import contextlib
import importlib
import io
import re
import traceback
import threading
import time
import signal
import multiprocessing
import sys
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Any, Dict, List, Optional, Tuple, Union

import black
import numpy as np
import pandas as pd
import sympy as sp


class TimeoutException(Exception):
    """Exceção levantada quando a execução excede o tempo limite."""
    pass

class AdvancedDynamicCodeExecutor:
    """
    Classe avançada para execução segura e extensível de código gerado por LLM.
    
    Características principais:
    - Execução segura de código
    - Suporte a múltiplos tipos de saída
    - Validação e limpeza avançadas
    - Gerenciamento de dependências
    - Tratamento de diferentes contextos de execução
    """
    
    def __init__(
        self, 
        allowed_imports: Optional[List[str]] = None,
        timeout: int = 30,
        max_output_size: int = 1024 * 1024,  # 1 MB
        use_multiprocessing: bool = True
    ):
        """
        Inicializa o executor com configurações de segurança.
        
        Args:
            allowed_imports (Optional[List[str]]): Lista de imports permitidos
            timeout (int): Tempo máximo de execução em segundos
            max_output_size (int): Tamanho máximo da saída em bytes
            use_multiprocessing (bool): Se True, usa multiprocessing para timeout robusto
                                        Se False, usa threading (mais seguro para alguns ambientes)
        """
        self.allowed_imports = allowed_imports or [
            'numpy', 'pandas', 'matplotlib', 'scipy', 'sympy', 
            'statistics', 're', 'math', 'random', 'datetime', 
            'json', 'itertools', 'collections'
        ]
        self.timeout = timeout
        self.max_output_size = max_output_size
        self.use_multiprocessing = use_multiprocessing
        self.imported_modules = {}
    
    @staticmethod
    def sanitize_code(code: str) -> str:
        """
        Limpa e normaliza o código gerado com regras avançadas.
        
        Args:
            code (str): Código gerado pelo modelo de linguagem.
        
        Returns:
            str: Código limpo e formatado.
        """
        # Remove comentários de bloco e linhas em branco excessivas
        code = re.sub(r'^\s*#.*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'\n{3,}', '\n\n', code)
        
        # Remove imports potencialmente perigosos
        dangerous_imports = [
            'os', 'sys', 'subprocess', 'eval', 'exec', 
            'pickle', 'marshal', 'ctypes', 'threading'
        ]
        for imp in dangerous_imports:
            # Usa substituição menos agressiva
            code = re.sub(
                rf'^(import\s+{imp}|from\s+{imp}\s+import).*$', 
                f'# Import removido por segurança: {imp}', 
                code, 
                flags=re.MULTILINE
            )
        
        # Previne criação de funções perigosas
        code = re.sub(r'lambda\s*.*:\s*exec\(', 'lambda x: None  # Blocked', code)
        
        return code.strip()
    
    def basic_code_validation(self, code: str) -> Tuple[bool, str]:
        """
        Valida a sintaxe do código com verificações avançadas.
        
        Args:
            code (str): Código a ser validado.
        
        Returns:
            Tuple[bool, str]: 
            - Booleano indicando se o código é válido
            - Mensagens de erro (se houver)
        """
        try:
            # Verifica a sintaxe usando AST
            tree = ast.parse(code)
            
            # Análise de nós AST para segurança
            for node in ast.walk(tree):
                # Bloqueia chamadas potencialmente perigosas
                # Removida a verificação de ast.Exec que não existe mais
                if isinstance(node, ast.Global) or isinstance(node, ast.Nonlocal):
                    return False, f"Operação não permitida: {type(node).__name__}"
                
                # Previne imports não autorizados
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        base_module = alias.name.split('.')[0]
                        if base_module not in self.allowed_imports:
                            return False, f"Import não autorizado: {alias.name}"
                
                # Previne chamadas de funções do sistema
                if isinstance(node, ast.Call):
                    if (isinstance(node.func, ast.Name) and 
                        node.func.id in ['open', 'exec', 'eval', 'compile']):
                        return False, f"Chamada não permitida: {node.func.id}"
            
            # Verifica indentação e blocos
            lines = code.split('\n')
            for i, line in enumerate(lines):
                if line.strip() and not re.match(r'^\s*\S', line):
                    return False, f"Erro de indentação na linha {i+1}"
            
            return True, "Código válido"
        
        except SyntaxError as e:
            return False, f"Erro de sintaxe: {e}"
        except Exception as e:
            return False, f"Erro durante validação: {str(e)}"
    
    @staticmethod
    def format_code(code: str) -> str:
        """
        Formata o código usando Black para manter consistência.
        
        Args:
            code (str): Código a ser formatado.
        
        Returns:
            str: Código formatado.
        """
        try:
            return black.format_str(code, mode=black.FileMode())
        except Exception:
            # Se a formatação falhar, retorna o código original
            return code
    
    def _safe_import(self, module_name: str) -> Optional[Any]:
        """
        Importa módulos de forma segura.
        
        Args:
            module_name (str): Nome do módulo a ser importado.
        
        Returns:
            Optional[Any]: Módulo importado ou None
        """
        try:
            # Verifica se o módulo está na lista de imports permitidos
            base_module = module_name.split('.')[0]
            if base_module not in self.allowed_imports:
                return None
            
            # Importa o módulo
            module = importlib.import_module(module_name)
            
            # Armazena o módulo importado
            self.imported_modules[module_name] = module
            
            return module
        except ImportError:
            return None
    
    def _execute_code_internal(
        self,
        formatted_code: str, 
        exec_namespace: Dict[str, Any],
        output: io.StringIO,
        error_output: io.StringIO,
        result_container: Dict[str, Any]
    ) -> None:
        """
        Executa o código em um ambiente isolado.
        Esta função é projetada para ser executada em uma thread separada.
        
        Args:
            formatted_code: Código já formatado e validado
            exec_namespace: Namespace para execução
            output: Buffer para capturar a saída padrão
            error_output: Buffer para capturar a saída de erro
            result_container: Dicionário para armazenar resultados
        """
        try:
            with contextlib.redirect_stdout(output), \
                 contextlib.redirect_stderr(error_output):
                # Executa o código em um namespace isolado
                exec(formatted_code, exec_namespace)
            
            # Armazena o resultado da execução no container
            result_container["success"] = True
            result_container["result_var"] = exec_namespace.get('result')
            
        except Exception as e:
            result_container["success"] = False
            result_container["error"] = f"Exceção durante execução: {traceback.format_exc()}"
            
    def _execute_in_process(
        self, 
        formatted_code: str, 
        namespace_dict: Dict[str, Any],
        result_queue: multiprocessing.Queue
    ) -> None:
        """
        Executa código em processo separado para interrupção segura.
        
        Args:
            formatted_code: Código formatado e validado
            namespace_dict: Dicionário com variáveis de ambiente para execução
            result_queue: Fila para retornar resultados
        """
        try:
            # Recria o namespace a partir do dicionário
            # Obs: nem todas as variáveis podem ser serializadas para multiprocessing
            exec_namespace = {
                'np': np,
                'pd': pd,
                'sp': sp,
                'math': __import__('math'),
                'random': __import__('random'),
                'datetime': __import__('datetime'),
                'json': __import__('json'),
            }
            
            # Adiciona variáveis simples do contexto original
            for key, value in namespace_dict.items():
                # Só adiciona tipos serializáveis básicos
                if isinstance(value, (int, float, str, bool, list, dict, tuple, set)):
                    exec_namespace[key] = value
                    
            # Captura de saída
            output = io.StringIO()
            error_output = io.StringIO()
            
            # Executa o código
            with contextlib.redirect_stdout(output), \
                 contextlib.redirect_stderr(error_output):
                exec(formatted_code, exec_namespace)
                
            # Obtém o resultado
            result_var = exec_namespace.get('result')
            
            # Envia resultado pela fila
            result_queue.put({
                "success": True,
                "result_var": self._safe_serialize(result_var),
                "stdout": output.getvalue(),
                "stderr": error_output.getvalue()
            })
            
        except Exception as e:
            # Envia erro pela fila
            result_queue.put({
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            })

    def execute_code(
        self, 
        code: str, 
        context: Optional[Dict[str, Any]] = None,
        output_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executa o código em um ambiente isolado e controlado com timeout.
        
        Args:
            code (str): Código a ser executado.
            context (Optional[Dict[str, Any]], optional): 
                Contexto opcional para execução do código.
            output_type (Optional[str]): Tipo de saída esperado.
        
        Returns:
            Dict[str, Any]: Dicionário de resultado com status, saída, etc.
        """
        # Configurações iniciais
        context = context or {}
        result = {
            "success": False,
            "output": "",
            "error": "",
            "result": None,
            "output_type": None
        }
        
        # Limpa o código
        try:
            # Modificação para garantir captura de resultado
            modified_code = (
                "# Código original\n" + 
                code + 
                "\n\n# Captura de resultado\n" +
                "result = locals().get('result', locals().get('resultado', locals().get('df', locals().get('data', None))))"
            )
            
            # Limpa o código
            sanitized_code = self.sanitize_code(modified_code)
            
            # Formata o código
            formatted_code = self.format_code(sanitized_code)
            
            # Valida o código
            is_valid, validation_msg = self.basic_code_validation(formatted_code)
            if not is_valid:
                result["error"] = validation_msg
                return result
            
            # Preparação do ambiente de execução
            exec_namespace = {
                'np': np,
                'pd': pd,
                'sp': sp,
                'math': __import__('math'),
                'random': __import__('random'),
                'datetime': __import__('datetime'),
                'json': __import__('json'),
                'import_module': self._safe_import
            }
            exec_namespace.update(context)
            
            # Determina qual método de execução usar
            if self.use_multiprocessing and self._can_use_multiprocessing(context):
                # Executa com multiprocessing (mais seguro para interrupção)
                return self._execute_with_multiprocessing(
                    formatted_code, exec_namespace, output_type, result
                )
            else:
                # Executa com threading (mais compatível, não interrompe realmente)
                return self._execute_with_threading(
                    formatted_code, exec_namespace, output_type, result
                )
        
        except TimeoutException:
            result["error"] = f"Timeout de execução excedido ({self.timeout} segundos)"
            return result
        except Exception as e:
            result["error"] = f"Exceção durante execução: {traceback.format_exc()}"
            return result
            
    def _can_use_multiprocessing(self, context: Dict[str, Any]) -> bool:
        """
        Verifica se multiprocessing pode ser usado com o contexto fornecido.
        Algumas variáveis complexas não podem ser passadas entre processos.
        
        Args:
            context: O contexto de execução para verificação
            
        Returns:
            bool: True se multiprocessing é seguro para usar
        """
        # Verifica se há objetos não serializáveis no contexto
        for key, value in context.items():
            # Tipos que são problemáticos para serialização entre processos
            if isinstance(value, (pd.DataFrame, pd.Series, np.ndarray)):
                return False
                
            # Funções e métodos não são serializáveis
            if callable(value):
                return False
                
            # Objetos personalizados provavelmente não são serializáveis
            if not isinstance(value, (int, float, str, bool, list, dict, tuple, set, type(None))):
                return False
                
        return True
        
    def _execute_with_threading(
        self, 
        formatted_code: str, 
        exec_namespace: Dict[str, Any],
        output_type: Optional[str],
        result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Executa código usando threading para timeout.
        Mais compatível, mas não garante interrupção real.
        
        Args:
            formatted_code: Código já formatado e validado
            exec_namespace: Namespace para execução
            output_type: Tipo de saída esperado
            result: Dicionário para armazenar resultado
            
        Returns:
            Dict[str, Any]: Dicionário de resultado com status, saída, etc.
        """
        # Captura de saída
        output = io.StringIO()
        error_output = io.StringIO()
        
        # Container para compartilhar resultados entre threads
        execution_result = {
            "success": False,
            "result_var": None,
            "error": ""
        }
        
        # Executa o código em uma thread separada com timeout
        execution_thread = threading.Thread(
            target=self._execute_code_internal,
            args=(formatted_code, exec_namespace, output, error_output, execution_result),
            daemon=True
        )
        
        # Inicia a thread
        execution_thread.start()
        
        # Aguarda a conclusão com timeout
        execution_thread.join(timeout=self.timeout)
        
        # Verifica se a thread ainda está em execução (timeout ocorreu)
        if execution_thread.is_alive():
            # Não é possível matar threads em Python de forma segura,
            # mas podemos sinalizar o timeout e continuar
            result["success"] = False
            result["error"] = f"Timeout de execução excedido ({self.timeout} segundos)"
            return result
        
        # Recupera a saída
        stdout_result = output.getvalue()
        stderr_result = error_output.getvalue()
        
        # Verifica se houve erro durante a execução
        if not execution_result["success"]:
            result["error"] = execution_result.get("error", "Erro durante execução")
            return result
        
        # Tratamento de erros de stderr
        if stderr_result:
            result["error"] = f"Erro durante execução: {stderr_result}"
            return result
        
        # Determina o resultado
        result_var = execution_result["result_var"]
        
        # Validação do tipo de saída
        if output_type:
            result["output_type"] = self._validate_output_type(result_var, output_type)
        
        # Serialização segura do resultado
        result["success"] = True
        result["output"] = stdout_result.strip()
        result["result"] = self._safe_serialize(result_var)
        
        return result
        
    def _execute_with_multiprocessing(
        self,
        formatted_code: str,
        exec_namespace: Dict[str, Any],
        output_type: Optional[str],
        result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Executa código usando multiprocessing para timeout.
        Permite interrupção real de código que trava, mas menos compatível.
        
        Args:
            formatted_code: Código já formatado e validado
            exec_namespace: Namespace para execução
            output_type: Tipo de saída esperado
            result: Dicionário para armazenar resultado
            
        Returns:
            Dict[str, Any]: Dicionário de resultado com status, saída, etc.
        """
        try:
            # Cria uma fila para comunicação entre processos
            result_queue = multiprocessing.Queue()
            
            # Filtra o namespace para ter apenas valores serializáveis
            filtered_namespace = {}
            for key, value in exec_namespace.items():
                if isinstance(value, (int, float, str, bool, list, dict, tuple, set, type(None))):
                    filtered_namespace[key] = value
            
            # Cria o processo
            process = multiprocessing.Process(
                target=self._execute_in_process,
                args=(formatted_code, filtered_namespace, result_queue),
                daemon=True
            )
            
            # Inicia o processo
            process.start()
            
            # Aguarda a conclusão com timeout
            process.join(timeout=self.timeout)
            
            # Verifica se o processo ainda está em execução (timeout ocorreu)
            if process.is_alive():
                # No caso de multiprocessing, podemos encerrar o processo
                process.terminate()
                process.join(1)  # Dá 1 segundo para finalização
                
                # Se ainda estiver vivo, usa medidas mais drásticas (SIGKILL)
                if process.is_alive():
                    os.kill(process.pid, signal.SIGKILL)
                
                result["success"] = False
                result["error"] = f"Timeout de execução excedido ({self.timeout} segundos)"
                return result
            
            # Obtém o resultado da fila, se disponível
            try:
                process_result = result_queue.get(block=False)
                
                # Verifica se o processo teve sucesso
                if not process_result.get("success", False):
                    result["error"] = process_result.get("error", "Erro durante execução")
                    if "traceback" in process_result:
                        result["error"] += f"\n{process_result['traceback']}"
                    return result
                
                # Recupera saídas
                stdout_result = process_result.get("stdout", "")
                stderr_result = process_result.get("stderr", "")
                
                # Tratamento de erros de stderr
                if stderr_result:
                    result["error"] = f"Erro durante execução: {stderr_result}"
                    return result
                
                # Determina o resultado
                result_var = process_result.get("result_var")
                
                # Validação do tipo de saída
                if output_type:
                    result["output_type"] = self._validate_output_type(result_var, output_type)
                
                # Configura o resultado
                result["success"] = True
                result["output"] = stdout_result.strip()
                result["result"] = result_var  # Já serializado pelo processo filho
                
                return result
                
            except Exception as e:
                result["error"] = f"Erro ao obter resultado do processo: {str(e)}"
                return result
                
        except Exception as e:
            result["error"] = f"Erro no gerenciamento do processo: {traceback.format_exc()}"
            return result
    
    def _validate_output_type(
        self, 
        value: Any, 
        expected_type: str
    ) -> Optional[str]:
        """
        Valida o tipo de saída de acordo com o esperado.
        
        Args:
            value (Any): Valor a ser validado
            expected_type (str): Tipo esperado
        
        Returns:
            Optional[str]: Tipo correspondente ou None
        """
        type_mapping = {
            "number": (int, float, np.number),
            "string": str,
            "list": list,
            "dict": dict,
            "dataframe": pd.DataFrame,
            "series": pd.Series,
            "array": np.ndarray,
            "plot": (str, bytes)
        }
        
        if expected_type not in type_mapping:
            return None
        
        expected_cls = type_mapping[expected_type]
        if isinstance(value, expected_cls):
            return expected_type
        
        return None
    
    def _safe_serialize(self, obj: Any) -> Any:
        """
        Serializa objetos de forma segura, limitando tipos complexos.
        
        Args:
            obj (Any): Objeto a ser serializado
        
        Returns:
            Any: Objeto serializado ou representação segura
        """
        # Tipos primitivos podem ser serializados diretamente
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        
        # Serialização de estruturas de dados comuns
        if isinstance(obj, (list, tuple, set)):
            return [self._safe_serialize(item) for item in obj]
        
        if isinstance(obj, dict):
            return {
                self._safe_serialize(k): self._safe_serialize(v) 
                for k, v in obj.items()
            }
        
        # Serialização de DataFrames e Series
        if isinstance(obj, (pd.DataFrame, pd.Series)):
            return obj.to_dict()
        
        # Serialização de arrays numpy
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        
        # Para objetos complexos, tenta uma representação segura
        try:
            return str(obj)
        except Exception:
            return repr(obj)
    
    def analyze_code_complexity(self, code: str) -> Dict[str, Any]:
        """
        Analisa a complexidade do código.
        
        Args:
            code (str): Código a ser analisado
        
        Returns:
            Dict[str, Any]: Métricas de complexidade
        """
        try:
            tree = ast.parse(code)
            
            # Contadores básicos
            metrics = {
                "lines_of_code": len(code.split('\n')),
                "functions": 0,
                "classes": 0,
                "imports": 0,
                "complexity": 0
            }
            
            # Análise de complexidade ciclomática
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    metrics["functions"] += 1
                    metrics["complexity"] += sum(
                        1 for child in ast.walk(node)
                        if isinstance(child, (ast.If, ast.While, ast.For, ast.Try, ast.ExceptHandler))
                    )
                elif isinstance(node, ast.ClassDef):
                    metrics["classes"] += 1
                elif isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                    metrics["imports"] += 1
            
            return metrics
        
        except Exception as e:
            return {"error": str(e)}
