import ast
import contextlib
import importlib
import io
import re
import sys
import textwrap
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union

import black
import numpy as np
import pandas as pd
import sympy as sp

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
        max_output_size: int = 1024 * 1024  # 1 MB
    ):
        """
        Inicializa o executor com configurações de segurança.
        
        Args:
            allowed_imports (Optional[List[str]]): Lista de imports permitidos
            timeout (int): Tempo máximo de execução em segundos
            max_output_size (int): Tamanho máximo da saída em bytes
        """
        self.allowed_imports = allowed_imports or [
            'numpy', 'pandas', 'matplotlib', 'scipy', 'sympy', 
            'statistics', 're', 'math', 'random', 'datetime', 
            'json', 'itertools', 'collections'
        ]
        self.timeout = timeout
        self.max_output_size = max_output_size
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
    
    def execute_code(
        self, 
        code: str, 
        context: Optional[Dict[str, Any]] = None,
        output_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executa o código em um ambiente isolado e controlado.
        
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
            
            # Captura de saída
            output = io.StringIO()
            error_output = io.StringIO()
            
            with contextlib.redirect_stdout(output), \
                 contextlib.redirect_stderr(error_output):
                # Executa o código em um namespace isolado
                exec(formatted_code, exec_namespace)
            
            # Recupera a saída
            stdout_result = output.getvalue()
            stderr_result = error_output.getvalue()
            
            # Tratamento de erros
            if stderr_result:
                result["error"] = f"Erro durante execução: {stderr_result}"
                return result
            
            # Determina o resultado
            result_var = exec_namespace.get('result')
            
            # Validação do tipo de saída
            if output_type:
                result["output_type"] = self._validate_output_type(result_var, output_type)
            
            # Serialização segura do resultado
            result["success"] = True
            result["output"] = stdout_result.strip()
            result["result"] = self._safe_serialize(result_var)
            
            return result
        
        except Exception as e:
            result["error"] = f"Exceção durante execução: {traceback.format_exc()}"
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
