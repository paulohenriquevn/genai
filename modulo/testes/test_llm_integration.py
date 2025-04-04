#!/usr/bin/env python3
"""
Testes para integração com modelos de linguagem (LLM)
====================================================

Este módulo contém testes para verificar a integração do sistema com
diferentes modelos de linguagem, incluindo:

- Teste com modelo simulado (mock)
- Teste com OpenAI (opcional)
- Teste com Anthropic (opcional)
- Testes de fallback quando o LLM falha
"""

import unittest
import os
import sys
import json
from unittest.mock import patch, MagicMock

# Adiciona diretório pai ao PATH para importar módulos adequadamente
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llm_integration import (
    LLMIntegration, 
    ModelType, 
    LLMQueryGenerator,
    create_llm_integration
)


class TestLLMIntegration(unittest.TestCase):
    """Testes para a integração com modelos de linguagem"""
    
    def setUp(self):
        """Configurações iniciais para cada teste"""
        # Configuração para testes com modelo mock
        self.mock_llm = LLMIntegration(model_type=ModelType.MOCK)
        
        # Configuração temporária para testes
        self.test_prompt = """
        Gere código Python para responder à seguinte consulta:
        
        Mostre o total de vendas agrupadas por cliente e 
        crie um gráfico de barras para visualizar os resultados.
        """
    
    def test_mock_integration(self):
        """Testa a integração com o modelo simulado (mock)"""
        # Verifica se o modelo mock foi inicializado corretamente
        self.assertEqual(self.mock_llm.model_type, ModelType.MOCK)
        
        # Gera código com o modelo mock
        code = self.mock_llm.generate_code(self.test_prompt)
        
        # Verifica se o código foi gerado
        self.assertIsNotNone(code)
        self.assertIsInstance(code, str)
        self.assertIn("import pandas", code)
        self.assertIn("execute_sql_query", code)
        self.assertIn("result", code)
    
    def test_model_type_enum(self):
        """Testa a enumeração de tipos de modelos"""
        # Verifica os valores da enumeração
        self.assertEqual(ModelType.OPENAI.value, "openai")
        self.assertEqual(ModelType.HUGGINGFACE.value, "huggingface")
        self.assertEqual(ModelType.ANTHROPIC.value, "anthropic")
        self.assertEqual(ModelType.LOCAL.value, "local")
        self.assertEqual(ModelType.MOCK.value, "mock")
        
        # Testa a conversão de string para enum
        self.assertEqual(ModelType("openai"), ModelType.OPENAI)
        self.assertEqual(ModelType("mock"), ModelType.MOCK)
    
    def test_create_llm_integration(self):
        """Testa a função factory para criar integrações LLM"""
        # Cria integração com configuração padrão
        llm = create_llm_integration()
        self.assertEqual(llm.model_type, ModelType.MOCK)
        
        # Cria um arquivo de configuração temporário
        temp_config = {
            "model_type": "mock",
            "model_name": "test_model",
            "api_key": "test_key",
            "api_endpoint": "test_endpoint",
            "config": {"temperature": 0.7}
        }
        
        with open("temp_llm_config.json", "w") as f:
            json.dump(temp_config, f)
        
        try:
            # Cria integração com arquivo de configuração
            llm = create_llm_integration("temp_llm_config.json")
            self.assertEqual(llm.model_type, ModelType.MOCK)
            self.assertEqual(llm.model_name, "test_model")
            self.assertEqual(llm.api_key, "test_key")
            self.assertEqual(llm.api_endpoint, "test_endpoint")
        finally:
            # Remove o arquivo temporário
            if os.path.exists("temp_llm_config.json"):
                os.remove("temp_llm_config.json")
    
    @patch('llm_integration.LLMIntegration')
    def test_llm_query_generator(self, mock_llm_integration):
        """Testa o gerador de consultas usando LLM"""
        # Configura o mock
        mock_instance = MagicMock()
        mock_instance.model_type = ModelType.MOCK
        mock_instance.generate_code.return_value = "# Mock code\nresult = {'type': 'string', 'value': 'Test result'}"
        mock_llm_integration.return_value = mock_instance
        
        # Cria o gerador de consultas
        generator = LLMQueryGenerator()
        
        # Gera código
        code = generator.generate_code(self.test_prompt)
        
        # Verifica se o método foi chamado corretamente
        mock_instance.generate_code.assert_called_once_with(self.test_prompt)
        self.assertEqual(code, "# Mock code\nresult = {'type': 'string', 'value': 'Test result'}")
        
        # Verifica as estatísticas
        stats = generator.get_stats()
        self.assertEqual(stats["query_count"], 1)
        self.assertEqual(stats["error_count"], 0)
    
    def test_openai_integration_fallback(self):
        """Testa o fallback quando a integração com OpenAI falha"""
        # Tenta criar uma integração OpenAI sem API key
        with patch('llm_integration.openai') as mock_openai:
            # Configura o mock para lançar uma exceção
            mock_openai.api_key = None
            mock_openai.api_base = None
            
            # Inicializa com OpenAI, mas deve fazer fallback para mock
            llm = LLMIntegration(model_type=ModelType.OPENAI, model_name="gpt-3.5-turbo")
            
            # Verifica se fez fallback para o modelo mock
            self.assertEqual(llm.model_type, ModelType.MOCK)
            
            # Verifica se ainda gera código
            code = llm.generate_code(self.test_prompt)
            self.assertIsNotNone(code)
            self.assertIn("import pandas", code)
    
    def test_generate_code_error_handling(self):
        """Testa o tratamento de erros na geração de código"""
        # Cria uma instância do LLMIntegration com um modelo falho
        with patch.object(self.mock_llm, '_generate_mock', side_effect=Exception("Test error")):
            # Mesmo com erro, deve retornar um código fallback
            code = self.mock_llm.generate_code(self.test_prompt)
            self.assertIsNotNone(code)
            self.assertIn("fallback", code)
    
    @unittest.skipIf(not os.environ.get("OPENAI_API_KEY"), "API key not available")
    def test_openai_integration_real(self):
        """Testa a integração real com OpenAI (requer API key)"""
        # Este teste só é executado se a API key da OpenAI estiver disponível
        llm = LLMIntegration(
            model_type=ModelType.OPENAI,
            model_name="gpt-3.5-turbo",
            api_key=os.environ.get("OPENAI_API_KEY")
        )
        
        # Gera código
        code = llm.generate_code(self.test_prompt)
        
        # Verifica se o código foi gerado corretamente
        self.assertIsNotNone(code)
        self.assertIn("import pandas", code)
        self.assertIn("execute_sql_query", code)
        self.assertIn("result", code)
    
    @unittest.skipIf(not os.environ.get("ANTHROPIC_API_KEY"), "API key not available")
    def test_anthropic_integration_real(self):
        """Testa a integração real com Anthropic (requer API key)"""
        # Este teste só é executado se a API key da Anthropic estiver disponível
        llm = LLMIntegration(
            model_type=ModelType.ANTHROPIC,
            model_name="claude-3-opus-20240229",
            api_key=os.environ.get("ANTHROPIC_API_KEY")
        )
        
        # Gera código
        code = llm.generate_code(self.test_prompt)
        
        # Verifica se o código foi gerado corretamente
        self.assertIsNotNone(code)
        self.assertIn("import pandas", code)
        self.assertIn("execute_sql_query", code)
        self.assertIn("result", code)


if __name__ == '__main__':
    unittest.main()