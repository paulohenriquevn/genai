from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import os
import time
import logging
from enum import Enum

logger = logging.getLogger("GenBI.LLM")

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

class LLMClient(ABC):
    """Interface base para todos os clientes de LLM"""
    
    @abstractmethod
    def complete(self, messages: List[LLMMessage], 
                temperature: float = 0.0,
                max_tokens: Optional[int] = None) -> LLMResponse:
        """
        Envia mensagens para o LLM e recebe uma resposta
        
        Args:
            messages: Lista de mensagens para enviar ao LLM
            temperature: Temperatura para geração (0.0 = determinístico)
            max_tokens: Número máximo de tokens na resposta
            
        Returns:
            LLMResponse: Resposta do LLM
        """
        pass

class OpenAIClient(LLMClient):
    """Cliente para a API da OpenAI"""
    
    def __init__(self, api_key: Optional[str] = None, 
                model: str = "gpt-4o",
                base_url: Optional[str] = None):
        """
        Inicializa o cliente OpenAI
        
        Args:
            api_key: Chave de API OpenAI (ou usa OPENAI_API_KEY do ambiente)
            model: Nome do modelo a usar
            base_url: URL base para API (opcional, para compatibilidade com compatíveis)
        """
        try:
            import openai
        except ImportError:
            raise ImportError("OpenAI SDK não está instalado. Instale com 'pip install openai'")
        
        # Usar API key do ambiente se não fornecida
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API key da OpenAI não fornecida e não encontrada no ambiente")
        
        self.model = model
        
        # Configurar cliente
        self.client = openai.OpenAI(api_key=self.api_key)
        if base_url:
            self.client.base_url = base_url
    
    def complete(self, messages: List[LLMMessage], 
                temperature: float = 0.0,
                max_tokens: Optional[int] = None) -> LLMResponse:
        """Envia mensagens para a API OpenAI e recebe uma resposta"""
        try:
            # Converter mensagens para formato OpenAI
            openai_messages = [{"role": msg.role, "content": msg.content} for msg in messages]
            
            # Configurar parâmetros da chamada
            params = {
                "model": self.model,
                "messages": openai_messages,
                "temperature": temperature
            }
            
            if max_tokens:
                params["max_tokens"] = max_tokens
                
            # Fazer chamada à API
            start_time = time.time()
            response = self.client.chat.completions.create(**params)
            
            # Obter primeiro item da resposta
            completion = response.choices[0]
            
            # Construir resposta
            return LLMResponse(
                content=completion.message.content,
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                created_at=start_time,
                finish_reason=completion.finish_reason
            )
            
        except Exception as e:
            logger.error(f"Erro ao chamar OpenAI: {str(e)}")
            raise