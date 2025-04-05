import os
import shutil
from fastapi import UploadFile
import pandas as pd
from typing import Optional, List, Dict, Any

class FileManager:
    """
    Gerencia o armazenamento e acesso a arquivos carregados.
    """
    
    def __init__(self, base_dir: str = "uploads"):
        """
        Inicializa o gerenciador de arquivos.
        
        Args:
            base_dir: Diretório base para armazenamento
        """
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
    
    async def save_file(self, file: UploadFile, file_id: str) -> str:
        """
        Salva um arquivo carregado.
        
        Args:
            file: Arquivo carregado
            file_id: Identificador único para o arquivo
            
        Returns:
            Caminho completo do arquivo salvo
        """
        # Cria diretório específico para o arquivo
        file_dir = os.path.join(self.base_dir, file_id)
        os.makedirs(file_dir, exist_ok=True)
        
        # Define o caminho do arquivo
        file_path = os.path.join(file_dir, file.filename)
        
        # Salva o arquivo
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Retorna o caminho completo
        return file_path
    
    async def delete_file(self, file_id: str) -> bool:
        """
        Remove um arquivo e seu diretório.
        
        Args:
            file_id: Identificador do arquivo
            
        Returns:
            True se removido com sucesso, False caso contrário
        """
        file_dir = os.path.join(self.base_dir, file_id)
        
        if os.path.exists(file_dir):
            shutil.rmtree(file_dir)
            return True
        return False
    
    def get_file_path(self, file_id: str, filename: Optional[str] = None) -> str:
        """
        Obtém o caminho completo para um arquivo.
        
        Args:
            file_id: Identificador do arquivo
            filename: Nome do arquivo (opcional)
            
        Returns:
            Caminho completo do arquivo
        """
        file_dir = os.path.join(self.base_dir, file_id)
        
        if filename:
            return os.path.join(file_dir, filename)
        
        # Se o nome do arquivo não for fornecido, retorna o primeiro arquivo no diretório
        if os.path.exists(file_dir):
            files = os.listdir(file_dir)
            if files:
                return os.path.join(file_dir, files[0])
        
        return ""
    
    def list_files(self, file_id: str) -> List[str]:
        """
        Lista todos os arquivos em um diretório de ID.
        
        Args:
            file_id: Identificador do diretório
            
        Returns:
            Lista de nomes de arquivos
        """
        file_dir = os.path.join(self.base_dir, file_id)
        
        if os.path.exists(file_dir):
            return os.listdir(file_dir)
        return []