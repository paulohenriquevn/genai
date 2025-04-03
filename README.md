# GenBI: Sistema Modular de Business Intelligence Generativa

## Visão Geral

GenBI é uma aplicação de Business Intelligence Generativa projetada com foco em processamento de dados e geração de relatórios no backend, sem interface gráfica de usuário.

## Principais Características

- **Processamento de Consultas em Linguagem Natural**: Converte perguntas em linguagem natural para consultas SQL precisas.
- **Geração Automática de Relatórios**: Cria relatórios com visualizações e envia por e-mail.
- **Suporte a Múltiplas Fontes de Dados**: Conecta-se a diferentes bancos de dados.
- **Integração com LLMs**: Utiliza modelos de linguagem para interpretação de consultas.

## Arquitetura

### Componentes Principais

1. **API REST**: Interface para interação com o sistema
2. **Processador NL-to-SQL**: Converte linguagem natural em consultas SQL
3. **Executor de Consultas**: Processa consultas em diferentes fontes de dados
4. **Gerador de Relatórios**: Cria visualizações e envia relatórios por e-mail

## Configuração

### Pré-requisitos

- Python 3.9+
- Bibliotecas listadas em `requirements.txt`

### Passos de Instalação

1. Clone o repositório
2. Crie um ambiente virtual
3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure as variáveis de ambiente:
   - Copie `.env.example` para `.env`
   - Preencha as configurações necessárias

## Uso via API

### Consultas em Linguagem Natural

```python
# Exemplo de chamada para consulta em linguagem natural
POST /query/natural_language
{
    "question": "Quais são os 5 produtos mais vendidos no último mês?",
    "use_cache": true,
    "explain_sql": true
}
```

### Geração de Relatórios

```python
# Exemplo de geração de relatório
POST /reports/generate
{
    "title": "Relatório de Vendas Mensal",
    "description": "Resumo de desempenho de vendas",
    "query_ids": ["query_id_1", "query_id_2"],
    "recipients": ["usuario@empresa.com"],
    "smtp_config": {
        "host": "smtp.gmail.com",
        "port": 587,
        "username": "seu_email@gmail.com"
    }
}
```

## Configurações Importantes

- **Arquivo de Configuração**: `config/config.json`
  - Configurações de fonte de dados
  - Configurações de LLM
  - Configurações de cache

- **Arquivo de Configuração de E-mail**: `config/email_config.json`
  - Configurações de SMTP
  - Remetentes e destinatários padrão

## Segurança

- Use variáveis de ambiente para credenciais sensíveis
- Habilite TLS para conexões SMTP
- Utilize senhas de aplicativo para serviços como Gmail

## Limitações

- Sem interface gráfica de usuário
- Todas as interações via API REST
- Requer conhecimento técnico para configuração

## Contribuição

1. Faça um fork do repositório
2. Crie uma branch para sua feature
3. Commit suas alterações
4. Abra um Pull Request

## Licença

[Inserir detalhes da licença]