# Fluxo Alternativo para Falhas da LLM

Este documento detalha a implementação do fluxo alternativo para lidar com falhas na geração ou execução de consultas no Sistema de Consulta em Linguagem Natural.

## Visão Geral

Um dos principais desafios em sistemas baseados em LLM é lidar com falhas, que podem ocorrer por diversos motivos:
- Consultas sobre entidades inexistentes nos dados
- Erros na geração de código pela LLM
- Falhas na execução do código gerado
- Limitações de compreensão ou conhecimento do modelo

Para mitigar esses problemas, implementamos um fluxo alternativo robusto que detecta falhas, tenta corrigi-las automaticamente, coleta feedback do usuário e oferece sugestões alternativas quando necessário.

## Componentes do Fluxo Alternativo

### 1. Detecção de Entidades Inexistentes

**Implementação**: `_create_missing_entity_response()` e verificação em `process_query()`

```python
# Verifica menções a dados inexistentes antes de chamar a LLM
for entity_type, keywords in missing_entity_keywords.items():
    if any(keyword in query.lower() for keyword in keywords) and not any(entity_type in ds.name.lower() for ds in self.datasets.values()):
        alternative_queries = self._generate_alternative_queries()
        datasets_desc = ", ".join([f"{name}" for name, _ in self.datasets.items()])
        
        return self._create_missing_entity_response(
            entity_type, 
            datasets_desc, 
            alternative_queries
        )
```

**Funcionalidade**:
- Verifica se a consulta menciona entidades como "produtos", "funcionários", etc. que não existem nos dados disponíveis
- Evita chamadas desnecessárias à LLM para consultas que não podem ser satisfeitas
- Retorna resposta amigável explicando o problema e sugerindo alternativas

### 2. Reformulação Automática de Consultas

**Implementação**: `_rephrase_query()` e `_simplify_query()`

```python
# Quando ocorre um erro, tenta reformular a consulta
rephrased_query = self._rephrase_query(query, error_msg)
logger.info(f"Consulta reformulada: {rephrased_query}")
                        
# Reinicia o processamento com a consulta reformulada
return self.process_query(rephrased_query, retry_count + 1, max_retries)
```

**Funcionalidade**:
- Usa a LLM para reformular a consulta original com base no erro encontrado
- Considera as estruturas de dados disponíveis
- Implementa uma estratégia de simplificação progressiva para casos difíceis
- Limita o número de tentativas para evitar loops infinitos

### 3. Coleta e Uso de Feedback do Usuário

**Implementação**: `process_query_with_feedback()` e `_store_user_feedback()`

```python
def process_query_with_feedback(self, query: str, feedback: str = None) -> BaseResponse:
    """
    Processa uma consulta e inclui feedback do usuário quando disponível.
    """
    return self.process_query(query, feedback=feedback)

def _store_user_feedback(self, query: str, feedback: str) -> None:
    """
    Armazena feedback do usuário para melhorias futuras.
    """
    # Cria o diretório de feedback se não existir
    feedback_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_feedback")
    os.makedirs(feedback_dir, exist_ok=True)
    
    # Armazena em um arquivo JSON
    feedback_file = os.path.join(feedback_dir, "user_feedback.json")
    
    # Carrega o feedback existente e adiciona o novo
    existing_feedback = []
    if os.path.exists(feedback_file):
        with open(feedback_file, 'r', encoding='utf-8') as f:
            existing_feedback = json.load(f)
    
    existing_feedback.append({
        "timestamp": time.time(),
        "query": query,
        "feedback": feedback
    })
    
    # Salva o feedback atualizado
    with open(feedback_file, 'w', encoding='utf-8') as f:
        json.dump(existing_feedback, f, indent=2, ensure_ascii=False)
```

**Funcionalidade**:
- Permite que o usuário forneça feedback para melhorar a resposta
- Armazena o feedback em um arquivo JSON para análise futura
- Usa o feedback para enriquecer o prompt enviado à LLM
- Permite aprendizado contínuo com base nas interações do usuário

### 4. Sugestões Predefinidas

**Implementação**: `_generate_alternative_queries()` e `_offer_predefined_options()`

```python
def _generate_alternative_queries(self) -> List[str]:
    """
    Gera consultas alternativas baseadas nos datasets disponíveis.
    """
    alternatives = []
    
    # Consultas básicas para cada dataset
    for name in self.datasets.keys():
        alternatives.append(f"Mostre um resumo do dataset {name}")
        alternatives.append(f"Quais são as principais informações em {name}?")
    
    # Consultas mais específicas baseadas nos metadados dos datasets
    for name, dataset in self.datasets.items():
        # Consultas baseadas em tipos de colunas, relacionamentos, etc.
        # Código detalhado na implementação
    
    # Remove duplicatas e limita a 10 alternativas
    unique_alternatives = list(set(alternatives))
    return unique_alternatives[:10]
```

**Funcionalidade**:
- Gera sugestões de consultas alternativas baseadas nos datasets disponíveis
- Considera metadados como tipos de colunas, relacionamentos entre tabelas, etc.
- Oferece opções para explorar os dados quando a consulta original falha
- Ajuda o usuário a reformular sua consulta de forma mais eficaz

## Fluxo de Processamento

1. **Verificação Inicial**: Verifica se a consulta menciona entidades inexistentes
2. **Tentativa Normal**: Processa a consulta normalmente
3. **Detecção de Erro**: Em caso de falha, identifica o tipo de erro
4. **Reformulação**: Tenta reformular a consulta (até `max_retries` vezes)
5. **Feedback**: Se disponível, utiliza feedback do usuário
6. **Alternativas**: Após esgotar as tentativas, oferece sugestões predefinidas

## Diagrama de Fluxo

```
┌─────────────────┐
│ Recebe Consulta │
└────────┬────────┘
         ▼
┌────────────────────┐     Sim    ┌─────────────────────────┐
│Menciona Entidades  ├───────────►│ Resposta com Alternativas│
│   Inexistentes?    │            └─────────────────────────┘
└────────┬───────────┘
         │ Não
         ▼
┌────────────────────┐
│ Processa Consulta  │
└────────┬───────────┘
         │
         ▼
┌────────────────────┐     Sim    ┌─────────────────────────┐
│     Falhou?        ├───────────►│ Tentativa < max_retries? │
└────────┬───────────┘            └──────────┬──────────────┘
         │ Não                              │
         │                      Sim         │ Não
         │               ┌──────────────────┘     ┌───────────────────┐
         │               │                        │ Oferece Sugestões │
         │               ▼                        │   Predefinidas    │
         │      ┌────────────────┐                └───────────────────┘
         │      │   Reformula    │
         │      │   Consulta     │
         │      └────────────────┘
         │
         ▼
┌────────────────────┐
│ Retorna Resultado  │
└────────────────────┘
```

## Melhorias Futuras

1. **Análise de Feedback**: Implementar análise automática do feedback coletado para melhorar o sistema
2. **Aprendizado de Reformulações**: Armazenar reformulações bem-sucedidas para uso futuro
3. **Personalização por Usuário**: Adaptar sugestões com base no histórico de consultas do usuário
4. **Feedback Visual**: Melhorar a interface para coletar feedback de forma mais intuitiva
5. **Explicabilidade**: Oferecer explicações sobre por que a consulta original falhou

## Uso e Testes

Para testar o fluxo alternativo, execute o script de demonstração:

```bash
python fallback_flow_example.py
```

Este script demonstra todos os componentes do fluxo alternativo:
- Detecção de entidades inexistentes
- Reformulação automática
- Coleta de feedback
- Sugestões predefinidas

Também inclui um modo interativo que permite testar o sistema com suas próprias consultas e fornecer feedback.

## Conclusão

O fluxo alternativo para falhas da LLM torna o sistema mais robusto e amigável, permitindo que os usuários obtenham resultados úteis mesmo quando a consulta original não pode ser processada. A combinação de detecção precoce, reformulação automática, feedback do usuário e sugestões predefinidas cria uma experiência mais eficaz e satisfatória.