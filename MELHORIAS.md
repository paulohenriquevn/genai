# Prompts para o Time de Desenvolvimento

## 1. Restruturação de Importações e Resolução de Dependências Circulares

```
Precisamos resolver os problemas de importação circular e referências incorretas no sistema. Específicamente:

1. Corrija todos os caminhos de importação no estilo "from modulo.connector.exceptions" para "from connector.exceptions" para garantir consistência.

2. Resolva as dependências circulares nas seguintes classes:
   - CsvConnector e DataConnectorFactory
   - BaseQueryBuilder e suas implementações

3. Implemente um padrão de injeção de dependência onde apropriado para desacoplar componentes.

4. Revise todos os arquivos __init__.py para definir claramente o que é exportado por cada módulo.

Tempo estimado: 3 dias
Prioridade: Alta (Bloqueador para produção)
```

## 2. Implementação de Testes Unitários e de Integração

```
Nossa cobertura de testes é insuficiente para um sistema desta complexidade. Precisamos:

1. Desenvolver testes unitários para cada componente principal:
   - Conectores de dados (CSV, PostgreSQL, DuckDB)
   - Executor de código seguro
   - Geradores de prompts LLM
   - Construtores de queries SQL
   - Parsers de respostas

2. Implementar testes de integração para cenários completos:
   - Carregamento de dados → Processamento de consulta → Geração de resultado
   - Verificação de erro → Tentativa de correção → Recuperação

3. Configurar um pipeline de CI que execute todos os testes automaticamente e gere relatórios de cobertura.

4. Criar casos de teste específicos para validar limites e situações de erro:
   - Consultas maliciosas
   - Datasets corrompidos
   - Falhas de API de LLM

Objetivo de cobertura: Mínimo de 80% de cobertura de código
Tempo estimado: 2 semanas
Prioridade: Alta
```

## 3. Segurança e Gestão de Credenciais

```
Precisamos melhorar significativamente a segurança do sistema, especialmente:

1. Implementar sistema seguro para gerenciamento de credenciais:
   - Remover todas as chaves de API hardcoded e variáveis de ambiente diretamente acessadas
   - Integrar com um serviço de gerenciamento de segredos (AWS Secrets Manager, HashiCorp Vault, etc.)
   - Desenvolver um método seguro para rotação de credenciais

2. Reforçar a sandboxing de execução de código:
   - Considerar o uso de contenedores isolados para execução de código gerado
   - Implementar limites estritos de recursos (memória, CPU, tempo)
   - Criar listas brancas explícitas de operações permitidas

3. Revisar e reforçar a sanitização de consultas:
   - Adicionar validação robusta para todas as entradas do usuário
   - Implementar mitigações específicas para injeção SQL
   - Criar sistema de detecção de padrões de consulta potencialmente perigosos

4. Configurar trilhas de auditoria abrangentes:
   - Registrar todas as consultas executadas
   - Manter históricos de acesso a dados
   - Implementar alertas para padrões de uso suspeitos

Tempo estimado: 2 semanas
Prioridade: Crítica (Bloqueador para produção)
```

## 4. Otimização para Grandes Datasets

```
O sistema atualmente não está preparado para trabalhar com datasets que excedem a memória disponível. Precisamos:

1. Implementar processamento em streaming:
   - Modificar os conectores para suportar carregamento parcial de dados
   - Adaptar o executor SQL para trabalhar com chunks de dados
   - Garantir que as transformações possam ser aplicadas incrementalmente

2. Adicionar suporte a paginação de resultados:
   - Modificar a API de resposta para incluir paginação
   - Implementar resumo de resultados grandes
   - Permitir que o usuário explore conjuntos de dados maiores em partes

3. Criar sistema inteligente de cache:
   - Armazenar resultados de consultas frequentes
   - Cachear código gerado para consultas similares
   - Implementar invalidação de cache baseada em alterações de dados

4. Adicionar sistema de monitoramento de recursos:
   - Rastrear uso de memória e CPU
   - Implementar timeouts configuráveis
   - Criar mecanismos de terminação segura para consultas excessivamente custosas

Tempo estimado: 3 semanas
Prioridade: Média-Alta
```

## 5. Documentação e Interface do Usuário

```
A documentação atual é insuficiente para desenvolvedores e usuários. Precisamos:

1. Criar documentação abrangente da API:
   - Documentar todas as classes e métodos públicos
   - Gerar especificação OpenAPI/Swagger para endpoints
   - Documentar todos os parâmetros de configuração disponíveis

2. Desenvolver guias de uso:
   - Tutorial passo a passo para novos usuários
   - Exemplos práticos para casos de uso comuns
   - Guia de resolução de problemas

3. Melhorar a interface de linha de comando:
   - Adicionar mais comandos e opções
   - Implementar formatação de saída configurável
   - Criar assistente interativo para consultas

4. Desenvolver interface web básica:
   - Painel para upload e gerenciamento de dados
   - Editor de consultas com histórico
   - Visualização interativa de resultados

Tempo estimado: 3 semanas
Prioridade: Média
```

## 6. Processamento Assíncrono e Sistema de Filas

```
Precisamos implementar processamento assíncrono para consultas longas:

1. Desenvolver sistema de filas para consultas:
   - Integrar com sistema de mensageria (RabbitMQ, SQS, Kafka)
   - Implementar workers para processar consultas da fila
   - Criar sistema de prioridade para consultas

2. Adicionar endpoints assíncronos à API:
   - Permitir solicitações não-bloqueantes
   - Implementar mecanismo de polling para status
   - Criar sistema de callbacks para notificação

3. Desenvolver sistema de persistência de resultados:
   - Armazenar resultados de consultas concluídas
   - Implementar expiração configurável de resultados
   - Permitir compartilhamento de resultados entre usuários

4. Criar sistema de consultas agendadas:
   - Permitir agendamento de consultas recorrentes
   - Implementar sistema de templates de consulta
   - Adicionar alertas baseados em resultados

Tempo estimado: 4 semanas
Prioridade: Média
```

## 7. Expansão de Conectores de Dados

```
Precisamos ampliar o suporte a fontes de dados:

1. Implementar conectores para serviços de armazenamento em nuvem:
   - Amazon S3
   - Google Cloud Storage
   - Azure Blob Storage

2. Adicionar suporte a formatos adicionais:
   - Parquet
   - Avro
   - ORC
   - JSON Lines

3. Expandir conectores de banco de dados:
   - MySQL/MariaDB
   - SQLite
   - MongoDB
   - Elasticsearch

4. Implementar conectores para APIs:
   - Conectores REST genéricos
   - GraphQL
   - APIs específicas (Google Analytics, Salesforce, etc.)

Cada novo conector deve incluir:
- Testes unitários completos
- Documentação de uso
- Exemplos práticos

Tempo estimado: 6 semanas (implementação incremental)
Prioridade: Baixa (podem ser adicionados após lançamento inicial)
```

## 8. Monitoramento e Observabilidade

```
Precisamos implementar um sistema abrangente de monitoramento:

1. Integrar com ferramentas de APM:
   - New Relic ou Datadog
   - Rastreamento de desempenho de consultas
   - Alertas baseados em latência e taxa de erro

2. Desenvolver sistema de métricas:
   - Tempo médio de processamento por tipo de consulta
   - Taxa de sucesso/falha de geração de código
   - Utilização de recursos (memória, CPU)
   - Contadores de uso por tipo de conector e fonte de dados

3. Implementar logs estruturados e centralizados:
   - Formato JSON para todos os logs
   - Níveis de log configuráveis
   - Integração com ELK ou serviço similar

4. Criar dashboards operacionais:
   - Visão geral de saúde do sistema
   - Estatísticas de uso
   - Taxas de erro e latência
   - Uso de recursos

Tempo estimado: 2 semanas
Prioridade: Média-Alta
```

## 9. Ajustes Finais e Preparação para Produção

```
Antes de lançar em produção, precisamos concluir:

1. Auditoria de código completa:
   - Revisão de segurança
   - Verificação de eficiência de algoritmos
   - Validação de práticas de codificação

2. Testes de carga:
   - Simular casos de uso com volumes crescentes
   - Identificar gargalos de desempenho
   - Estabelecer limites operacionais seguros

3. Documentação de implantação:
   - Requisitos de sistema
   - Procedimento de instalação passo a passo
   - Guia de configuração
   - Procedimentos de backup e recuperação

4. Plano de lançamento faseado:
   - Definir conjunto inicial de usuários
   - Estabelecer métricas para monitorar
   - Criar plano de escalabilidade

5. Sistema de feedback e iteração:
   - Implementar mecanismo para coleta de feedback
   - Criar processo para priorizar melhorias
   - Estabelecer ciclo de lançamento de atualizações

Tempo estimado: 2 semanas
Prioridade: Alta (Final)
```