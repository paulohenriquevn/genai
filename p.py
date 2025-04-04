import os
import pandasai as pai
from pandasai.helpers.env import load_dotenv

# Carregar variáveis de ambiente (opcional)
load_dotenv()

# Configurar a chave de API (obtenha sua chave em app.pandabi.ai)
# Você pode configurar via variável de ambiente ou diretamente no código
# os.environ["PANDABI_API_KEY"] = "sua-chave-api"
# ou
# pai.api_key.set("sua-chave-api")

# Carregar o dataset de doenças cardíacas
df = pai.read_csv("vendas_perdidas.csv")

print("Dataset carregado com sucesso!")
print(f"Dimensões: {df.shape[0]} linhas x {df.shape[1]} colunas")
print("\nPrimeiras 5 linhas:")
print(df.head())

# Vamos fazer algumas perguntas ao dataset

# Análise estatística básica
print("\n\n--------- ANÁLISE ESTATÍSTICA ---------")
response1 = df.chat("Quais são as estatísticas básicas de idade e colesterol?")
print(response1)

# Análise de correlação
print("\n\n--------- ANÁLISE DE CORRELAÇÃO ---------")
response2 = df.chat("Qual a correlação entre idade e colesterol? Explique e mostre um gráfico.")
print(response2)
# Para visualizar o gráfico, você pode usar:
# response2.show()

# Análise de distribuição
print("\n\n--------- ANÁLISE DE DISTRIBUIÇÃO ---------")
response3 = df.chat("Mostre a distribuição da idade dos pacientes em um histograma.")
print(response3)
# Para visualizar o gráfico, você pode usar:
# response3.show()

# Análise comparativa
print("\n\n--------- ANÁLISE COMPARATIVA ---------")
response4 = df.chat("Compare a porcentagem de doenças cardíacas entre homens e mulheres com um gráfico de barras.")
print(response4)
# Para visualizar o gráfico, você pode usar:
# response4.show()

# Análise preditiva
print("\n\n--------- FATORES DE RISCO ---------")
response5 = df.chat("Quais são os principais fatores associados a doenças cardíacas neste dataset?")
print(response5)

# Pergunta de acompanhamento (follow-up)
print("\n\n--------- PERGUNTA DE ACOMPANHAMENTO ---------")
response6 = df.follow_up("Pode mostrar esses fatores em um gráfico?")
print(response6)
# Para visualizar o gráfico, você pode usar:
# response6.show()

# DICA: Para executar este código e visualizar os gráficos,
# descomente as linhas response.show() após cada resposta que gera um gráfico