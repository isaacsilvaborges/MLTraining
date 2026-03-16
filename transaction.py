import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score

# ==========================================
# PARTE 01: CRIAÇÃO DO DATASET
# ==========================================

transacoes_suspeitas = [
    "500 > 1s > 7000 > 2s > 15000",
    "300 > 0s > 5000 > 1s > 12000",
    "1000 > 2s > 2000 > 1s > 8000",
    "150 > 1s > 9000 > 1s > 20000",
    "400 > 0s > 6000 > 2s > 18000"
] * 20

transacoes_normais = [
    "120 > 80s > 90 > 67s > 150",
    "300 > 120s > 250 > 95s > 400",
    "500 > 200s > 450 > 180s > 600",
    "100 > 300s > 130 > 250s > 90",
    "200 > 150s > 220 > 170s > 210"
] * 20

dados = []

for t in transacoes_suspeitas:
    dados.append({"texto": t, "label": 1})

for t in transacoes_normais:
    dados.append({"texto": t, "label": 0}) 

df = pd.DataFrame(dados)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print("Tamanho total do dataset:", len(df))
print("\nAmostra:")
print(df.head())


# ==========================================
# PARTE 02: DIVISÃO E VETORIZAÇÃO
# ==========================================

X = df['texto']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer()

X_train_vetorizado = vectorizer.fit_transform(X_train)
X_test_vetorizado = vectorizer.transform(X_test)


# ==========================================
# PARTE 03: TREINAMENTO
# ==========================================

modelo = LogisticRegression(random_state=42)

print("\nTreinando modelo...")
modelo.fit(X_train_vetorizado, y_train)
print("Modelo treinado!\n")

previsoes = modelo.predict(X_test_vetorizado)

print("=== MATRIZ DE CONFUSÃO ===")
print("(Eixo Y = Realidade | Eixo X = Previsão)")
print(confusion_matrix(y_test, previsoes))

print("\n=== RELATÓRIO DE MÉTRICAS ===")
print(classification_report(y_test, previsoes))

acuracia = accuracy_score(y_test, previsoes)
print(f"\nAcurácia geral: {acuracia:.4f}")


# ==========================================
# PARTE 04: MODO INTERATIVO
# ==========================================

print("\n" + "="*50)
print("🤖 MODO INTERATIVO: TESTE O MODELO")
print("Digite uma sequência no formato:")
print("500 > 1s > 7000 > 2s > 15000")
print("Digite 'sair' para encerrar.")
print("="*50 + "\n")

while True:
    
    entrada = input("Descreva a sequência: ")
    
    if entrada.lower() in ['sair', 'exit']:
        print("Encerrando testes.")
        break
        
    if not entrada.strip():
        continue
        
    entrada_vetorizada = vectorizer.transform([entrada])
    
    previsao = modelo.predict(entrada_vetorizada)[0]
    probabilidades = modelo.predict_proba(entrada_vetorizada)[0]
    certeza = probabilidades[previsao] * 100
    
    print("\n--- RESULTADO ---")
    
    if previsao == 1:
        print("🚨 ALERTA: Transação SUSPEITA detectada!")
    else:
        print("✅ OK: Transação NORMAL.")
        
    print(f"Certeza do modelo: {certeza:.2f}%")
    print("-" * 50 + "\n")