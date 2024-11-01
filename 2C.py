import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

print("Carregando dados:")
dados = pd.read_csv('column_2C_weka.csv')

print("Verificando se existem dados nulos")
print(dados.isnull().sum())

print("Removendo linhas com dados nulos")
dados = dados.dropna()

print("Codificando os rótulos da coluna")
label_encoder_2C = LabelEncoder()
label_encoder_2C.fit(dados['class'])
dados['class'] = label_encoder_2C.transform(dados['class'])

print("Separando os dados em treino e teste")
dados_treino_X, dados_teste_X, dados_treino_Y, dados_teste_Y = train_test_split(
    dados.drop('class', axis=1), dados['class'], test_size=0.2, random_state=42)

print("Treinando Árvore de Decisão")
arvore_decisao = DecisionTreeClassifier(random_state=42)
arvore_decisao.fit(dados_treino_X, dados_treino_Y)

print("Treinando Classificador Bayesiano")
classif_bayesiano = GaussianNB()
classif_bayesiano.fit(dados_treino_X, dados_treino_Y)

print("Treinando Support Vector Machine")
svm = SVC(kernel='linear', random_state=42)
svm.fit(dados_treino_X, dados_treino_Y)

print("Verificando acuracia dos algoritmos e imprimindo")
dt_pred = arvore_decisao.predict(dados_teste_X)
pred_nb = classif_bayesiano.predict(dados_teste_X)
pred_svm = svm.predict(dados_teste_X)

print("Acurácia Árvore de Decisão (2C):", accuracy_score(dados_teste_Y, dt_pred))
print("Acurácia Naive Bayes (2C):", accuracy_score(dados_teste_Y, pred_nb))
print("Acurácia SVM (2C):", accuracy_score(dados_teste_Y, pred_svm))

print("Configurando matrizes de confusão")
matriz_dt = confusion_matrix(dados_teste_Y, dt_pred)
matriz_nb = confusion_matrix(dados_teste_Y, pred_nb)
matriz_svm = confusion_matrix(dados_teste_Y, pred_svm)

print("Plotando matrizes de confusão")
classes = ['Normal', 'Anormal'] 
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
sns.heatmap(matriz_dt, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title('Árvore de Decisão')
plt.xlabel('Previsões')
plt.ylabel('Valores Reais')

plt.subplot(1, 3, 2)
sns.heatmap(matriz_nb, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title('Naive Bayes')
plt.xlabel('Previsões')
plt.ylabel('Valores Reais')

plt.subplot(1, 3, 3)
sns.heatmap(matriz_svm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title('SVM')
plt.xlabel('Previsões')
plt.ylabel('Valores Reais')

plt.tight_layout()
plt.show()
