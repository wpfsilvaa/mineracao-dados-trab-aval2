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

dados = pd.read_csv('column_3C_weka.csv')
print("Dados carregados")

print("Verificando se existem dados nulos:")
print(dados.isnull().sum())

print("Removendo linhas com dados nulos")
dados = dados.dropna()

print("Codificando os rótulos da coluna")
label_encoder = LabelEncoder()
label_encoder.fit(dados['class'])
dados['class'] = label_encoder.transform(dados['class'])

print("Separando os dados em treino e teste")
dados_treino_X, dados_teste_X, dados_treino_y, dados_teste_y = train_test_split(
    dados.drop('class', axis=1), dados['class'], test_size=0.2, random_state=42)

print("Treinando Árvore de Decisão")
arvore_decisao = DecisionTreeClassifier(random_state=42)
arvore_decisao.fit(dados_treino_X, dados_treino_y)

print("Treinando Classificador Bayesiano")
classificador_beyesiano = GaussianNB()
classificador_beyesiano.fit(dados_treino_X, dados_treino_y)

print("Treinando Support Vector Machine")
svm = SVC(kernel='linear', random_state=42)
svm.fit(dados_treino_X, dados_treino_y)

print("Verificando acuracia dos algoritmos e imprimindo")
arvore_decisao_predict = arvore_decisao.predict(dados_teste_X)
classificador_beyesiano_predict = classificador_beyesiano.predict(dados_teste_X)
svm_predict = svm.predict(dados_teste_X)

print("Acurácia Árvore de Decisão:", accuracy_score(dados_teste_y, arvore_decisao_predict))
print("Acurácia Naive Bayes:", accuracy_score(dados_teste_y, classificador_beyesiano_predict))
print("Acurácia SVM:", accuracy_score(dados_teste_y, svm_predict))

print("Configurando matrizes de confusão")
matriz_dt = confusion_matrix(dados_teste_y, arvore_decisao_predict)
matriz_nb = confusion_matrix(dados_teste_y, classificador_beyesiano_predict)
matriz_svm = confusion_matrix(dados_teste_y, svm_predict)

print("Plotando matrizes de confusão")
classes_3C = ['Normal', 'Hernia', 'Espondilolistese'] 
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
sns.heatmap(matriz_dt, annot=True, fmt='d', cmap='Blues', xticklabels=classes_3C, yticklabels=classes_3C)
plt.title('Árvore de Decisão')
plt.xlabel('Previsões')
plt.ylabel('Valores Reais')

plt.subplot(1, 3, 2)
sns.heatmap(matriz_nb, annot=True, fmt='d', cmap='Blues', xticklabels=classes_3C, yticklabels=classes_3C)
plt.title('Naive Bayes')
plt.xlabel('Previsões')
plt.ylabel('Valores Reais')

plt.subplot(1, 3, 3)
sns.heatmap(matriz_svm, annot=True, fmt='d', cmap='Blues', xticklabels=classes_3C, yticklabels=classes_3C)
plt.title('SVM')
plt.xlabel('Previsões')
plt.ylabel('Valores Reais')

plt.tight_layout()
plt.show()
