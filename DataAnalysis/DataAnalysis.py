import pandas as pd
from IPython.display import display

tabela = pd.read_csv("clientes.csv")
display(tabela)

print(tabela.info())
print(tabela.columns)

from sklearn.preprocessing import LabelEncoder

codificador = LabelEncoder() #transformar os textos das tabela em numeros a fim de permitir a classifiação e previsão, com excecão de score credito

for coluna in tabela.columns:
    if tabela[coluna].dtype == "object" and coluna != "score_credito":
        tabela[coluna] = codificador.fit_transform(tabela[coluna])

for coluna in tabela.columns:
    if tabela[coluna].dtype == "object" and coluna != "score_credito":
        print(coluna)


display(tabela)

#seleçao de colunas que utilizaremos para a previsão de score crédito
x= tabela.drop(["score_credito", "id_cliente"], axis=1)
y = tabela["score_credito"]

from sklearn.model_selection import train_test_split


x_treino, x_teste, y_treino, y_teste = train_test_split(x,y, test_size=0.3, random_state=1)

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier 

modelo_arvore= RandomForestClassifier()
modelo_kvizinhos = KNeighborsClassifier()
#treino de modelos
modelo_arvore.fit(x_treino,y_treino)
modelo_kvizinhos.fit(x_treino, y_treino)

#modelo chutando tudo standard
contagem_scores = tabela["score_credito"].value_counts()
print("Testando a acurácia do modelo chutando tudo Standard(as divisões são: Poor, Standard, Good):")
print(contagem_scores['Standard']/sum(contagem_scores))



from sklearn.metrics import accuracy_score
#calculando previsões
pred_arvore = modelo_arvore.predict(x_teste)
pred_kvizinhos = modelo_kvizinhos.predict(x_teste.to_numpy())
#acurácia
print("Acurácia do Modelo de Previsão de Árvores:")
print(accuracy_score(y_teste,pred_arvore))
print("Acurácia do Modelo de Previsão de K-vizinhos:")
print(accuracy_score(y_teste,pred_kvizinhos))

#importância das features 
colunas = list(x_teste.columns)
importancia = pd.DataFrame(index=colunas, data = modelo_arvore.feature_importances_)
importancia = importancia * 100
print(importancia)


