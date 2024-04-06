This algorithm had been developed to demonstrate the data management, labeling, organisation, KPI identification and which AI training model to choose depending on the situation. And for this task, our dataset will be `clients.csv`.

The libraries used are `SciKit-Learn, Pandas, IPython and Matplotlib`. 

The first and the following part of the code is the introduction for the data analysis. The first step is to import the libraries that are going to be used, and then, check for the data that is going to be managed.


<code>import pandas as pd
from IPython.display import display
tabela = pd.read_csv("clientes.csv")
display(tabela)
print(tabela.info())
print(tabela.columns)
</code>

And the output must be:![Capturar1](https://github.com/iagoPB/PythonDataAnalisys/assets/50721035/c6da5eae-847a-4274-bae6-88f4b7ec48bb)
As can be seen, the values must be labeled with numerical characters, otherwise the AI's models are not going to be able to work with String values for making the predictions and classifications. Because of that, the Label Encoder from SkLearn is going to be used, what can be checked in the cell below:

<code>from sklearn.preprocessing import LabelEncoder
codificador = LabelEncoder() 
for coluna in tabela.columns:
    if tabela[coluna].dtype == "object" and coluna != "score_credito":
        tabela[coluna] = codificador.fit_transform(tabela[coluna])
for coluna in tabela.columns:
    if tabela[coluna].dtype == "object" and coluna != "score_credito":
        print(coluna)
display(tabela)</code>

The `display (tabela)` will output the following values:
![Capturar2](https://github.com/iagoPB/PythonDataAnalisys/assets/50721035/45efa430-15ec-4ccd-9cc3-7b73f7012fab)

Now it's possible to realize that all String values were labeled as numbers. And more important than that, the decision structure in the code cell above was used for every preventing the column "score_credito" to be affected by the Label Encoder, because that's the column which values will be predicted.

Finished the data preparations and the "cleaning", we must organize it for the AI's models. The first step will be droping irrelevant columns for the predictions.

<code>x= tabela.drop(["score_credito", "id_cliente"], axis=1)
y = tabela["score_credito"]</code>

The choice of which column to drop was not random. We must consider what kind of data will be useful for predicting the value we want to. Considering that, the wisest choice was to drop 'id_cliente', because it's a value that will not cause any impact into the analysis and 'score_credito', because it's the value that we are going to predict.

The next is step is to split the data. But why? Well, data spliting is a secure method that was developed for AI training sets for making it hard for the model to overfit the data, or even worse, underfit it. So 30% of the data is reserved for training, preventing the model from accessing this data, what will result in more reliable results.

<code>from sklearn.model_selection import train_test_split
x_treino, x_teste, y_treino, y_teste = train_test_split(x,y, test_size=0.3, random_state=1)
</code>

With the data split, the choice of a model for prediction is very important, but to judge which one is better, the accuracy of each must be compared, and that's what is going to be done. The chosen models for training are `K-Neighbors` and `Random Forest`.

<code>modelo_arvore= RandomForestClassifier()
modelo_kvizinhos = KNeighborsClassifier()
modelo_arvore.fit(x_treino,y_treino)
modelo_kvizinhos.fit(x_treino, y_treino)
</code>

And for even more comparison, the code below is going to check the accuracy of the model if it guessed every "score_credito" as "Standard" -> Possible values: Poor, Standard, Good.

<code>contagem_scores = tabela["score_credito"].value_counts()
print("Testando a acurácia do modelo chutando tudo Standard(as divisões são: Poor, Standard, Good):")
standard = contagem_scores['Standard']/sum(contagem_scores)
print(standard)
</code>

![Capturar3](https://github.com/iagoPB/PythonDataAnalisys/assets/50721035/52ccbe7f-8be4-4462-8522-1214c8639f3e)

The accuracy was near 54%, pretty low as expected. This value alone means nothing, but when put alongside the other accuracy tests, will show us the how important is to chose the right model for each situation in data analysis.

Now, the measurement of accuracy is going to be applied in the models trained and declared before:


<code>from sklearn.metrics import accuracy_score
pred_arvore = modelo_arvore.predict(x_teste)
pred_kvizinhos = modelo_kvizinhos.predict(x_teste.to_numpy())
print("Acurácia do Modelo de Previsão de Floresta Aleatória:")
print(accuracy_score(y_teste,pred_arvore))
print("Acurácia do Modelo de Previsão de K-vizinhos:")
print(accuracy_score(y_teste,pred_kvizinhos))
</code>

The output:

![Capturar4](https://github.com/iagoPB/PythonDataAnalisys/assets/50721035/e39747da-31cf-450c-82e6-4c6daccccaaf)

And for comparison, a graph is plotted to show the great differences among each one of the accuracy tests:

![Capturar5](https://github.com/iagoPB/PythonDataAnalisys/assets/50721035/be796451-8c6f-43ef-8dbe-f78d8ffdea86)

Therefore, it's possible to conclude that among all the models, the Random Forest performed better in this analysis, so it's going to be the one chosen. Even if the percentage may seems to be very close to each other(K-Neighbors vs Random Forest), a difference of 10% from both models means a lot when dealing with accuracy.

And for the last part, the identification of the KPI values can be done through the presentation of the most important values (the ones with more weight) for the prediction of the "score_credito".

<code>colunas = list(x_teste.columns)
importancia = pd.DataFrame(index=colunas, data = modelo_arvore.feature_importances_)
importancia = importancia * 100
print(importancia)
</code>

With the following output:


![Capturar6](https://github.com/iagoPB/PythonDataAnalisys/assets/50721035/852d46cb-0b3c-4746-a728-95dc1661664a)

With a cautious analysis, it's possible to conclude that the values of 'divida_total, mix_credito, juros_emprestimo' will impact way more than any other value when the prediction of 'score_credito' is done. Considering a hypothetical enterprise, with this informations in hands, it would be possible for it to reduce costs, develop new criteria for classifying it's clients credit score or notify their clients when their score is going low,etc.
This algorithm showed that the data analysis has enormous power into enterprises, making it possible to recognize weak points and bringing to evidence valuable data that can can guide wise decision-making.



