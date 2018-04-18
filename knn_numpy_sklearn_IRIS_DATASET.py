#utilizando iris dataset para classificação
import numpy as np
from sklearn import  neighbors,datasets

iris = datasets.load_iris()

#descrição do dataset
# print(iris.DESCR)

#entrada
x = iris.data
# print (x)

#saida
y = iris.target
# print(y)


knn = neighbors.KNeighborsClassifier(n_neighbors=5)

knn.fit(x,y)  # treinar o modelo
# print(knn.fit(x, y))

knn.predict(x) # predição
# print(knn.predict(x))

# verificando acuracia
acuracia = knn.score(x,y) # 96% de acerto
print(acuracia)


