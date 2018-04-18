from sklearn.datasets import load_boston  # dataset de precos de casas de bostom

boston = load_boston()
# print(boston.DESCR)  # descrição do dataset
# print(boston.data.shape) # 506 amostras 13 dimensoes
# print(boston.feature_names) #caractereisticas 

from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

k = 9
knn = KNeighborsRegressor(n_neighbors=k)
knn.fit(boston.data, boston.target)
print(boston.target[0])  # saida do dataset
print(knn.predict([boston.data[0]]))  # predicao

import matplotlib.pyplot as plt
import numpy as np
knn = KNeighborsRegressor(n_neighbors=k)
x, y = boston.data[:50], boston.target[:50]
y_ = knn.fit(x,y).predict(x)


plt.plot(np.linspace(-1,1,50), y, label="data",color="black" )
plt.plot(np.linspace(-1,1,50), y_, label="prediction",color="red" )
plt.legend()
plt.show()
