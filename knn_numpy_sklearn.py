'''
Datase
https://archive.ics.uci.edu/ml/datasets/Balance+Scale


Attribute Information:

1. Class Name: 3 (L, B, R)  # alterado para 1 2 3 respectivamente para facilitar o calculo
2. Left-Weight: 5 (1, 2, 3, 4, 5) 
3. Left-Distance: 5 (1, 2, 3, 4, 5) 
4. Right-Weight: 5 (1, 2, 3, 4, 5) 
5. Right-Distance: 5 (1, 2, 3, 4, 5)

'''


import numpy as np

# x entrada     Y saidas
x = np.genfromtxt('balance-scale.data', delimiter=',',usecols=(1,2,3,4))
y = np.genfromtxt('balance-scale.data',delimiter=',',usecols=(0))

from sklearn.model_selection import train_test_split
# from sklearn.cross_validation import train_test_split
# documentação http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

x_treino , x_teste, Y_treino, Y_teste = train_test_split(x,y,test_size=0.3,random_state = 42)


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=17,p=2)
knn.fit(x_treino,Y_treino)
labels = knn.predict(x_teste)

# print(len(labels))
print(np.sum(labels == Y_teste)) #Acertos

print(100*(labels == Y_teste).sum() / len(x_teste)) # percentual de acertos