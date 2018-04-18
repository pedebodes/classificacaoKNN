# prever tamanho de sapato

import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#print(train.head())
#print(test.head())


from sklearn.neighbors import KNeighborsClassifier
cols = ['shoe size','height']
cols2 = ['class']

x_train = train.as_matrix(cols)
y_train = train.as_matrix(cols2)
x_test = test.as_matrix(cols)
y_test = test.as_matrix(cols2)


knn = KNeighborsClassifier(n_neighbors=3,weights='distance')
knn.fit(x_train,y_train.ravel())

output = knn.predict(x_test)

print(knn.predict([x_test[0]]))
print(y_test[0])


# verificar se percentual de acertos
correct = 0.0 

for i in range(len(output)):
    if y_test[i][0] == output[i]:
        correct +=1

# percentual de acertos 99% de acerto
print(correct / len(output))


        
        
