# dataset de diabets

from sklearn import datasets 

diabetes = datasets.load_diabetes()
# print(diabetes.data)

# x = entrada y = saida
x,y = diabetes.data,diabetes.target

from sklearn.model_selection import train_test_split
x_train, x_test,y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=42)
# print(len(x_train)) # amostras para treinamento
# print(len(x_test)) # amostras para tese


from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(n_neighbors=17,p=2,)
knn.fit(x_train,y_train)
outputs = knn.predict(x_test)

print("Saida esperada %f " % y_test[3] ) 
print("Saida predita %f " % outputs[3] ) 


# erro quedratico medio

from sklearn.metrics import mean_squared_error
print('Erro quadratico medio %f' %mean_squared_error(y_test,outputs)) 

# quanto menor o erro quadratico medio melhor o ajuste, para alterar o valor é somente aumentar ou diminuir
# o valor de k (n_neighbors)
