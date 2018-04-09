#instalar pip install sklearn  SciPy
#implementação do knn com sklearn
# http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

from sklearn.neighbors import KNeighborsClassifier
entradas, saidas = [] , []

with open('haberman.data', 'r') as f:
    for linha in f.readlines():
        atrib = linha.replace('\n','').split(',')
        entradas.append([int(atrib[0]),int(atrib[2])])
        saidas.append(int(atrib[3]))
        
p = 0.6 #porcentagem 

limite  = int(p*len(entradas))
knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(entradas[:limite],saidas[:limite])
labels = knn.predict(entradas[limite:])

acertos, indice_label = 0,0

for i in range(limite,len(entradas)):
    if labels[indice_label] == saidas[i]:
        acertos +=1
    indice_label +=1

print("total treinamento %d" % limite)
print("total testes %d" % (len(entradas)- limite))
print("total acertos %d" % acertos)
print("Porcentagem  acertos %.2f" % (100*acertos / (len(entradas)-limite))  )

        

