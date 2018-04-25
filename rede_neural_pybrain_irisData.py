import numpy as np

# carregando os dados
entradas = np.genfromtxt('iris.data', delimiter=',', usecols=(0, 1, 2, 3))
saidas = np.genfromtxt('iris.data', delimiter=',', usecols=(4))

# Tamanho das entradas e saidas
# print(len(entradas))
# print(len(saidas))

'''
 iris-setosa = 0
 iris-versicolor = 1
 iris-virginica = 2
'''

# Imprimindo os 5 primeiros amasotras são iris setosa
# print(entradas[:5])
# print(saidas[:5])


'''
50 primeiros registro setosa
50 - 100  registro versicolor
50 ultimas registro virginica
'''

# utilizando 105 amostras para treino  sendo 35 de cada  70% do dataset
# entradas_treino = np.concatenate((setosa, virginica, versicolor))
entradas_treino = np.concatenate((entradas[:35], entradas[50:85], entradas[100:135]))
saidas_treino = np.concatenate((saidas[:35], saidas[50:85], saidas[100:135]))

# restante do dataset usado para treino
entradas_teste = np.concatenate((entradas[35:50], entradas[85:100], entradas[135:]))
saidas_teste = np.concatenate((saidas[35:50], saidas[85:100], saidas[135:]))

#utilizando pybrain
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised import BackpropTrainer
# $adicionar as entradas
treinamento = SupervisedDataSet(4,1)

for i in range(len(entradas_treino)):
    treinamento.addSample(entradas_treino[i],saidas_treino[i])

print(len(treinamento)) #informação do dataset de treino, quantidade
print(treinamento.indim) #dimensão do dataset
print(treinamento.outdim)


# contruindo a rede
rede = buildNetwork(treinamento.indim,2,treinamento.outdim,bias=True)
trainer = BackpropTrainer(rede,treinamento,learningrate=0.01,momentum=0.7)# alterar o momentun e taxa de aprendizagem para ficar mais proximos do acerto


#treinando a rede
# for epoca in range(1000):
#     trainer.train()
trainer.trainEpochs(1000)

#testando a rede
teste = SupervisedDataSet(4,1)

for i in range(len(entradas_teste)):
    teste.addSample(entradas_teste[i],saidas_teste[i])

trainer.testOnData(teste,verbose=True)