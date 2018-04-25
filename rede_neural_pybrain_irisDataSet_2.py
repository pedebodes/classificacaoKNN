#carregando dados do iris dataset com sklearn
from sklearn import datasets
from pybrain.datasets.classification import ClassificationDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised import BackpropTrainer

iris = datasets.load_iris()
#obtendo entradas e saidas
x, y = iris.data, iris.target

dataset = ClassificationDataSet(4, 1, nb_classes=3)# nb_classes=3 (setosa,virginica,versicolor)
#adicionando amostras
for i in range(len(x)):
    dataset.addSample(x[i], y[i])

#particionando os dados para treinamento
train_data_temp, part_data_temp = dataset.splitWithProportion(0.6)
teste_data_temp, val_data_temp = part_data_temp.splitWithProportion(0.5)

print(type(train_data_temp)) 

#converter para o tipo para classificationDatase
train_data = ClassificationDataSet(4,1,nb_classes=3)
for i in range(train_data_temp.getLength()):
    train_data.addSample(train_data_temp.getSample(i)[0], train_data_temp.getSample(i)[1])
print(type(train_data))

#repetir o mesmo para os demais
teste_data = ClassificationDataSet(4, 1, nb_classes=3)
for i in range(teste_data_temp.getLength()):
    teste_data.addSample(teste_data_temp.getSample(i)[0], teste_data_temp.getSample(i)[1])

val_data = ClassificationDataSet(4, 1, nb_classes=3)
for i in range(val_data_temp.getLength()):
    teste_data.addSample(val_data_temp.getSample(i)[0], val_data_temp.getSample(i)[1])

print('==============================================')
#==============================================
#altera a saida de 0 ou 1 para (001, 100, 010 ,etc)
print("Antes de converter _convertToOneOfMany .... ")
print(train_data['target'][:5])

val_data._convertToOneOfMany()
train_data._convertToOneOfMany()
teste_data._convertToOneOfMany()


print("Depois de converter _convertToOneOfMany .... ")
print(train_data['target'][:5])
#==============================================
print('==============================================')


#criando a rede
from pybrain.structure.modules import SoftmaxLayer

rede = buildNetwork(train_data.indim, 5, train_data.outdim, outclass=SoftmaxLayer)
trainer = BackpropTrainer(rede,dataset=train_data,learningrate=0.01,momentum=0.1)
#treinando
trainer.trainOnDataset(train_data,100)


#taxa de erro
from pybrain.utilities import percentError
out = rede.activateOnDataset(teste_data).argmax(axis=1)
print("Erro de teste %d " % percentError(out,teste_data['class']))

import numpy
#verificar erro de validação
out = rede.activateOnDataset(val_data).argmax(axis=1)
print('Erro de validação: %f' % percentError(out, val_data['class']))


print("validação")
print("Saida da rede \t",out)
print("Correto rede \t",val_data['class'][:,0])

