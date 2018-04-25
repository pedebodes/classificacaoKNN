from sklearn import datasets
from pybrain.datasets.classification import ClassificationDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer

iris = datasets.load_iris()
X, y = iris.data, iris.target
dataset = ClassificationDataSet(4, 1, nb_classes=3)
for i in range(len(X)):
    dataset.addSample(X[i], y[i])

train_data_temp, part_data_temp = dataset.splitWithProportion(0.6)
test_data_temp, val_data_temp = part_data_temp.splitWithProportion(0.5)

train_data = ClassificationDataSet(4, 1, nb_classes=3)
for n in range(train_data_temp.getLength()):
    train_data.addSample(train_data_temp.getSample(
        n)[0], train_data_temp.getSample(n)[1])

test_data = ClassificationDataSet(4, 1, nb_classes=3)
for n in range(test_data_temp.getLength()):
    test_data.addSample(test_data_temp.getSample(
        n)[0], test_data_temp.getSample(n)[1])

val_data = ClassificationDataSet(4, 1, nb_classes=3)
for n in range(val_data_temp.getLength()):
    val_data.addSample(val_data_temp.getSample(
        n)[0], val_data_temp.getSample(n)[1])

print('Antes do _convertToOneOfMany...')
print(train_data['target'][:5])

train_data._convertToOneOfMany()
test_data._convertToOneOfMany()
val_data._convertToOneOfMany()

print('Depois do _convertToOneOfMany...')
print(train_data['target'][:5])

print('-------------------------------------------')

from pybrain.structure.modules import SoftmaxLayer

net = buildNetwork(4, 5, 3, outclass=SoftmaxLayer)
trainer = BackpropTrainer(net, dataset=train_data,
                          learningrate=0.01, momentum=0.5)
trainer.trainOnDataset(train_data, 100)

from pybrain.utilities import percentError

out = net.activateOnDataset(test_data).argmax(axis=1)
print('Erro de teste: %f' % percentError(out, test_data['class']))

import numpy

out = net.activateOnDataset(val_data).argmax(axis=1)
print('Erro de validação: %f' % percentError(out, val_data['class']))

print('-------------------------------------------')

print('Validação')
print('saída da rede:\t', out)
print('correto:\t', val_data['class'][:, 0])
