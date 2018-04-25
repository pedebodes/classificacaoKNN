#carregando dados do iris dataset com sklearn
from sklearn import datasets
iris = datasets.load_iris()

#obtendo entradas e saidas
x,y = iris.data, iris.target

from pybrain.datasets.classification import ClassificationDataSet
dataset = ClassificationDataSet(4, 1, nb_classes=3)  # nb_classes=3 (setosa,virginica,versicolor)

#adicionando amostras
for i in range(len(x)):
    dataset.addSample(x[i],y[i])

# print(len(dataset))#tamanho do dataset
# print((dataset['input']))#imput
# print((dataset['target']))  # saida (0=setosa,1=virginica,2=versicolor)


#particionando os dados para treinamento
train_data,part_data = dataset.splitWithProportion(0.6)# particionado o dataset em 60% para treinamento e 40% para o part_data
print("Quantidade para treino %d " % len(train_data))
print("Quantidade para teste %d " % len(part_data))

#dividindo dos dados para teste e validação
teste_data,val_data = part_data.splitWithProportion(0.5)#20% para teste e 5% para validação
print("Quantidade para teste %d " % len(teste_data))
print("Quantidade para validação %d " % len(val_data))


from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised import BackpropTrainer

#contruindo rede
rede = buildNetwork(dataset.indim,3,dataset.outdim)
trainer = BackpropTrainer(rede,dataset=train_data,learningrate=0.01,momentum=0.1,verbose=True)

#treinamento ate convergir obtendo erros
train_erros, val_erros = trainer.trainUntilConvergence(dataset=train_data,maxEpochs=100)

import matplotlib.pyplot as plt
plt.plot(train_erros,'b',val_erros,'r') # azul erro de treinamento, vermelhor erro de validação
plt.show()


print('Total de Epocas %d ' %trainer.totalepochs)

#outra forma de treinar o dataset
trainer.trainOnDataset(train_data,500)

#testar rede e obter saidas
out = rede.activateOnDataset(teste_data)

for i in range(len(out)):
    print('saida out: %f, correto %f' %(out[i],teste_data['target'][i]))


