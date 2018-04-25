#camadas = neuronios artificiais

#classificação com pybrain com XOR de 3 dimensoes 
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised import BackpropTrainer

#criar conjunto de dados Dataset
dataset = SupervisedDataSet(3,1)

# adicionando as dimensoes = 3
dataset.addSample([0,0,0],[0])
dataset.addSample([0,0,1],[1])
dataset.addSample([0,1,0],[1])
dataset.addSample([0,1,1],[0])
dataset.addSample([1,0,0],[1])
dataset.addSample([1,0,1],[0])
dataset.addSample([1,1,0],[0])
dataset.addSample([1,1,1],[1])

#contruindo rede
#buildNetwork(dataset.indim,QTD NEURONIOS 1 CAMANDA ESCONDIDA, QTD NEURONIOS 2 CAMANDA ESCONDIDA,dataset.outdim,bias=True)

rede = buildNetwork(dataset.indim,4,4,dataset.outdim,bias=True)
trainer = BackpropTrainer(rede,dataset,learningrate=0.01,momentum=0.9)

#treinando
trainer.trainEpochs(1500)

#teste
teste = SupervisedDataSet(3, 1)
teste.addSample([0, 0, 0], [0])
teste.addSample([0, 0, 1], [1])
teste.addSample([0, 1, 0], [1])
teste.addSample([0, 1, 1], [0])
teste.addSample([1, 0, 0], [1])
teste.addSample([1, 0, 1], [0])
teste.addSample([1, 1, 0], [0])
teste.addSample([1, 1, 1], [1])

#processando
trainer.testOnData(teste,verbose=True)


'''
Resultado com 8 neuronios
esting on data:
('out:    ', '[0     ]')
('correct:', '[0     ]')
error:  0.00000000
('out:    ', '[1     ]')
('correct:', '[1     ]')
error:  0.00000000
('out:    ', '[1     ]')
('correct:', '[1     ]')
error:  0.00000000
('out:    ', '[0     ]')
('correct:', '[0     ]')
error:  0.00000000
('out:    ', '[1     ]')
('correct:', '[1     ]')
error:  0.00000000
('out:    ', '[0     ]')
('correct:', '[0     ]')
error:  0.00000000
('out:    ', '[0     ]')
('correct:', '[0     ]')
error:  0.00000000
('out:    ', '[1     ]')
('correct:', '[1     ]')
error:  0.00000000
('All errors:', [1.5777218104420236e-30, 8.874685183736383e-31, 8.874685183736383e-31, 0.0, 3.944304526105059e-31, 1.5777218104420236e-30, 0.0, 9.860761315262648e-32])
('Average error:', 6.77927340424307e-31)
('Max error:', 1.5777218104420236e-30, 'Median error:', 8.874685183736383e-31)
'''
