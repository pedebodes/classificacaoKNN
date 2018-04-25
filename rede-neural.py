# Redes Neurais utilizando PyBrain

from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised import BackpropTrainer

# passa as dimensões dos vetores de entrada e do objetivo
dataset = SupervisedDataSet(2, 1)
#exmplo porta logica XOR 
#tabela da verdade
dataset.addSample([1, 1], [0])
dataset.addSample([1, 0], [1])
dataset.addSample([0, 1], [1])
dataset.addSample([0, 0], [0])

#Construi rede (tamanho de entrada,quantidade de camadas,dimensao da saida,bias = melhor adaptacao da rede)
network = buildNetwork(dataset.indim, 4, dataset.outdim, bias=True)
# treino (rede,dataset,taxa de aprendizado,aumentar a velocidade de treinamento)
trainer = BackpropTrainer(network, dataset, learningrate=0.01, momentum=0.99)

#pode usar o for
'''
for epoch in range(1000): # treina por 1000 épocas
	trainer.train()
'''
# mas dessa forma e mais facil similar ao for
trainer.trainEpochs(1000)
'''
	treinar até a convergência: trainer.trainUntilConvergence
'''
#testa a rede com o conjunto de dados
test_data = SupervisedDataSet(2, 1)

test_data.addSample([1, 1], [0])
test_data.addSample([1, 0], [1])
test_data.addSample([0, 1], [1])
test_data.addSample([0, 0], [0])

#verbose para exibir msg
trainer.testOnData(test_data, verbose=True)

# Saidas sem erro  >> trainer.trainEpochs(1000)
# Testing on data:
# ('out:    ', '[0     ]')
# ('correct:', '[0     ]')
# error:  0.00000000
# ('out:    ', '[1     ]')
# ('correct:', '[1     ]')
# error:  0.00000000
# ('out:    ', '[1     ]')
# ('correct:', '[1     ]')
# error:  0.00000000
# ('out:    ', '[0     ]')
# ('correct:', '[0     ]')
# error:  0.00000000
# ('All errors:', [5.831334885496242e-18, 6.5285858131026876e-18,
#                  1.1039192816369955e-17, 6.3833172486639965e-18])
# ('Average error:', 7.44560769090822e-18)
# ('Max error:', 1.1039192816369955e-17, 'Median error:', 6.5285858131026876e-18)


# saida com erro  sendo refinado no numero de epocas para acertar  >> trainer.trainEpochs(100)
# Testing on data:
# ('out:    ', '[-0.05 ]')
# ('correct:', '[0     ]')
# error:  0.00125215
# ('out:    ', '[1.15  ]')
# ('correct:', '[1     ]')
# error:  0.01124466
# ('out:    ', '[1.148 ]')
# ('correct:', '[1     ]')
# error:  0.01102223
# ('out:    ', '[0.004 ]')
# ('correct:', '[0     ]')
# error:  0.00000874
# ('All errors:', [0.0012521496305153293, 0.011244657694179214,
#                  0.011022228793109615, 8.736959097024393e-06])
# ('Average error:', 0.005881943269225295)
# ('Max error:', 0.011244657694179214, 'Median error:', 0.011022228793109615)
