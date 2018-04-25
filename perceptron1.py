# implementação da perceptron sem bibliotecas

# Implementação Perceptron

import random


class Perceptron:

	def __init__(self, amostras, saidas, taxa_aprendizado=0.1, epocas=1000, limiar=-1):
		self.amostras = amostras
		self.saidas = saidas
		self.taxa_aprendizado = taxa_aprendizado
		self.epocas = epocas
		self.limiar = limiar
		self.n_amostras = len(amostras)
		self.n_atributos = len(amostras[0])
		self.pesos = []

	def treinar(self):

		for amostra in self.amostras:
			amostra.insert(0, -1) #adiciona -1 para a amostra
        
       	for i in range(self.n_atributos): # adiciona peso na amostra randonico (aleatorio)
			self.pesos.append(random.random())

		self.pesos.insert(0, self.limiar) 
		n_epocas = 0  # contador de épocas

		while True:

			erro = False  # erro inicialmente inexiste
# TODO implementação
			if not erro:
				break


# OR
amostras = [[0, 0], [0, 1], [1, 0], [1, 1]]
saidas = [0, 1, 1, 1]
rede = Perceptron(amostras, saidas)
rede.treinar()
