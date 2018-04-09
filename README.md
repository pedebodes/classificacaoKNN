# classificacao k-NN

O nome k-NN vem de “k-Nearest Neighbor” (k-ésimo vizinho mais próximo), ou seja, o método tem como objetivo classificar x_t atribuindo a ele o rótulo representado mais frequentemente dentre as k amostras mais próximas e utilizando um esquema de votação. O que determina essa “proximidade entre vizinhos” é uma distância calculada entre pontos em um espaço euclidiano (instâncias), cujas fórmulas mais comuns são a distância Euclidiana e a distância Manhattan.

O algoritmo k-NN consiste em, para cada nova amostra:

* Calcular a distância entre o exemplo desconhecido e o outros exemplos do conjunto de treinamento
* Identificar os K vizinhos mais próximos
* O rótulo da classe com mais representantes no conjunto definido no passo anterior será o escolhido (voto majoritário

https://www.monolitonimbus.com.br/classificacao-usando-knn/
