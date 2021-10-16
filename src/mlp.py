

import numpy as np
import math

class neuron:

    def __init__(self,input_shape,activation):
        self.weights = np.random.rand(input_shape+1)
        self.output = 0

        if activation == "sigmoid":
            self.activation = lambda x: 1 / (1 + math.exp(-x))
        elif activation == "relu":
            self.activation = lambda x: max(0,x)
        else:
            self.activation = lambda x: x


    def compute(self,input):
        self.input = input
        self.input = np.insert(self.input, 0, 1, axis=0)
        if len(self.input) == len(self.weights):
            self.output = np.dot(self.weights,self.input)
            self.output = self.activation(self.output)
            return self.output

    def update_weights(self):
        pass

class layers:

    def __init__(self):
        self.neurons = list()
        self.output = list()

    def add_neurons(self ,nNeurons ,activation ,input_dim):
        for i in range(nNeurons):
                self.neurons.append(neuron(input_dim, activation))


    def get_output(self ,input):
        for neuron in self.neurons:
            self.output.append(neuron.compute(np.array(input)))
        return self.output


class mlp:
    def __init__(self):
        self.layers = list()
        self.output = 0
    def add_layer(self,nNeurons,activation,input_shape=False):

        layer = layers()
        if input_shape == False:
            layer.add_neurons(nNeurons, activation, self.prev_shape)
            self.prev_shape = nNeurons
        else:
            layer.add_neurons(nNeurons, activation, input_shape)
            self.prev_shape = nNeurons

        self.layers.append(layer)


    def train(self,x_train,y_train,epochs,learning_rate,loss):
        self.x_train = x_train
        self.y_train = y_train
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.loss = loss
        layer_out = list()
        for row in self.x_train:
            for i,layer in enumerate(self.layers):
                if i == 0:
                    layer_out = layer.get_output(row)
                else:
                    layer_out = layer.get_output(layer_out)

        self.output = layer_out

    def predict(self,x_input):
        return "output"



m = mlp()
m.add_layer(2,"identity",3)
m.add_layer(2,"identity")
m.add_layer(1,"identity")
m.output
m.train([[1,1,1]],0,1,1,1)

for l in m.layers:
    for n in l.neurons:
        print(n.weights)