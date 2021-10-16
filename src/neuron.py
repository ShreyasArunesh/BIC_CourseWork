
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
        elif activation == "identity":
            self.activation = lambda x: x


    def compute(self,input):
        self.input = input
        self.input = np.insert(self.input, 0, 1, axis=0)
        if len(self.input) == len(self.weights):
            self.output = np.dot(self.weights,self.input)
            self.output = self.activation(self.output)

    def update_weights(self):
        pass
