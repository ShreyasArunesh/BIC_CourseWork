


import numpy as np
import math

class activation:

    def sigmoid(self,x,derivative=False):
        if not derivative:
            return 1 / (1 + math.exp(-x))
        else:
            return self.sigmoid(x) * (1 - self.sigmoid(x))

    def relu(self,x,derivative=False):
        if not derivative:
            return max(0,x)
        else:
            return np.greater(x, 0).astype(int)

class lossFunction:
    def mse(y_true, y_pred,derivative=False):
        if not derivative:
            return np.mean(np.power(y_true - y_pred, 2))
        else:
            return 2 * (y_pred - y_true) / y_true.size

    def binary_cross_entropy(y_true, y_pred,derivative=False):
        if not derivative:
            return np.mean(np.power(y_true - y_pred, 2))
        else:
            return 2 * (y_pred - y_true) / y_true.size

class layer:

    def __init__(self,input_dimention,nNeurons,activation):
        self.weights = np.random.rand(input_dimention,nNeurons)
        self.bias = np.random.rand(1,nNeurons)
        self.activation = activation

        def forward_propagation(self, input_data):
            self.input = input_data
            self.neuronOutput = np.dot(self.input, self.weights) + self.bias
            self.activationOutput = self.activation(self.neuronOutput)
            return self.output

        def backward_propagation(self, output_error, learning_rate):
            activation_error = self.activation(self.neuronOutput,True) * output_error
            input_error = np.dot(activation_error, self.weights.T)
            weights_error = np.dot(self.input.T, activation_error)

            # update parameters
            self.weights -= learning_rate * weights_error
            self.bias -= learning_rate * activation_error
            return input_error

class mlp:
    def __init__(self):
        self.layers = list()
        self.loss = None
        self.loss_prime = None

    def add_layer(self, layer):
        self.layers.append(layer)

    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime


    def predict(self, input_data):
        pred = []
        for i in range(len(input_data)):
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            pred.append(output)
        return pred

    def fit(self, x_train, y_train, epochs, learning_rate):

        for i in range(epochs):
            delta = 0
            for j in range(len(x_train)):
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                delta += self.loss(y_train[j], output)

                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            delta /= len(x_train)
            print('epoch %d/%d   error=%f' % (i+1, epochs, delta))

