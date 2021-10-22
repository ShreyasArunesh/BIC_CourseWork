
import math
import numpy as np

# activation function and its derivative
def relu(x):
    return np.maximum(x,0)

def d_relu(x):
    return np.greater(x, 0).astype(int)

def tanh(x):
    return np.tanh(x)

def d_tanh(x):
    return 1 - np.tanh(x) ** 2

def sigmoid(x):
    return np.array([1 / (1 + math.exp(-i)) for i in x])

def d_sigmoid(x):
    return sigmoid(x) * 1-sigmoid(x)

# loss function and its derivative
def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2))

def d_mse(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size

def binary_cross_entropy(y_true, y_pred):
    if y_true == 1:
        return - np.log(y_pred)
    else:
        return - np.log(1-y_pred)

def d_binary_cross_entropy(y_true, y_pred):
    if y_true == 1:
        return np.array(-1/ y_pred)
    else:
        return np.array(1/(1-y_pred))

# inherit from base class Layer
class FCLayer:
    def __init__(self, input_size, output_size,activation):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5
        if activation == "relu":
            self.activation = relu
            self.activation_prime = d_relu
        if activation == "sigmoid":
            self.activation = sigmoid
            self.activation_prime = d_sigmoid
        if activation == "tanh":
            self.activation = tanh
            self.activation_prime = d_tanh

    # returns output for a given input
    def forward_propagation(self, input_data):
        self.input = input_data
        self.neuron_output = np.dot(self.input, self.weights) + self.bias
        self.activation_output = self.activation(self.neuron_output)
        return self.activation_output

    # computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
    def backward_propagation(self, output_error, learning_rate):
        activation_error = self.activation_prime(self.neuron_output) * output_error
        input_error = np.dot(activation_error, self.weights.T)
        weights_error = np.dot(self.input.T, activation_error)
        # dBias = output_error

        # update parameters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * activation_error
        return input_error


class MLP:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.d_loss = None

    # add layer to network
    def add_layer(self, layer):
        self.layers.append(layer)

    # set loss to use
    def compile(self, loss):
        if loss == 'mse':
            self.loss = mse
            self.d_loss = d_mse
        if loss == 'binary_cross_entropy':
            self.loss = binary_cross_entropy
            self.d_loss = d_binary_cross_entropy

    # predict output for given input
    def predict(self, input_data):
        # sample dimension first
        result = []
        # run network over all samples
        for i in range(len(input_data)):
            # forward propagation
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)

        return result

    # train the network
    def fit(self, x_train, y_train, epochs, learning_rate):

        # training loop
        for i in range(epochs):
            err = 0
            for j in range(len(x_train)):
                # forward propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # compute loss (for display purpose only)

                err += self.loss(y_train[j], output)

                # backward propagation
                error = np.array(self.d_loss(y_train[j], output)).reshape(1, 1)

                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)
            err /= len(x_train)


#
# # x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
# # y_train = np.array([[[0]], [[1]], [[1]], [[0]]])
#
# #
# input_array = np.loadtxt("dataset/data_banknote_authentication.txt",delimiter=',')
# np.random.shuffle(input_array)
# x_train = input_array[:,:-1]
# x_train = np.reshape(x_train,(-1,1,4))
# y_train = input_array[:,-1]
# y_train = np.reshape(y_train,(-1,1,1))
#
# # network
# model = MLP()
# model.add_layer(FCLayer(4, 3,"tanh"))
# model.add_layer(FCLayer(3, 3,"tanh"))
# model.add_layer(FCLayer(3, 2,"tanh"))
# model.add_layer(FCLayer(2, 1,"sigmoid"))
#
#
# # train
# model.compile("binary_cross_entropy")  #binary_cross_entropy
# model.fit(x_train, y_train, 1000, learning_rate=0.1)
#
# # test
# out = model.predict(x_train)
#
# # Calculating Accuracy
# def accuracy(y_train, out):
#     accuracy = np.mean(out == y_train)
#     return accuracy
#
# print(accuracy(np.round(y_train,0), np.round(out,0)))