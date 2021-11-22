import numpy as np

def relu(x):
    return np.maximum(x, 0)

def leaky_rely(x, alpha=0.01):
    nonlin = relu(x)
    nonlin[nonlin==0] = alpha * x[nonlin == 0]
    return nonlin

def sigmoid(x):
    return 1 / (1 + np.exp(x))

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def tanh(x):
    return np.tanh(x)


class Loss:

    def __init__(self, data_loader, layers, n_wts, dims):
        self.data_loader = data_loader
        self.layers = layers
        self.n_wts = n_wts
        self.dims = dims

    def _forward(self, wts):
        w_index = 0
        X, y = next(self.data_loader)
        for i, layer in enumerate(self.layers):
            X = layer.forward(wts[w_index:w_index+self.n_wts[i]], X)
            w_index += self.n_wts[i]
        return y, X

    def _loss(self, y, y_hat):
        raise NotImplementedError()

    def __call__(self, wts):
        raise NotImplementedError()

class MSELoss(Loss):

    def _loss(self, y, y_hat):
        return np.mean((y - y_hat) ** 2)

    def __call__(self, wts):
        y, y_hat = self._forward(wts)
        return self._loss(y, y_hat)

class RMSELoss(Loss):

    def _loss(self, y, y_hat):
        return np.sqrt(np.mean((y - y_hat) ** 2))

    def __call__(self, wts):
        y, y_hat = self._forward(wts)
        return self._loss(y, y_hat)

class BinaryCrossEntropyLoss(Loss):

    def _loss(self, y, y_hat):
        left = y * np.log(y_hat + 1e-7)
        right = (1 - y) * np.log((1 - y_hat) + 1e-7)
        return -np.mean(left + right)

    def __call__(self, wts):
        y, y_hat = self._forward(wts)
        return self._loss(y, y_hat)

class CrossEntropyLoss(Loss):

    def _loss(self, y, y_hat):
        return -np.mean(y * np.log(y_hat + 1e-7))

    def __call__(self, wts):
        y, y_hat = self._forward(wts)
        return self._loss(y, y_hat)


class Layer:

    def __init__(self, in_units, units, activation):
        self.w_shape = (in_units, units)
        self.b_shape = (1, units)
        self.n_wts = in_units * units + units
        self.shape = (-1, units)
        self.activation = activation

    def _reshape_weights(self, wts):
        W = np.reshape(wts[:self.w_shape[0] * self.w_shape[1]], self.w_shape)
        b = np.reshape(wts[self.w_shape[0] * self.w_shape[1]:], self.b_shape)
        return W, b

    def forward(self, wts, x):
        W, b = self._reshape_weights(wts)
        return self.activation(np.dot(x, W) + b)




def __set_shape(self, new_shape: Tuple[4, 3) -> None:
    self.weights = np.full((new_shape[1], new_shape[0] + 1), 1.)  # add 1 to new_shape[0] for the bias
    self.__shape = new_shape