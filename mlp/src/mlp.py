
'''importing packages'''
from typing import Tuple, Sequence, Callable
import numpy as np


ActivationFn = Callable[[float], float]


class Layer:
    """
    *
    * Summary :    This class implements a single layer of the Network.
    *
    """
    
    def __init__(self, perceptron_nb: int, activation: ActivationFn, input_nb=0) -> None:
        
        '''
         *
         *  Summary : this block implements the constructor for the class
         *
         *  Args    : perceptron_nb     -              int         - Number of neurons in layer, which is also the shape 
         *                                                           of the layer's output        
         *        
         *            activation        -    Activation function   - it defines the activation function
         *
         *            input_nb          -              int         - Should be only provided for the first layer of the 
         *                                                           perceptron It gives the expected input shape. 
         *                                                           This is automatically computed for all the other 
         *                                                           layers and should be left at 0 for all layers that 
         *                                                           are not the first one.
         *
         *
         *  Returns : no return value
         *
        '''
     
        self.__shape: Tuple[int, int] = (perceptron_nb, input_nb)
        self.__set_shape([input_nb, perceptron_nb])
        self.activation = activation

    def __set_shape(self, new_shape: Tuple[int, int]) -> None:
        
        '''
         *
         *  Summary : this function sets the shape of the layer. When the shape of the layer is modified, the weights 
         *            are modified too to keep the layer fully connected with the previous layer.
         *
         *  Args    : new_shape   -   Tuple[int, int]   -   this defines the input and output shape
         *                                                  shape[0]: shape of the layer's input
         *                                                  shape[1]: shape of the layer's output         
         *        
         *  Returns : no return value
         *
        '''

        # TODO: try to reshape weights by reshaping arrays instead of creating new ones
        # self.weights = np.random.rand(new_shape[0], new_shape[1])
        self.weights = np.full((new_shape[1], new_shape[0] + 1), 1.) # add 1 to new_shape[0] for the bias
        self.__shape = new_shape


    
    
    
    def __get_shape(self) -> Tuple[int, int]:
        
        '''
         *
         *  Summary : This function gets the shape of the layer.
         *
         *  Args    : No Arguments
         * 
         *  Returns : It returns the shape of the layer. 
         *            this defines the input and output shape
         *            shape[0]: shape of the layer's input
         *            shape[1]: shape of the layer's output 
         *
        '''
        return self.__shape


    shape = property(__get_shape, __set_shape)



    def output(self, inputs: np.ndarray) -> np.ndarray:
        
        '''
         *
         *  Summary : This function computes the activation on the output.
         *            activation(dot(weights, inputs) + bias)
         *
         *  Args    : inputs  - np.ndarray  -  numpy array of input values
         * 
         *  Returns : It returns numpy array of output values. 
         *            
         *
        '''

        ''' add 1 to beginning of inputs for the bias '''
        inputs = np.insert(inputs, 0, 1., axis=0)
        ''' verifies inputs' shape is correct '''
        if inputs.shape[0] != self.weights.shape[1]:
            raise ArithmeticError("Invalid input dimension")
            
        ''' computes result '''
        fn = lambda i: np.dot(self.weights[i], inputs)
        weighted_sum = np.fromfunction(fn, (self.shape[1],), dtype=int)
        return self.activation(weighted_sum)


class MultiLayerPerceptron:
    """
    *
    * Summary :    This class implements a multiple layers neural network.
    *
    """
    def __init__(self) -> None:
        
        '''
         *
         *  Summary : This block implements the constructor for the class
         *
         *  Args    : No Arguments
         *
         *  Returns : No return value
         *
        '''
        self.layers = np.array([])


    def add_layer(self, new_layer: Layer) -> None:
        
        '''
         *
         *  Summary : This block adds a layer to the neural network.
         *
         *  Args    : new_layer  -  Layer (class)  -  This holds an object of the class 'Layer'.
         *                                            if this holds the first layer, then 'input_nb' must be provided.
         *
         *  Returns : No return value
         *
        '''
            
        ''' if hidden or output layer, set  the input shape to be the output shape of the previous layer'''
        if self.layers.size != 0:
            prev_layer = self.layers[-1]
            if prev_layer.shape[1] != new_layer.shape[0]:
                new_layer.shape = (prev_layer.shape[1], new_layer.shape[1])
        self.layers = np.append(self.layers, new_layer)

    def set_activations(self, fns: np.ndarray):
        for (layer, fn) in zip(self.layers, fns):
            layer.activation = fn

    def set_weights(self, weights: np.ndarray):
        """
         *
         * Summary : set the weights of the neural network
         *
         * Args    : weights  -  The weights to use for the network
         *
         * Returns : No return value.
        """
        """ check if new weights corresponds to architecture of network """
        if self.weights_size != weights.size:
            raise Exception('Bad weights size: expected {}, got {}'.format(self.weights_size, weights.size))
        for layer in self.layers:
            w = weights[:layer.weights.size]
            w.resize((layer.shape[1], layer.shape[0] + 1))  # add 1 to shape[0] to account for the bias
            layer.weights = w
            weights = weights[layer.shape[0] * layer.shape[1]:]

    @property
    def weights_size(self) -> int:
        """
         *
         * Summary : Return the total number of connections (i.e. the number of weights) in the Network.
         *
         * Args    : No Arguments
         *
         * Returns : Return the total number of connections (i.e. the number of weights) in the Network.
        """
        return sum([layer.weights.size for layer in self.layers])

    @property
    def parameters_size(self) -> int:
        """
         *
         * Summary : Return the total number of parameters to be optimized by the PSO i.e. number of weights + number of layers
         *
         * Args    : No Arguments
         *
         * Returns : Return the total number of parameters to be optimized by the PSO i.e. number of weights + number of layers
        """
        return self.weights_size + self.layers.size

    def evaluate(self, inputs: np.ndarray) -> np.ndarray:
        
        '''
         *
         *  Summary : This block evaluates the output of the network from the provided inputs
         *
         *  Args    : inputs  -  np.adarray  -  It holds the data in the form of numpy array.
         *
         *  Returns : It returns computed outputs of the Neural network
         *
        '''
        # evaluate each layer
        prev_output = inputs
        for layer in self.layers:
            prev_output = layer.output(prev_output)
        return prev_output

    def summary(self) -> None:
        '''
         *
         *  Summary : This block prints out the summary of the Feedforward Neural Network.
         *
         *  Args    : No Arguments
         *
         *  Returns : No Return value.
         *
        '''
        
        print('Total Layers: {}'.format(self.layers.size))
        print('Input Size: {}'.format(self.layers[0].shape[0]))
        i = 0
        for layer in self.layers:
            i += 1
            print('\nLayer #{}: size {}'.format(i, layer.shape[1]))
            print(layer.weights)
            print("Activation Function : " + layer.activation.__name__)
