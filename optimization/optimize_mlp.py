'''Importing the Datasets'''
from mlp.src.mlp import MultiLayerPerceptron, Layer
from mlp.src.activation import *
from pso.pso import PSO
import matplotlib.pyplot as plt
import math

'''Declaring the Variable'''
activations = [
    sigmoid,
    hyperbolicTangent,
    cosine,
    gaussian
]


def value_to_activation_fn(value: float):
    '''
    *
    *  Summary : This function returns the activation function on input.
    *
    *  Args    : value  -   float    -    value to get the activation function.
    *
    *  Returns : Returns the Activation function.
    *
    '''
    if value < -1:
        value = -1
    elif value > 1:
        value = 1
    index = math.floor(value * len(activations)/2)
    return activations[index]


class MlpOptimizer:
    """
    *
    *  Summary  :  Optimize a MLP to approximate given data
    *
    """
    def __init__(self, mlp: MultiLayerPerceptron, data: np.array):
        '''
        *
        *  Summary : This function initializes the MlpOptimizer.py object .
        *
        *  Args    : mlp    -    MultiLayerPerceptron  -   object of MultiLayerPerceptron
        *            data   -    np.array              -   input data
        *
        *  Returns : No Return Values.
        *
        '''


        self.mlp = mlp
        self.best = None
        self.data = data

        def fitness_fn(position: np.array):
            '''
            *
            *  Summary : This function sets mlp weights & activation functions from position.
            *
            *  Args    : position   -    np.array   -   it holds the positions
            *
            *  Returns : It returns the Mean Square Error.
            *
            '''

            self.set_parameters(position)
            mse = 0
            for d in self.data:
                """ difference between expected and predicted """
                pred = mlp.evaluate(np.array(d[:-1]))
                # print('predicted value: f({}) = {}'.format(d[0], pred))
                mse += (pred - d[-1])**2
            mse /= len(self.data)
            return -mse

        self.fitness_fn = fitness_fn

    def set_parameters(self, position: np.ndarray):
        '''
        *
        *  Summary : This function sets mlp weights & activation functions from position.
        *
        *  Args    : position   -    np.array   -   it holds the positions
        *
        *  Returns : No Return Value.
        *
        '''

        """ set activation functions """
        activation_values = position[:self.mlp.layers.size]
        activation_fns = np.array([value_to_activation_fn(val) for val in activation_values])
        self.mlp.set_activations(activation_fns)

        """ set weights """
        weights = position[self.mlp.layers.size:]
        self.mlp.set_weights(weights)

    def optimize(self, max_iterations, file_path='', **pso_params):
        '''
        *
        *  Summary : This function will optimise the given MLP with a PSO and return the final
        *            best MLP's fitness.
        *
        *  Args    :  max_iterations  -    int               -   it holds the value for maximum iterations.
        *             file_path       -   string             -   it holds the path for the file.
        *             **pso_params    -   default parameters -   it holds the default parameters for the pso algorithm.
        *
        *  Returns : It returns the output from the fitness function.
        *
        '''
        pso = PSO(self.fitness_fn, self.mlp.parameters_size, **pso_params)
        self.best = pso.run(max_iterations, 10, save_path=file_path)
        self.set_parameters(self.best)
        return self.fitness_fn(self.best)

    def graph(self):
        '''
        *
        *  Summary : This function will optimise the given MLP with a PSO and return the final
        *            best MLP's fitness.
        *
        *  Args    : No Arguments.
        *
        *  Returns : No Return Value.
        *
        '''
        y_expected = [d[1] for d in self.data]
        y_predicted = [self.mlp.evaluate(np.array([d[0]])) for d in self.data]
        x_data = [d[0] for d in self.data]
        plt.plot(x_data, y_predicted)
        plt.plot(x_data, y_expected)
        plt.show()

