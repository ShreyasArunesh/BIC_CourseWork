
'''importing packages'''
import csv
import matplotlib.pyplot as plt
import os

from mlp.src.activation import *
from mlp.src.mlp import MultiLayerPerceptron, Layer
from optimization.optimize_mlp import MlpOptimizer


def compare_graphs(file_paths):
    '''
    *
    *  Summary : It visualizes the data in the form of a plot.
    *
    *  Args    : file_paths   -   List   -   it holds the path for the files to visualize.
    *
    *  Returns : no return value
    *
    '''

    data = []
    for file in file_paths:
        with open(file) as f:
            reader = csv.reader(f)
            d = np.array([line for line in reader], dtype=float)
            data.append(d.flatten())
    for index, series in enumerate(data):
        label = file_paths[index].split('/')[-2]
        plt.plot(series[:30], label=label)
    plt.legend(loc="lower right")
    plt.show()

'''storing the dataset path to a list'''
datasets = [
    "/Users/shreyasarunesh/Desktop/bic-coursework-2020-master/dataset/data_banknote_authentication.txt",
]


def run_experiment(run_path, write_mean, **params):
    '''
    *
    *  Summary : This method will test MLP and PSO on every dataset and writes the data to a file.
    *
    *  Args    : run_path    -  String              -   it holds the path for the files to visualize.
    *            write_mean  -  function            -   its a function to write MSE to the files.
    *            **params    -  default parameters  -   default parameters
    *
    *  Returns : no return value
    *
    '''

    for ds in datasets:
        mse = []
        func_name = "Output"
        for i in range(5):
            mlp = MultiLayerPerceptron()
            mlp.add_layer(Layer(2, hyperbolicTangent, 4))
            mlp.add_layer(Layer(1, hyperbolicTangent))
            mlp.summary()
            data = np.loadtxt(ds,delimiter=',')
            opt = MlpOptimizer(mlp, data)
            file_path = '{}/{}_{}.csv'.format(run_path, func_name, i)
            res = opt.optimize(max_iterations=75, file_path=file_path, **params)
            mse.append(res)
        mean_mse = sum(mse)/len(mse)
        write_mean(func_name, mean_mse)

def run_acceleration_coeff(range_start, range_end, **params):

    '''
    *
    *  Summary : This method will test MLP and PSO on every dataset with different Hyper-parameters.
    *            it takes in a range i.e. start and end values and tests within the given range.
    *            this method conducts experiments varying the number for social and cognitive weights.
    *
    *  Args    : range_start   -   var                -   it holds the initial value to start testing the algorithms.
    *            range_end     -   var                -   it holds the highest value to end testing the algorithms.
    *            **params      -  default parameters  -   default parameters
    *
    *  Returns : no return value
    *
    '''

    r = [x/10 for x in range(range_start * 10, range_end * 10)]
    experiment_path = './acceleration_coeff_experiment'
    if not os.path.isdir(experiment_path):
        os.mkdir(experiment_path)

    for i in r:
        params['social_weight'] = i
        params['cognitive_weight'] = 4 - i
        run_path = '{}/acc_coff_{}_{}'.format(experiment_path, i, 4-i)

        def write_mean(func_name, mean_mse):
            mean_path = '{}/{}_mean_mse.csv'.format(experiment_path, func_name)
            with open(mean_path, 'a+') as f:
                f.write('{}, {}, {}\n'.format(i, 4-i, mean_mse[0]))

        if not os.path.isdir(run_path):
            os.mkdir(run_path)
        run_experiment(run_path, write_mean, **params)


def run_informants_nb(range_start, range_end, **params):
    '''
    *
    *  Summary : This method will test MLP and PSO on every dataset with different Hyper-parameters.
    *            it takes in a range i.e. start and end values and tests within the given range.
    *            this method conducts experiments varying the number of informants
    *
    *  Args    : range_start   -   var                -   it holds the initial value to start testing the algorithms.
    *            range_end     -   var                -   it holds the highest value to end testing the algorithms.
    *            **params      -  default parameters  -   default parameters
    *
    *  Returns : no return value
    *
    '''

    r = [x for x in range(range_start, range_end)]
    experiment_path = './informants_nb_experiment'
    if not os.path.isdir(experiment_path):
        os.mkdir(experiment_path)

    for i in r:
        params['informants_nb'] = i
        run_path = '{}/informants_{}'.format(experiment_path, i)

        def write_mean(func_name, mean_mse):
            mean_path = '{}/{}_mean_mse.csv'.format(experiment_path, func_name)
            with open(mean_path, 'a+') as f:
                f.write('{}, {}\n'.format(i, mean_mse[0]))

        if not os.path.isdir(run_path):
            os.mkdir(run_path)
        run_experiment(run_path, write_mean, **params)

def run_swarm_size(value_range, **params):
    '''
        *
        *  Summary : This method will test MLP and PSO on every dataset with different Hyper-parameters.
        *            it takes in a range value and tests within the given range.
        *            this method conducts experiments varying the swarm size.
        *
        *  Args    : value_range   -   var                -   it holds the highest value to end testing the algorithms.
        *            **params      -  default parameters  -   default parameters
        *
        *  Returns : no return value
        *
    '''
    experiment_path = './swarm_size_experiment'
    if not os.path.isdir(experiment_path):
        os.mkdir(experiment_path)

    for i in value_range:
        params['swarm_size'] = i
        run_path = '{}/swarm_size_{}'.format(experiment_path, i)

        def write_mean(func_name, mean_mse):
            mean_path = '{}/{}_mean_mse.csv'.format(experiment_path, func_name)
            with open(mean_path, 'a+') as f:
                f.write('{}, {}\n'.format(i, mean_mse[0]))

        if not os.path.isdir(run_path):
            os.mkdir(run_path)
        run_experiment(run_path, write_mean, **params)
