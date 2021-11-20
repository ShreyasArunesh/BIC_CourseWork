
'''Importing the Packages'''
from experiments.experiments import *
from experiments.data_vis import DataVisualisation
import sys

def main():
    '''
    *
    * Summary  :  This code carries out experiments on different hyper-parameters.
    *             mainly on informants, swarm size, cognitive and social weights.
    *
    * Args     : sys.argv   -   CLI      -    takes in value '--informants' to activate
    *                                         the informant testing.
    *
    '''
    if '--informants' in sys.argv:
        run_informants_nb(2, 10, swarm_size=100)
    elif '--swarm_size' in sys.argv:
        run_swarm_size([1, 3, 5, 10, 20, 50, 100, 250, 500])
    else:
        run_acceleration_coeff(0, 4, swarm_size=100, informants_nb=5)

if __name__ == '__main__':
    main()

    # paths = [
    #     'results/informants_nb_experiment/informants_2/',
    #     'results/informants_nb_experiment/informants_3/',
    #     'results/informants_nb_experiment/informants_4/',
    #     'results/informants_nb_experiment/informants_5/',
    #     'results/informants_nb_experiment/informants_6/',
    #     'results/informants_nb_experiment/informants_7/',
    #     'results/informants_nb_experiment/informants_8/',
    #     'results/informants_nb_experiment/informants_9/',
    # ]
    # dv = DataVisualisation(paths)
    # dv.plot_speed()
