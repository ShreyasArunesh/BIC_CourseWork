
'''
 *
 * Name                   :   pso.py
 *
 * Description            :   it implements the following class
 *
 *                            1. PSO   -   implements the PSO algorithm.
 *
 *
 * Authors                :   Rémi Desmartin and Mohith Gowda Heggur Ramesh
 *
 *
'''


'''importing the dataset'''
import numpy as np
from random import randrange, uniform, seed
from typing import TypeVar, Callable

seed(10)

'''initializing the varibles'''
Param = TypeVar('Param')
FitnessFn = Callable[[Param], Param]


def random_particle(dims: int) -> np.array:
    '''
    *
    * Summary  :  it generates random points for the given dimensions.
    *
    * Args     :  dims   -   int    -   it hold the value for the dimensions.
    *
    * Returns  :  it returns random points in the given dimensions
    *
    '''

    return np.array([uniform(0, 1) for _ in range(dims)])


class PSO:
    '''
    *
    * Summary  : this class implements the PSO algorithm.
    *
    '''

    def __init__(self, fitness_fn: FitnessFn, dimensions: int, **kwargs):

        '''
           *
           * Summary  :  constructor to initialize the parameters of the PSO class.
           *
           * Args     :  REQUIRED :
           *                       fitness_fn          -  FitnessFn          -   reference to the function FitnessFn.
           *                       dimensions          -  int                -   it holds the value for the dimension.
           *             OPTIONAL :
           *                       **kwargs            - default parameters  -   default parameters
           *                        swarm_size         - int                 -   it holds the value for population size or swarm size.
           *                        informants_nb      - int                 -   it holds the value for the number of informants.
           *                        inertia            - float               -   it holds the value for inertia
           *                        cognitive_weight   - float               -   it holds the value for cognitive weight
           *                        social_weight      - float               -   it holds the value for social weight
           *                        step_size          - int                 -   it holds the value for step size
           *                        max_velocity       - float               -   it holds the value for maximum velocity of the particles.
           *                        min_velocity       - float               -   it holds the value for minimum velocity of the particles.
           *
           *
           * Returns  :  No return value.
           *
        '''


        parameters = {
            "swarm_size": 10,
            "informants_nb": 3,
            "step_size": 1,
            "inertia": 1,
            "cognitive_weight": 1,
            "social_weight": 1,
            "max_velocity": 1,
            "min_velocity": -1
        }
        for key in parameters:
            if key in kwargs:
                parameters[key] = kwargs[key]

        self.dimensions = dimensions

        """ hyper-parameters initialization """
        self.fitness_fn = fitness_fn
        self.swarm_size = parameters["swarm_size"]
        self.informants_nb = parameters["informants_nb"]
        self.step_size = parameters["step_size"]
        self.inertia = parameters["inertia"]
        self.cognitive_weight = parameters["cognitive_weight"]
        self.social_weight = parameters["social_weight"]
        self.max_v = parameters["max_velocity"]
        self.min_v = parameters["min_velocity"]

        """ particles & velocity vectors initialization"""
        self.particles = np.array([random_particle(dimensions) for _ in range(self.swarm_size)])
        self.velocities = np.array([random_particle(dimensions) - self.particles[i] for i in range(self.swarm_size)])
        self.fitness = np.array([0 for _ in range(self.swarm_size)])
        self.fittest_positions = np.copy(self.particles)
        self.fittest_values = np.array([self.fitness_fn(p) for p in self.particles])
        self.best = random_particle(self.dimensions)
        self.best_value = self.fitness_fn(self.best)

    def get_best_informant(self, index: int) -> np.array:
        '''
           *
           * Summary  :  it returns the best position of random informants for the particle with given index.
           *
           * Args     :  index   -   int    -    it holds the value for informants.
           *
           * Returns  :  it returns the best position of random informants for the particle with given index.
           *
           *
        '''

        informants_id = np.random.choice(range(self.swarm_size), self.informants_nb)
        if index not in informants_id:
            np.append(informants_id, index)
        informants_positions = np.array([self.fittest_positions[i] for i in informants_id])
        informants_values = np.array([self.fittest_values[i] for i in informants_id])
        max_index = np.argmax(informants_values)
        return informants_positions[max_index]

    def update_velocity(self, index: int) -> None:
        '''
           *
           * Summary  :  This function updates the velocity for the particle with given index.
           *
           * Args     :  index   -   int    -   it holds the value for the accessing the particles.
           *
           * Returns  :  No return value.
           *
        '''

        p = self.particles[index]
        a = uniform(0, self.inertia)
        b = uniform(0, self.cognitive_weight)
        c = uniform(0, self.social_weight)
        inertia_vector = a * self.velocities[index]
        cognitive_vector = b * (self.fittest_positions[index] - p)
        best_informant = self.get_best_informant(index)
        social_vector = c * (best_informant - p)
        v = inertia_vector + cognitive_vector + social_vector
        if np.max(v) > self.max_v or np.min(v) < self.min_v:
            v = np.clip(v, self.min_v, self.max_v)
        self.velocities[index] = v

    def update_best(self):
        '''
           *
           * Summary  : This function updates the best position found by the swarm.
           *
           * Args     : No arguments.
           *
           * Returns  : No return value.
           *
        '''

        max_index = np.argmax(self.fitness)
        if self.fitness[max_index] > self.fitness_fn(self.best):
            self.best = self.particles[max_index]
            self.best_value = self.fitness[max_index]

    def update_particles(self):
        '''
           *
           * Summary  : This fucntion updates the best position, velocities, and particles.
           *            It Represents one iteration of the PSO algorithm
           *
           * Args     : No arguments.
           *
           * Returns  : No return value.
           *
        '''


        self.fitness = np.apply_along_axis(self.fitness_fn, axis=1, arr=self.particles)
        # Update best solution
        self.update_best()

        for i in range(self.swarm_size):
            # update velocity
            self.update_velocity(i)
        # update particles positions
        self.particles = self.particles + self.step_size * self.velocities
        # update fittest positions & values for all particles
        for i, p in enumerate(self.particles):
            if self.fitness[i] > self.fittest_values[i]:
                self.fittest_positions[i] = p
                self.fittest_values[i] = self.fitness[i]
        # print('\n# Particles:')
        # print(self.particles)
        # print('# Velocities:')
        # print(self.velocities)

    def run(self, max_iter=200, precision=5, save_path=None):

        '''
           *
           * Summary  : This function will run the PSO algorithm with the given number of iterations.
           *
           * Args     : max_iter    -     int    -    holds the number of iterations to run on the algorithm.
           *            precision   -     int    -    it defines the precision of the output. default is 5.
           *            save-path   -     var    -    it holds the path to save the output. default is set to Null.
           *
           * Returns  : it returns the best fitness value.
           *
        '''

        convergence = 0
        f = None
        if save_path:
            print(save_path)
            f = open(save_path, 'w+')

        for i in range(max_iter):
            print('------------------- Iteration {} ---------------------------'.format(i))

            self.update_particles()
            print('best fitness: {}'.format(self.best_value))
            if save_path:
                f.write('{}\n'.format(self.best_value[0]))

        return self.best
