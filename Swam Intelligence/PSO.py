from collections import Iterable
import numpy as np
from tqdm import tqdm

class ParticleSwarmOptimizer:

    def __init__(self, particle_cls, c1, c2, n_particles,
                 fitness_fn, compare_fn, n_iter=1, dims=None,
                 random=True, particles_list=None, position_range=None,
                 velocity_range=None):

        # self._validate(particle_cls, c1, c2, n_particles,
        #                fitness_fn, compare_fn, n_iter, dims,
        #                random, particles_list)

        self.particle_cls = particle_cls
        self.c1 = c1
        self.c2 = c2
        self.n_particles = n_particles
        self.n_iter = n_iter
        self.fitness_fn = fitness_fn
        self.compare_fn = compare_fn
        self.position_range = position_range
        self.velocity_range = velocity_range
        self.dims = dims
        self.random = random
        self.particles_list = particles_list

        self._init_particles_list()

    # def _validate(self, particle_cls, c1, c2, n_particles,
    #               fitness_fn, compare_fn, n_iter, dims,
    #               random, particles_list):
    #     if not issubclass(particle_cls, Particle):
    #         raise TypeError('particle_cls should inherit from particle.Particle class')
    #     if not isinstance(c1, float):
    #         raise TypeError('c1 should be of type float but got {}'.format(type(c1)))
    #     if not isinstance(c2, float):
    #         raise TypeError('c2 should be of type float but got {}'.format(type(c2)))
    #
    #     if not hasattr(fitness_fn, '__call__'):
    #         raise TypeError('fitness_fn should be a callable')
    #     temp = fitness_fn(np.random.randn(dims))
    #     if not isinstance(temp, float):
    #         raise TypeError('fitness_fn should return a single valued float but returned {}'.format(type(temp)))
    #
    #     if not hasattr(compare_fn, '__call__'):
    #         raise TypeError('compare_fn should be a callable')
    #     temp = compare_fn(10, 20)
    #     if not isinstance(temp, bool):
    #         raise TypeError('compare_fn should return a bool but returned {}'.format(type(temp)))
    #
    #     if not isinstance(n_particles, int):
    #         raise TypeError('n_particles should be of type int but got {}'.format(type(n_particles)))
    #     if n_particles <= 0:
    #         raise ValueError('n_particles should be a positive integer')
    #
    #     if not isinstance(dims, int):
    #         raise TypeError('dims should be of type int but got {}'.format(type(dims)))
    #     if not isinstance(random, bool):
    #         raise TypeError('random should be of type bool but got {}'.format(type(c1)))
    #
    #     if not random:
    #         if not isinstance(particles_list, Iterable):
    #             raise TypeError('particles_list should be an Iterable but got {}'.format(type(particles_list)))
    #         if len(particles_list) == n_particles:
    #             raise ValueError('particles_list should {} number of particles.'.format(n_particles))
    #
    #         for i, particle in enumerate(particles_list):
    #             if not isinstance(particle, particle_cls):
    #                 raise TypeError('Every particle in particles_list must be an object of class particle_cls'
    #                                 ' but got object of type {} at position {}'.format(type(particle), i))

    def _get_fitness(self, position):
        return self.fitness_fn(position)

    def _update_gbest(self):
        for particle_i in self.particles_list:
            l1, l2 = self._get_fitness(particle_i.pbest), self._get_fitness(self.gbest)
            print(l1)
            if self.compare_fn(l1, l2):
                self.gbest = particle_i.position

    def _init_particles_list(self):
        if self.random:
            self.particles_list = []

            for i in range(self.n_particles):
                particle = self.particle_cls(self.random, position_range=self.position_range,
                                             velocity_range=self.velocity_range, dims=self.dims)
                self.particles_list.append(particle)

        self.gbest = self.particles_list[0].position
        self._update_gbest()

        self.dims = self.particles_list[0].dims

    def optimize(self):
        for i in tqdm(range(self.n_iter)):
            for particle in self.particles_list:
                particle.update(self.c1, self.c2, self.gbest,
                                self.fitness_fn, self.compare_fn)
            self._update_gbest()
        return self


from collections import Iterable
import numpy as np


class Particle:

    def __init__(self, random, position=[0.],
                 velocity=[0.], position_range=None,
                 velocity_range=None, dims=None, alpha=0.1):
        self._validate(random, position, velocity, position_range, velocity_range, dims, alpha)

        self.random = random
        self.position = position
        self.velocity = velocity
        self.position_range = position_range
        self.velocity_range = velocity_range
        self.dims = dims
        self.alpha=alpha

        self._init_particle()

        self.pbest = self.position

    def _validate(self, random, position,
                 velocity, position_range,
                 velocity_range, dims, alpha):
        if not isinstance(random, bool):
            raise TypeError('random should be of type bool but got type {}'.format(type(random)))
        if not isinstance(alpha, float):
            raise TypeError('alpha should be of type float but got type {}'.format(type(alpha)))

        if random is True:
            if not isinstance(position_range, Iterable):
                raise TypeError('When random is True position_range should be an'
                                 ' Iterable of length 2 but got {}.'.format(type(position_range)))
            if not isinstance(velocity_range, Iterable):
                raise TypeError('When random is True velocity_range should be an'
                                 ' Iterable of length 2 but got {}.'.format(type(position_range)))
            if not isinstance(dims, int):
                raise TypeError('When random is True dims should be an'
                                 ' int but got {}.'.format(type(position_range)))
        elif random is False:
            if not isinstance(position, Iterable):
                raise TypeError('When random is False position should be an'
                                 ' Iterable but got {}.'.format(type(position_range)))
            if not isinstance(velocity, Iterable):
                raise TypeError('When random is False velocity should be an'
                                 ' Iterable but got {}.'.format(type(position_range)))

    def _init_particle(self):
        if self.random:
            self.position = np.random.uniform(low=self.position_range[0],
                                              high=self.position_range[1],
                                              size=(self.dims,))
            self.velocity = np.random.uniform(low=-abs(self.velocity_range[1]-self.velocity_range[0]),
                                              high=abs(self.velocity_range[1]-self.velocity_range[0]),
                                              size=(self.dims,))
        else:
            self.position = np.asarray(position)
            self.velocity = np.asarray(velocity)
            self.dims = self.position.shape[0]

    def update(self, c1, c2, gbest, fitness_fn, compare_fn):
        if not isinstance(c1, float):
            raise TypeError('c1 should be of type float but got {}'.format(type(c1)))
        if not isinstance(c2, float):
            raise TypeError('c2 should be of type float but got {}'.format(type(c2)))
        if not isinstance(gbest, type(self.position)):
            raise TypeError('gbest should have same type as Particle\'s velocity,'
                            'which is of type {}'.format(type(self.velocity)))
        if self.position.shape[0] != gbest.shape[0]:
            raise ValueError('gbest should have shape {} but got shape {}'.format(
                self.position.shape, gbest.shape))

        self._update_velocity(c1, c2, gbest)
        self._update_position(fitness_fn, compare_fn)

    def _update_velocity(self, c1, c2, gbest):
        self.alpha = self.alpha / 2
        wrt_pbest = c1 * np.random.rand() * (self.pbest - self.position)
        wrt_gbest = c2 * np.random.rand() * (gbest - self.position)
        self.velocity = self.alpha * self.velocity + wrt_pbest + wrt_gbest

    def _update_position(self, fitness_fn, compare_fn):
        self.position = self.position + self.velocity + 0.01 * self.position
        if compare_fn(fitness_fn(self.position), fitness_fn(self.pbest)):
            self.pbest = self.position

    def __repr__(self):
        return '<Particle: dims={} random={}>'.format(self.dims, self.random)


