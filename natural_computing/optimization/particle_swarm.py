"""
Particle Swarm Optimization
    This module implements the Particle Swarm Optimization algorithm.

Classes:
    ParticleSwarmOptimization: The implementation of the Particle Swarm
        Optimization algorithm.
"""

import random
from typing import Dict, List, Tuple, cast

from natural_computing.objective_functions import BaseFunction
from natural_computing.optimization import BaseOptimizer


class ParticleSwarmOptimization(BaseOptimizer):
    """
    Particle Swarm Optimization (PSO) algorithm implementation.

    Args:
        num_particles (int): The number of particles in the swarm.
        max_iterations (int): The maximum number of optimization iterations.
        inertia_weight (float): The inertia weight for velocity update.
        cognitive_weight (float): The cognitive weight for velocity update.
        social_weight (float): The social weight for velocity update.
        search_space (list of tuples): The search space for particle positions.

    Attributes:
        num_particles (int): The number of particles in the swarm.
        max_iterations (int): The maximum number of optimization iterations.
        inertia_weight (float): The inertia weight for velocity update.
        cognitive_weight (float): The cognitive weight for velocity update.
        social_weight (float): The social weight for velocity update.
        search_space (list of tuples): The search space for particle positions.
        best_global_position (list): The best global position found during
            optimization.
        best_global_value (float): The value of the objective function at
            the best global position.
        particles (list of dict): List of particle dictionaries.

    Methods:
        initialize_particles(): Initialize particles with random positions and
            velocities.
        optimize(objective_function): Optimize the given objective function
            using PSO.
        update_velocity(velocity, current_position, best_personal_position,
            best_global_position): Update particle velocity using PSO
            equations.
        update_position(current_position, velocity): Update particle position
            based on velocity.
    """

    def __init__(
        self,
        num_particles: int,
        max_iterations: int,
        inertia_weight: float,
        cognitive_weight: float,
        social_weight: float,
        search_space: List[Tuple[float, float]],
    ) -> None:
        self.num_particles: int = num_particles
        self.max_iterations: int = max_iterations
        self.inertia_weight: float = inertia_weight
        self.cognitive_weight: float = cognitive_weight
        self.social_weight: float = social_weight
        self.search_space: List[Tuple[float, float]] = search_space
        self.best_global_position: List[float] = [0.0 for _ in search_space]
        self.best_global_value = float('inf')
        self.particles: List[Dict[str, float | List[float]]] = []

    def initialize_particles(self) -> None:
        """
        Initialize particles with random positions and velocities.
        """
        # clear current particles
        self.particles.clear()

        # generate each particle
        for _ in range(self.num_particles):
            initial_position = [
                random.uniform(min_val, max_val)
                for min_val, max_val in self.search_space
            ]
            self.particles.append(
                {
                    'position': initial_position,
                    'velocity': [0.0] * len(self.search_space),
                    'best_personal_position': initial_position,
                    'best_personal_value': float('inf'),
                }
            )

    def optimize(
        self,
        objective_function: BaseFunction,
    ) -> None:
        """
        Optimize the given objective function using Particle Swarm
        Optimization (PSO).

        Args:
            objective_function (BaseFunction): The objective function to be
                optimized.
        """
        self.initialize_particles()

        for i in range(self.max_iterations):
            best_global_value: float = self.best_global_value
            best_global_position: List[float] = self.best_global_position

            for particle in self.particles:
                # get particle data
                current_position: List[float] = cast(
                    List[float], particle['position']
                )
                best_personal_position: List[float] = cast(
                    List[float], particle['best_personal_position']
                )
                velocity: List[float] = cast(List[float], particle['velocity'])

                # update particle velocity and position
                new_vel = self.update_velocity(
                    velocity,
                    current_position,
                    best_personal_position,
                    best_global_position,
                )
                new_pos = self.update_position(current_position, new_vel)

                # evaluate the objective function
                fit = objective_function.evaluate(new_pos)

                # update personal and global best
                if fit < best_global_value:
                    particle['best_personal_value'] = fit
                    particle['best_personal_position'] = new_pos

                    if fit < self.best_global_value:
                        self.best_global_value = fit
                        self.best_global_position = new_pos

                particle['position'] = new_pos
                particle['velocity'] = new_vel

            print(
                f'[{i + 1}] current min value: {self.best_global_value:.4f}',
                end='\r',
            )
        print()

    def update_velocity(
        self,
        velocity: List[float],
        current_position: List[float],
        best_personal_position: List[float],
        best_global_position: List[float],
    ) -> List[float]:
        """
        Update particle velocity using PSO equations.

        Args:
            velocity (list): Current velocity of the particle.
            current_position (list): Current position of the particle.
            best_personal_position (list): Best personal position of the
                particle.
            best_global_position (list): Best global position of the swarm.

        Returns:
            list: Updated velocity of the particle.
        """
        new_vel = []

        for vel, pos, p_best, g_best in zip(
            velocity,
            current_position,
            best_personal_position,
            best_global_position,
        ):
            cognitive_component = (
                self.cognitive_weight * random.random() * (p_best - pos)
            )
            social_component = (
                self.social_weight * random.random() * (g_best - pos)
            )
            inertia_component = self.inertia_weight * vel
            new_vel.append(
                inertia_component + cognitive_component + social_component
            )

        return new_vel

    def update_position(
        self, current_position: List[float], velocity: List[float]
    ) -> List[float]:
        """
        Update particle position based on velocity.

        Args:
            current_position (list): Current position of the particle.
            velocity (list): Particle velocity.

        Returns:
            list: Updated position of the particle.
        """
        new_pos = [x + v for x, v in zip(current_position, velocity)]
        return [
            min(max(x, x_min), x_max)
            for x, (x_min, x_max) in zip(new_pos, self.search_space)
        ]
