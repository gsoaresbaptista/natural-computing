"""
Genetic Algorithm
    This module implements the Genetic Algorithm optimization algorithm.

Classes:
    BinaryGeneticAlgorithm: An implementation of the Genetic Algorithm for
        binary encoding.

    RealGeneticAlgorithm: An implementation of the Genetic Algorithm for
        real-valued encoding.
"""

import random
from typing import List, Tuple

from natural_computing.objective_functions import BaseFunction
from natural_computing.utils import (binary_ieee754_to_float,
                                     float_to_binary_ieee754, inverse_binary)

from .base_optimizer import PopulationBaseOptimizer


class BinaryGeneticAlgorithm(PopulationBaseOptimizer):
    """
    Binary Genetic Algorithm (GA) optimizer for solving optimization problems.

    This class represents a binary genetic algorithm for optimization problems.
    It initializes a population of binary genomes, evolves them over multiple
    generations, and finds the optimal solution within the specified search
    space.

    Args:
        population_size (int): The size of the population.
        max_generations (int): The maximum number of generations to run the
            algorithm.
        search_space (List[Tuple[float, float]]): A list of tuples specifying
            the search space bounds for each parameter. Each tuple contains
            the minimum and maximum values for a parameter.
        mutation_probability (float, optional): The probability of mutation
            for each gene during evolution. Defaults to 0.01.

    Attributes:
        population_size (int): The size of the population.
        population (List[str]): A list of binary genomes representing
            individuals in the population.
        fits (List[float]): A list of fitness values corresponding to the
            individuals in the population.
        best_global_genome (List[str]): The binary genome representing the
            best solution found so far.
        mutation_probability (float): The probability of mutation for each
            gene.

    Methods:
        initialize_population(self) -> None:
            Initialize individuals with random values.

        _fit_population(self, objective_function: BaseFunction) -> None:
            Calculate and store fitness values for the current population.

        _select_fathers(self) -> List[int]:
            Select two fathers based on fitness.

        _mutation(
            self, child: List[str], mutation_probability: float
        ) -> List[str]:
            Perform mutation on a child genome.

        _crossover(
            self, parent_0: List[str], parent_1: List[str]
        ) -> List[List[str]]:
            Perform crossover between two parents.

        _optimization_step(
            self, objective_function: BaseFunction
        ) -> List[float]:
            Objective function optimization step provided using Binary Genetic
            Algorithm Optimization (BGA).

        best_global_phenotype(self) -> List[float]:
            Returns the decoded phenotype of the best global genome as a list
            of floating-point values.
    """

    def __init__(
        self,
        population_size: int,
        max_generations: int,
        search_space: List[Tuple[float, float]],
        mutation_probability: float = 0.01,
    ) -> None:
        super().__init__(max_generations, population_size, search_space)
        self.population_size = population_size
        self.population: List[List[str]] = []
        self.fits: List[float] = []
        self.best_global_genome: List[str] = []
        self.mutation_probability: float = mutation_probability
        self.initialize_population()

    @property
    def best_global_phenotype(self) -> List[float]:
        return [
            binary_ieee754_to_float(binary)
            for binary in self.best_global_genome
        ]

    def initialize_population(self) -> None:
        """
        Initialize individuals with random values.
        """
        # clear current population
        self.population.clear()

        # generate the first half of the individuals
        for _ in range(self.population_size // 2):
            genome = [
                float_to_binary_ieee754(
                    random.random() * (max_val - min_val) + min_val
                )
                for min_val, max_val in self.search_space
            ]
            self.population.append(genome)

        # generate the second half by flipping the bits
        for i in range(self.population_size // 2):
            genome = [
                inverse_binary(binary_float)
                for binary_float in self.population[i]
            ]
            self.population.append(genome)

    def _fit_population(self, objective_function: BaseFunction) -> None:
        """
        Calculate and store fitness values for the current population.

        Args:
            objective_function (BaseFunction): The objective function to
                evaluate fitness.
        """
        self.fits.clear()
        for genome in self.population:
            phenotype = [binary_ieee754_to_float(gene) for gene in genome]
            self.fits.append(objective_function.evaluate(phenotype))

    def _select_fathers(self) -> List[int]:
        """
        Select two fathers based on fitness.

        Returns:
            List[int]: List containing the indices of the selected fathers.
        """
        return [
            min(
                random.choices(range(self.population_size), k=2),
                key=lambda idx: self.fits[idx],
            )
            for _ in range(2)
        ]

    @staticmethod
    def _mutation(child: List[str], mutation_probability: float) -> List[str]:
        """
        Perform mutation on a child.

        Args:
            child (List[str]): The child genome.
            mutation_probability (float): The probability for the mutation to
                occur.

        Returns:
            List[str]: The mutated child genome.
        """
        mutated_child = []
        for genome in child:
            mutated_genome = ''.join(
                [
                    bit
                    if random.random() > mutation_probability
                    else '1'
                    if bit == '0'
                    else '0'
                    for bit in genome
                ]
            )
            mutated_child.append(mutated_genome)
        return mutated_child

    @staticmethod
    def _crossover(
        parent_0: List[str], parent_1: List[str]
    ) -> List[List[str]]:
        """
        Perform crossover between two parents.

        Args:
            parent_0 (List[str]): The first parent's genome.
            parent_1 (List[str]): The second parent's genome.

        Returns:
            List[List[str]]: A list containing two child genomes.
        """
        children = [[], []]

        for i in range(len(parent_0)):
            # generate positions
            cut_i = random.randint(0, 30)
            cut_j = random.randint(cut_i, 31)

            # combine to get each child
            children[0].append(
                parent_0[i][:cut_i]
                + parent_1[i][cut_i:cut_j]
                + parent_0[i][cut_j:]
            )
            children[1].append(
                parent_1[i][:cut_i]
                + parent_0[i][cut_i:cut_j]
                + parent_1[i][cut_j:]
            )

        return children

    def _optimization_step(
        self,
        objective_function: BaseFunction,
    ) -> List[float]:
        """
        Objective function optimization step provided using Binary Genetic
            Algorithm Optimization (BGA).

        Args:
            objective_function (BaseFunction): The objective function to be
                optimized.

        Returns:
            List[float]: List of fitness values for the population after
                a optimization step.
        """
        if not self.fits:
            self._fit_population(objective_function)

        new_population = []

        for _ in range(self.population_size // 2):
            fathers = self._select_fathers()
            children = self._crossover(
                self.population[fathers[0]], self.population[fathers[1]]
            )
            mutated_children = [
                self._mutation(child, self.mutation_probability)
                for child in children
            ]
            new_population.extend(mutated_children)

        # compute the fitness of each individual
        self.population = new_population
        self._fit_population(objective_function)

        # elitism, remove the worst and put the best of the last generation
        current_max = max(self.fits)
        current_max_id = self.fits.index(current_max)
        self.population[current_max_id] = self.best_global_genome

        # update the best value in the population
        current_min = min(self.fits)
        if current_min < self.best_global_value:
            current_min_id = self.fits.index(current_min)
            self.best_global_value = current_min
            self.best_global_genome = self.population[current_min_id]

        return self.fits
