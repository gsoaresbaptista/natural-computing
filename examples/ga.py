from natural_computing.objective_functions import RastriginFunction
from natural_computing.optimization import RealGeneticAlgorithm

if __name__ == '__main__':
    # Example of using the Genetic Algorithm class by
    # process of optimizing a rastrigin function.
    ga = RealGeneticAlgorithm(3000, 500, [(-5.12, 5.12) for _ in range(5)])
    rastrigin_function = RastriginFunction(5)
    ga.optimize(rastrigin_function)
    print(ga.best_global_phenotype)
