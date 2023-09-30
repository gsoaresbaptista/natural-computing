from natural_computing.objective_functions import RastriginFunction
from natural_computing.optimization import DifferentialEvolution

if __name__ == '__main__':
    # example of using the Differential Evolution Optimization class by
    # process of optimizing a rastrigin function.
    rastrigin_function = RastriginFunction(5)
    de = DifferentialEvolution(
        200, 1000, [(-5.12, 5.12) for _ in range(5)]
    )
    de.optimize(rastrigin_function)
    print(
        f'best position: {[round(p_i, 4) for p_i in de.best_global_position]}'
    )
