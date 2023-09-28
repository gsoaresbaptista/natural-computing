from natural_computing.objective_functions import RastriginFunction
from natural_computing.optimization import EvolutionStrategy

if __name__ == '__main__':
    # Example of using the Evolution Strategy Optimization class by
    # process of optimizing a rastrigin function.
    rastrigin_function = RastriginFunction(5)
    de = EvolutionStrategy(
        20, 100, 0.15, 10000, [(-5.12, 5.12) for _ in range(5)], False
    )
    de.optimize(rastrigin_function)
    print(
        f'best position: {[round(p_i, 4) for p_i in de.best_global_position]}'
    )
