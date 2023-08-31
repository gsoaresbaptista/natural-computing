from natural_computing.optimization import ParticleSwarmOptimization
from natural_computing.objective_functions import RastriginFunction

if __name__ == '__main__':
    rastrigin_function = RastriginFunction(3)
    pso = ParticleSwarmOptimization(
        100, 1000, 1, 2, 2, [[-8, 8], [-8, 8], [-8, 8]]
    )
    pso.optimize(rastrigin_function)
    print(
        f'best position: {[round(p_i, 4) for p_i in pso.best_global_position]}'
    )
