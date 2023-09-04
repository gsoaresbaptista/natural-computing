from natural_computing.objective_functions import RastriginFunction
from natural_computing.optimization import BareBonesParticleSwarmOptimization

if __name__ == '__main__':
    # Example of using the Particle Swarm Optimization class by
    # process of optimizing a rastrigin function.
    rastrigin_function = RastriginFunction(5)
    pso = BareBonesParticleSwarmOptimization(
        80, 10000, [(-5.12, 5.12) for _ in range(5)]
    )
    pso.optimize(rastrigin_function)
    print(
        f'best position: {[round(p_i, 4) for p_i in pso.best_global_position]}'
    )
