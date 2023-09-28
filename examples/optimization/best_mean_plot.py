import matplotlib.pyplot as plt

from natural_computing.objective_functions import RastriginFunction
from natural_computing.optimization import BareBonesParticleSwarmOptimization
from natural_computing.plotting import best_mean_plot

if __name__ == '__main__':
    # Simple example of plotting a history curve with the best value
    # of each iteration and the average value of each iteration
    rastrigin_function = RastriginFunction(5)
    bbpso = BareBonesParticleSwarmOptimization(
        80, 100, [(-5.12, 5.12) for _ in range(5)]
    )
    bbpso.optimize(rastrigin_function)
    best_mean_plot(bbpso)
    plt.show()
