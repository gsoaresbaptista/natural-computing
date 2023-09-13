from typing import List

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from natural_computing.objective_functions import SquaredError
from natural_computing.optimization import BareBonesParticleSwarmOptimization
from natural_computing.plotting import polynomial_plot

SAVE_GIF = False


def update(frame):
    plt.clf()
    pso._optimization_step(objective_function)
    polynomial_plot(pso.best_global_position, (-5, 5), 100, x_data, y_data)
    plt.xlim([-6, 6])
    plt.ylim([-5, 75])


if __name__ == '__main__':
    # Example of using optimizers to find polynomial coefficients.
    x_data = np.linspace(-5, 5, 30)
    y_data = (2 * x_data**2 - 3 * x_data + 1).tolist()
    x_data: List[float] = x_data.tolist()

    objective_function = SquaredError(x_data, y_data, mean=True)
    pso = BareBonesParticleSwarmOptimization(
        20, 300, [(-50, 50) for _ in range(3)]
    )

    fig = plt.figure()
    pso._optimization_step(objective_function)
    polynomial_plot(pso.best_global_position, (-5, 5), 100, x_data, y_data)
    plt.xlim([-6, 6])
    plt.ylim([-5, 75])

    ani = animation.FuncAnimation(
        fig, update, frames=60, repeat=True, interval=300
    )

    if SAVE_GIF:
        writer = animation.PillowWriter(fps=10)
        ani.save('scatter.gif', writer=writer)
    else:
        plt.show()
