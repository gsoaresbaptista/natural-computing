import matplotlib.pyplot as plt
import numpy as np

from natural_computing.objective_functions import SquaredError
from natural_computing.optimization import BareBonesParticleSwarmOptimization
from natural_computing.plotting import polynomial_plot

if __name__ == '__main__':
    # Example of using optimizers to find polynomial coefficients.
    x_data = np.linspace(-5, 5, 30).tolist()
    y_data = (2 * x_data**2 - 3 * x_data + 1).tolist()

    objective_function = SquaredError(x_data, y_data, mean=True)
    pso = BareBonesParticleSwarmOptimization(
        80, 300, [(-5, 5) for _ in range(3)]
    )
    pso.optimize(objective_function)
    polynomial_plot(pso.best_global_position, (-5, 5), 100, x_data, y_data)
    plt.show()
