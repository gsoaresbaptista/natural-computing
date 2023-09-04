from typing import List

import matplotlib.pyplot as plt
import pandas as pd

from natural_computing.objective_functions import SquaredError
from natural_computing.optimization import BareBonesParticleSwarmOptimization
from natural_computing.plotting import polynomial_plot


def read_txt(path: str) -> List[float]:
    return pd.read_csv(path, header=None).T.values.tolist()[0]


if __name__ == '__main__':
    # Example of using optimizers to find polynomial coefficients.
    x_data = read_txt('resources/x_data.txt')
    y_data = read_txt('resources/y_data.txt')

    objective_function = SquaredError(x_data, y_data, mean=True)
    pso = BareBonesParticleSwarmOptimization(
        80, 300, [(-5, 5) for _ in range(3)]
    )
    pso.optimize(objective_function)
    polynomial_plot(pso.best_global_position, (-5, 5), 100, x_data, y_data)
    plt.show()
