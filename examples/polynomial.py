from natural_computing.objective_functions import SquaredError
from natural_computing.optimization import BareBonesParticleSwarmOptimization
import pandas as pd
from typing import List


def read_txt(path: str) -> List[float]:
    return pd.read_csv(path, header=None).T.values.tolist()[0]


if __name__ == '__main__':
    # Example of using optimizers to find polynomial coefficients.
    x_data = read_txt('resources/x_data.txt')
    y_data = read_txt('resources/y_data.txt')

    objective_function = SquaredError(x_data, y_data, mean=True)
    pso = BareBonesParticleSwarmOptimization(
        80, 3000, [(-5, 5) for _ in range(3)]
    )
    pso.optimize(objective_function)
    print(pso.best_global_position)
    # print(
    #     f'best position: {[round(p_i, 4) for p_i in pso.best_global_position]}'
    # )
