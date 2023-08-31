import matplotlib.pyplot as plt
from natural_computing.objective_functions import RastriginFunction
from natural_computing.plotting import Plotter

if __name__ == '__main__':
    """
    Simple example of plotting a 3d rastrigin function with
    two variables using the Plotter class.
    """
    function = RastriginFunction(2)
    axes = Plotter.plot_3d_curve(function, [-8, 8], [-8, 8], 50)
    plt.show()
