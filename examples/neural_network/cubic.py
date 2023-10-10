import matplotlib.pyplot as plt
import numpy as np

from natural_computing import (
    NeuralNetwork,
    mse,
    make_cubic,
    LayerFactory,
)


if __name__ == '__main__':
    # example of using the neural network class to find a cubic equation
    x, y = make_cubic(320, -4, 4, 1, 0, -10, 0, 3)
    input_dim, output_dim = x.shape[1], y.shape[1]

    # min max scaling
    x_std = (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))
    x_std = 2 * x_std - 1

    # shuffle data
    indices = np.random.randint(0, x.shape[0], x.shape[0])
    x_shuffled, y_shuffled = x_std[indices], y[indices]
    reg = {'regularization_strength': 0.001}
    factory = LayerFactory()

    nn = NeuralNetwork(learning_rate=1e-3, loss_function=mse, momentum=0.9)
    nn.add_layer([
        factory.dense_layer(input_dim, 10, 'relu', **reg),
        factory.dense_layer(10, 10, 'relu', batch_normalization=True, **reg),
        factory.dense_layer(10, output_dim, 'linear', **reg),
    ])

    nn.fit(x_shuffled, y_shuffled, epochs=10000, batch_size=320, verbose=500)

    plt.scatter(x, y)
    plt.plot(x, nn.predict(x_std), c='green')
    plt.show()
