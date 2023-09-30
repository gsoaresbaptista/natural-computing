import numpy as np
from natural_computing.neural_network import (
    NeuralNetwork,
    softmax_neg_likelihood,
)
from natural_computing.utils import LayerFactory


if __name__ == '__main__':
    ...
    # example of using the neural network class in a simple regression problem.
    # note that weights were set manually for testing only and that is not
    # appropriate
    x = np.array([[0.1, 0.2, 0.7]])
    y = np.array([[1, 0, 0]])
    input_dim, output_dim = x.shape[1], y.shape[1]

    w1 = np.array([[0.1, 0.2, 0.3], [0.3, 0.2, 0.7], [0.4, 0.3, 0.9]])
    b1 = np.ones((1, 3))
    w2 = np.array([[0.2, 0.3, 0.5], [0.3, 0.5, 0.7], [0.6, 0.4, 0.8]])
    b2 = np.ones((1, 3))
    w3 = np.array([[0.1, 0.4, 0.8], [0.3, 0.7, 0.2], [0.5, 0.2, 0.9]])
    b3 = np.ones((1, 3))

    factory = LayerFactory()
    d1 = factory.dense_layer(input_dim, 3, activation='relu')
    d2 = factory.dense_layer(3, 3, activation='sigmoid')
    d3 = factory.dense_layer(3, output_dim, activation='linear')

    d1._weights = w1
    d1._biases = b1
    d2._weights = w2
    d2._biases = b2
    d3._weights = w3
    d3._biases = b3

    net = NeuralNetwork(0.01, loss_function=softmax_neg_likelihood)
    net.add_layer(d1)
    net.add_layer(d2)
    net.add_layer(d3)

    net.fit(x, y, epochs=300, verbose=30)

    for layer in net._layers:
        print(layer._weights)
