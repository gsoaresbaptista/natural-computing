import numpy as np
from natural_computing.neural_network import (
    NeuralNetwork,
    Dense,
    sigmoid,
    softmax_neg_likelihood,
    linear,
    relu,
)

if __name__ == '__main__':
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

    d1 = Dense(input_size=input_dim, output_size=3, activation=relu)
    d2 = Dense(input_size=3, output_size=3, activation=sigmoid)
    d3 = Dense(input_size=3, output_size=output_dim, activation=linear)
    net = NeuralNetwork(0.01, softmax_neg_likelihood)
    net._layers.extend([d1, d2, d3])

    net._layers[0]._weights = w1
    net._layers[0]._biases = b1
    net._layers[1]._weights = w2
    net._layers[1]._biases = b2
    net._layers[2]._weights = w3
    net._layers[2]._biases = b3

    net.fit(x, y, epochs=300, verbose=30)

    for layer in net._layers:
        print(layer._weights)
