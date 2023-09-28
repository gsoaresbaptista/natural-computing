import numpy as np
from natural_computing.neural_network import MultiLayerPerceptron

if __name__ == '__main__':
    # example of using the multi layer perceptron class in
    # the xor problem classification process.
    net = MultiLayerPerceptron(2, [128, 64], 2)

    # show network architecture
    print(f'- Layers:\n\t{[layer.shape for layer in net._layers]}\n')

    # create a simple xor dateset
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])
    print(f'- Dataset: \n\tfeatures: {x}\n\ttargets: {y}\n')

    # test forward phase
    output = net.forward(x)
    print(f'- Forward output:\n{output}')
