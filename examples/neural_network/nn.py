from natural_computing.neural_network import MultiLayerPerceptron

if __name__ == '__main__':
    # Example of using the Multi Layer Perceptron class by
    # process of optimizing xor problem.
    net = MultiLayerPerceptron(2, [128, 64], 2)

    # show network architecture
    print(f'- Layers:\n\t{[layer.shape for layer in net._layers]}\n')

    # create a simple xor dateset
    x = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = [0, 1, 1, 0]
    print(f'- Dataset: \n\tfeatures: {x}\n\ttargets: {y}')
