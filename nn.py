import numpy as np


class NeuralNetwork:

    def __init__(self, layer_sizes):
        """
        Neural Network initialization.
        Given layer_sizes as an input, you have to design a Fully Connected Neural Network architecture here.
        :param layer_sizes: A list containing neuron numbers in each layers. For example [3, 10, 2] means that there are
        3 neurons in the input layer, 10 neurons in the hidden layer, and 2 neurons in the output layer.
        """
        self.weights = list()
        self.b = list()
        self.layer_sizes = layer_sizes
        for i in range(0, len(layer_sizes)):
            if i == 0:
                continue
            last_one = layer_sizes[i - 1]
            self.weights.append(np.random.normal(size=(last_one, layer_sizes[i])))
            self.b.append(np.zeros((1, last_one)).astype(np.longdouble))

    def activation(self, x):
        """
        The activation function of our neural network, e.g., Sigmoid, ReLU.
        :param x: Vector of a layer in our network.
        :return: Vector after applying activation function.
        """
        z = np.exp(-x)
        sig = 1 / (1 + z)
        return sig

    def forward(self, x):
        """
        Receives input vector as a parameter and calculates the output vector based on weights and biases.
        :param x: Input vector which is a numpy array.
        :return: Output vector
        """
        weights = self.weights
        b = self.b
        temp = self.activation(weights[0].T.dot(x) + b[0].T)
        for i in range(1, len(self.layer_sizes)):
            temp = self.activation(weights[i].T.dot(temp) + b[i].T)
        return temp
