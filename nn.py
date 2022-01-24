from copy import deepcopy
from random import randint

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
            size = (last_one, layer_sizes[i])
            self.weights.append(np.random.normal(size=size))
            self.b.append(np.zeros((1, layer_sizes[i])).astype(np.longdouble))

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
        for i in range(1, len(self.layer_sizes) - 1):
            temp = self.activation(weights[i].T.dot(temp) + b[i].T)
        return temp

    def cross_over(self, p2, layer_num, point):

        self_layer = deepcopy(self.weights[layer_num].T)
        other_layer = deepcopy(p2.nn.weights[layer_num].T)
        self_bias = deepcopy(self.b[layer_num].T)
        other_bias = deepcopy(p2.nn.b[layer_num].T)
        st = self_layer
        ot = other_layer
        sb = self_bias
        ob = other_bias
        new1 = deepcopy(st)
        new2 = deepcopy(ot)
        new1b = deepcopy(sb)
        new2b = deepcopy(ob)
        for i in range(point, len(st)):
            new1[i] = ot[i]
            new1b[i] = ob[i]
        for i in range(point, len(ot)):
            new2[i] = st[i]
            new2b[i] = sb[i]
        self.weights[layer_num] = new1.T
        p2.nn.weights[layer_num] = new2.T
        self.b[layer_num] = new1b.T
        p2.nn.b[layer_num] = new2b.T

    def mutate(self):
        layer_number = randint(0, len(self.weights) - 1)
        perceptron_number = randint(0, self.weights[layer_number].shape[1] - 1)
        weights = self.weights[layer_number][:, [perceptron_number]]
        b = self.b[layer_number][:, [perceptron_number]]

        weights = np.random.normal(size=weights.shape) / 10
        b = np.random.normal(size=b.shape) / 10

        self.weights[layer_number][:, [perceptron_number]] = weights
        self.b[layer_number][:, [perceptron_number]] = b
