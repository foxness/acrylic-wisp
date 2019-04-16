import numpy as np
import random
import math

np.random.seed(1448)

def pr(x):
    print(repr(x))

def relu(x):
    return np.max(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def activation(x):
    return sigmoid(x)

def generate_data(n):
    a = np.random.randn(n, 1)
    b = np.random.randn(n, 1)
    c = np.random.randn(n, 1)

    X = np.hstack((a, b, c))
    Y = np.array([x[0] + x[1] * x[2] for x in X])

    return {'x': X, 'y': Y}

class Network:
    def __init__(self, architecture):
        self.architecture = architecture
        self.layer_count = len(architecture)
        self.weights = [np.random.randn(count, previous_count) for previous_count, count in zip(self.architecture[:-1], self.architecture[1:])]
        self.biases = [np.random.randn(count, 1) for count in self.architecture[1:]]
        
    def feedforward(self, a):
        for w, b in zip(self.weights, self.biases):
            a = activation(np.dot(w, a) + b)
        return a
    
    def cost(self, X, realY):
        Y = self.feedforward(X)
        return ((Y - realY) ** 2).sum()

def main():
    data = generate_data(10)

    # network architecture
    # X: o o o
    #    o o o
    #    o o o
    # Y:   o

    network = Network([3, 3, 3, 1])

    for i in range(len(data['x'])):
        X = np.array(data['x'][i]).reshape(-1, 1)
        Y = np.array(data['y'][i]).reshape(-1, 1)
        cost = network.cost(X, Y)
        pr(cost)

main()