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

# class Neuron:
#     def __init__(self, layer = None, weights = None, bias = None, calculated = None):
#         self.layer = layer
#         self.weights = weights
#         self.bias = bias
#         self.calculated = calculated
    
#     def calculate(self):
#         if self.calculated != None:
#             return self.calculated
        
#         layer = np.array([a.calculate() for a in self.layer])
#         return sigmoid(np.dot(layer, self.weights) + self.bias)

class Layer:
    def __init__(self, previous_count, count):
        self.previous_count = previous_count
        self.count = count

        self.weights = np.random.randn(count, previous_count)
        self.biases = np.random.randn(count, 1)
    
    def calculate(self, previous):
        return activation(np.matmul(self.weights, previous) + self.biases)

class Network:
    def __init__(self, architecture):
        self.layers = []
        for i in range(len(architecture) - 1):
            self.layers.append(Layer(architecture[i], architecture[i + 1]))
        
    def feedforward(self, X):
        current = X.reshape(-1, 1)

        for layer in self.layers:
            current = layer.calculate(current)
        
        return current
    
    def cost(self, X, realY):
        Y = self.feedforward(X)
        print('Y: {}, realY: {}'.format(Y, realY))
        pr(Y - realY)
        return ((Y - realY) ** 2).sum()

def main():
    data = generate_data(10)

    # network architecture
    # X: o o o
    #    o o o
    #    o o o
    # Y:   o

    network = Network([3, 3, 3, 1])
    costs = [network.cost(data['x'][i], np.array([[data['y'][i]]])) for i in range(len(data['x']))]
    pr(costs)

main()