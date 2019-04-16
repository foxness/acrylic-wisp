import numpy as np
import random
import math

np.random.seed(1448)

def pr(x):
    print(repr(x))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def generate_data(n):
    a = (np.random.rand(n, 1) - 0.5) * 100
    b = (np.random.rand(n, 1) - 0.5) * 100
    c = (np.random.rand(n, 1) - 0.5) * 100

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

        self.weights = np.random.rand(count, previous_count) - 0.5
        self.biases = np.random.rand(count, 1) - 0.5
    
    def calculate(self, previous):
        return sigmoid(np.matmul(self.weights, previous) + self.biases)

class Network:
    def __init__(self):

        # network architecture
        # X: o o o
        #    o o o
        #    o o o
        # Y:   o

        self.layers = [Layer(3, 3), Layer(3, 3), Layer(3, 1)]
        
    def feedforward(self, X):
        current = X.reshape(-1, 1)

        for layer in self.layers:
            current = layer.calculate(current)
        
        return current

def main():
    data = generate_data(10)

    network = Network()
    networkY = network.feedforward(data['x'][0])
    print(networkY)

main()