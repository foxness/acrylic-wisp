import numpy as np
import random
import math

np.random.seed(1448)

def pr(x):
    print(repr(x))

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def generate_data(n):
    a = (np.random.rand(n, 1) - 0.5) * 100
    b = (np.random.rand(n, 1) - 0.5) * 100
    c = (np.random.rand(n, 1) - 0.5) * 100

    X = np.hstack((a, b, c))
    Y = np.array([x[0] + x[1] * x[2] for x in X])

    return {'x': X, 'y': Y}

# network architecture
# X: o o o
#    o o o
#    o o o
# Y:   o

class Neuron:
    def __init__(self, layer = None, weights = None, bias = None, calculated = None):
        self.layer = layer
        self.weights = weights
        self.bias = bias
        self.calculated = calculated
    
    def calculate(self):
        if self.calculated != None:
            return self.calculated
        
        layer = np.array([a.calculate() for a in self.layer])
        return sigmoid(np.dot(layer, self.weights) + self.bias)


# class Network:
#     def __init__(self):
#         pass
    
#     def calculate_neuron(a, w, b):
        

def main():
    pr(generate_data(10))

main()