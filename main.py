import numpy as np
import random

np.random.seed(1448)

def pr(x):
    print(repr(x))

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

# class Network:
#     def __init__(self):
#         pass
    
#     def calculate_neuron(a, w, b):
        

def main():
    pr(generate_data(10))

main()