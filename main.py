import numpy as np
import random
import math

np.random.seed(1448)

def pr(x):
    print(repr(x))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

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
            a = sigmoid(np.dot(w, a) + b)
        return a
    
    def cost(self, x, realY):
        y = self.feedforward(x)
        return ((y - realY) ** 2).sum()

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data:
            n_test = len(test_data)

        n = len(training_data)

        for j in range(epochs):
            random.shuffle(training_data)

            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            
            if test_data:
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))
    
    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        
        self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

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