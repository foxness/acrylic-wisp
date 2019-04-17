import numpy as np
import random
import math
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def get_data():
    df = pd.read_csv('covertype')
    data = df.values

    x_data = data[:, :-1].astype(np.float32)
    y_data = data[:, -1].reshape(-1, 1)

    # DATA NORMALIZATION
    # (make every data point be in range [0, 1])
    for i in range(10):
        a = x_data[:, i]
        min_ = a.min()
        max_ = a.max()
        interval_ = max_ - min_
        a -= min_
        a /= interval_

    # for i, v in enumerate(df.columns.values[:-1]):
    #     a = x_data[:, i]
    #     print("{}. {} - max: {} - min: {} - mean: {}".format(i, v, a.max(), a.min(), a.mean()))

    # np.set_printoptions(suppress=True)
    le = OneHotEncoder(sparse=False)
    le.fit(y_data)

    y_data = [y.reshape(-1, 1) for y in le.transform(y_data)]
    x_data = [x.reshape(-1, 1) for x in x_data]

    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.25, random_state=42)

    training_data = list(zip(X_train, y_train))
    test_data = list(zip(X_test, y_test))

    return (training_data, test_data)

def main():
    seed = 1448
    np.random.seed(seed)
    random.seed(seed)

    training_data, test_data = get_data()

    print("\ndata stats:\n")
    display_data_stats(training_data)

    architecture = [54, 12, 8, 7]
    epoch_count = 60
    minibatch_size = 2
    learning_rate = 1.0

    print("\nnetwork architecture: {}".format(repr(architecture)))
    print("epoch count: {}".format(epoch_count))
    print("minibatch size: {}".format(minibatch_size))
    print("learning rate: {}\n".format(learning_rate))

    network = Network(architecture)
    print("Score before training: {} / {}".format(network.evaluate(test_data), len(test_data)))

    network.SGD(training_data, epoch_count, minibatch_size, learning_rate, test_data = test_data)

def display_data_stats(data):
    print("data point count: {}".format(len(data)))
    
    x, y = data[0]
    print("feature count: {}".format(x.shape[0]))

    a = np.array([a[0] for a in data]).squeeze()
    print("max: {}".format(a.max()))
    print("min: {}".format(a.min()))
    print("mean: {}".format(a.mean()))

def pr(x):
    print(repr(x))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

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
    
    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        
        for l in range(2, self.layer_count):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)
    
    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return (output_activations - y)

main()