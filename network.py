import numpy as np
import random
import torch as tr

def sigmoid(x):
    return 1 / (1 + tr.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

def softmax(x):
    e = tr.exp(x)
    return e / e.sum()

def get_score_string(correct, total):
    return "{} / {} ({}%)".format(correct, total, correct * 1.0 / total * 100)

class Network:
    def __init__(self, architecture):
        self.architecture = architecture
        self.layer_count = len(architecture)
        self.weights = [tr.randn(count, previous_count) for previous_count, count in zip(self.architecture[:-1], self.architecture[1:])]
        self.biases = [tr.randn(count, 1) for count in self.architecture[1:]]
        self.activation_funcs = [sigmoid] * (self.layer_count - 1) # [sigmoid] * (self.layer_count - 2) + [softmax]
        
    def feedforward(self, a):
        for w, b, af in zip(self.weights, self.biases, self.activation_funcs):
            a = af(tr.mm(w, a) + b)
        return a
    
    def cost(self, x, realY):
        y = self.feedforward(x)
        return ((y - realY) ** 2).sum()
    
    def backprop(self, x, y):
        for w, b in zip(self.weights, self.biases):
            w.requires_grad_(True)
            b.requires_grad_(True)

        loss = self.cost(x, y)

        loss.backward()

        for w, b in zip(self.weights, self.biases):
            w.requires_grad_(False)
            b.requires_grad_(False)

        nabla_w = [w.grad.clone() for w in self.weights]
        nabla_b = [b.grad.clone() for b in self.weights]

        for w, b in zip(self.weights, self.biases):
            w.grad.zero_()
            b.grad.zero_()

        return [nabla_w, nabla_b]
    
    def SGD(self, training_data, epoch_count, batch_size, learning_rate, test_data = None):
        n = len(training_data)

        for j in range(epoch_count):
            random.shuffle(training_data)

            batches = [training_data[k:k + batch_size] for k in range(0, n, batch_size)]
            for batch in batches:
                self.learn_from_batch(batch, learning_rate)
            
            if test_data:
                print("Epoch {}: {}".format(j, get_score_string(self.evaluate(test_data), len(test_data))))
            else:
                print("Epoch {0} complete".format(j))
    
    def learn_from_batch(self, batch, learning_rate):
        nabla_w = [tr.zeros(w.shape) for w in self.weights]
        nabla_b = [tr.zeros(b.shape) for b in self.biases]

        for x, y in batch:
            delta_nabla_w, delta_nabla_b = self.backprop(x, y)
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        
        self.weights = [w - (learning_rate / len(batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (learning_rate / len(batch)) * nb for b, nb in zip(self.biases, nabla_b)]
    
    def evaluate(self, test_data):
        test_results = [(tr.argmax(self.feedforward(x)), tr.argmax(y)) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return (output_activations - y)