import numpy as np
import pandas as pd
import torch as tr
import random
import time

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

def get_data(seed = None):
    dataframe = pd.read_csv('covertype')
    data = dataframe.values

    x_data = tr.tensor(data[:, :-1].astype(np.float32))
    y_data = data[:, -1].reshape(-1, 1)

    for i in range(10):
        x_data[:, i] = (x_data[:, i] - x_data[:, i].mean()) / x_data[:, i].std()

    le = OneHotEncoder(sparse=False)
    le.fit(y_data)

    y_data = tr.tensor(le.transform(y_data)).float()

    y_data = [y.reshape(-1, 1) for y in y_data]
    x_data = [x.reshape(-1, 1) for x in x_data]

    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.25, random_state = seed)

    training_data = list(zip(X_train, y_train))
    test_data = list(zip(X_test, y_test))
    feature_names = le.get_feature_names()
    
    return (training_data, test_data, feature_names)

def report(network, test_data, feature_names):
    test_results = tr.tensor([(tr.argmax(network.feedforward(x.to(device = network.device))), tr.argmax(y.to(device = network.device))) for (x, y) in test_data])
    y_pred = test_results[:, 0]
    y_true = test_results[:, 1]
    return classification_report(y_true, y_pred, target_names = feature_names)

def relu(x):
    return tr.max(x, tr.tensor(0., device = x.device))

# def sigmoid(x):
#     return 1 / (1 + tr.exp(-x))

def softmax(x):
    e = tr.exp(x - x.max(dim = 0)[0])
    return e / e.sum(dim = 0)

def logsoftmax(x):
    return tr.log_softmax(x, dim = 0, dtype = tr.float32)

def batchify(unbatchified, device = None):
    x_batch = []
    y_batch = []
    for datapoint in unbatchified:
        x, y = datapoint
        x_batch.append(x.tolist())
        y_batch.append(y.tolist())
    
    return [m.squeeze().t() for m in [tr.tensor(x_batch, device = device), tr.tensor(y_batch, device = device)]]

class Network:
    def __init__(self, architecture, use_gpu = False):
        self.device = tr.device('cuda' if use_gpu and tr.cuda.is_available() else 'cpu')
        
        self.architecture = architecture
        self.layer_count = len(architecture)
        self.weights = [tr.randn(count, previous_count, device = self.device) for previous_count, count in zip(self.architecture[:-1], self.architecture[1:])]
        self.biases = [tr.zeros(count, 1, device = self.device) for count in self.architecture[1:]]
        self.activation_funcs = [relu] * (self.layer_count - 2) + [logsoftmax]
        
    def feedforward(self, a):
        for w, b, af in zip(self.weights, self.biases, self.activation_funcs):
            a = af(tr.mm(w, a) + b)
        return a
    
    def loss(self, x, y_real):
        y = self.feedforward(x)
        batch_size = x.shape[1]
        return -((y_real * tr.log(y)).sum() / batch_size)
    
    def loss_no_log(self, x, y_real):
        y = self.feedforward(x)
        batch_size = x.shape[1]
        return -((y_real * y).sum() / batch_size)
    
    def SGD(self, training_data, epoch_count, batch_size, learning_rate, test_data = None):
        for j in range(epoch_count):
            random.shuffle(training_data)
            
            for batch in (batchify(training_data[k:k + batch_size], device = self.device) for k in range(0, len(training_data), batch_size)):
                self.learn_from_batch(batch, learning_rate)
            
            if test_data:
                print("Epoch {}: {}".format(j, self.evaluation_string(test_data)))
            else:
                print("Epoch {0} complete".format(j))
    
    def learn_from_batch(self, batch, learning_rate):
        x, y = batch
        
        for w, b in zip(self.weights, self.biases):
            w.requires_grad_(True)
            b.requires_grad_(True)

        loss = self.loss_no_log(x, y)
        loss.backward()

        for w, b in zip(self.weights, self.biases):
            w.requires_grad_(False)
            b.requires_grad_(False)
        
        for w, b in zip(self.weights, self.biases):
            w -= learning_rate * w.grad
            b -= learning_rate * b.grad
        
        for w, b in zip(self.weights, self.biases):
            w.grad.zero_()
            b.grad.zero_()
    
    def evaluation_string(self, test_data):
        x, y_real = batchify(test_data, device = self.device)
        y = self.feedforward(x)
        
        batch_size = x.shape[1]
        loss = -((y_real * y).sum() / batch_size)
        correct_count = (y.argmax(dim = 0) == y_real.argmax(dim = 0)).sum().item()
        percentage = correct_count / batch_size * 100
        
        return "{} / {} ({:.2f}%) (loss {})".format(correct_count, batch_size, percentage, loss)

def main():
    seed = 1234
    np.random.seed(seed)
    random.seed(seed)
    tr.manual_seed(seed)
    
    training_data, test_data, feature_names = get_data(seed)

    use_gpu = False
    architecture = [54, 16, 16, 7]

    network = Network(architecture, use_gpu)
    print("Score before training: {}".format(network.evaluation_string(test_data)))
    print()

    epoch_count = 10
    batch_size = 1024
    learning_rate = 0.01
    test = test_data

    start_time = time.time()

    network.SGD(training_data, epoch_count, batch_size, learning_rate, test_data = test)

    stop_time = time.time()
    elapsed_time = stop_time - start_time

    print()
    print("done in {:.3f} s".format(elapsed_time))
    print()
    print(report(network, test_data, feature_names))

main()