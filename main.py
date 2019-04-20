import numpy as np
import random
import math
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from network import Network
import torch as tr

def get_data():
    df = pd.read_csv('covertype')
    data = df.values

    x_data = data[:, :-1].astype(np.float32)
    y_data = data[:, -1].reshape(-1, 1)

    for i in range(10):
        x_data[:, i] = (x_data[:, i] - x_data[:, i].mean()) / x_data[:, i].std()

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

def grad_example(x, i):
    x.requires_grad_(True)
    ex = tr.exp(x)
    sm = ex[i] / ex.sum()
    res = sm.log()
    res.backward()
    
    x.requires_grad_(False)
    grad = x.grad.clone()
    x.grad.zero_()
    return [res, grad]

def testy_main():
    x = [0., 1, 2, 3]
    i = 2

    x = tr.tensor(x)
    print("x: {}".format(x))
    res, grad = grad_example(x, i)
    print("x: {}\nres: {}\ngrad: {}".format(x, res, grad))

    xpg = x + grad
    print("\nxpg: {}".format(xpg))
    res, grad = grad_example(xpg, i)
    print("x + grad: {}\nres: {}".format(xpg, res))

def mnist_main():
    seed = 1448
    tr.manual_seed(seed)
    random.seed(seed)
    print("asd")

    import mnist_loader
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

    network = Network([784, 30, 10])
    print("Before training evaluation: {} / {}".format(network.evaluate(test_data), len(test_data)))
    network.SGD(training_data, 30, 10, 3.0, test_data=test_data)

def main():
    seed = 1448
    np.random.seed(seed)
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

mnist_main()