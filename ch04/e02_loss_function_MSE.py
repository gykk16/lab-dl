import pickle

import numpy as np

from ch03.e11_teacher import forward
from dataset.mnist import load_mnist

if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = load_mnist(one_hot_label = True)

    y_true = y_test[:10]  # (10, 10)
    print('y_true =', y_true)

    with open('../ch03/sample_weight.pkl', mode = 'rb') as f:
        network = pickle.load(f)

    y_pred = forward(network, X_test[:10])
    print('y_pred =', y_pred)

    print(y_true[0])
    print(y_pred[0])

    error = y_pred[0] - y_true[0]
    print(error)

    print(error ** 2)
    print(np.sum(error ** 2))

    print('y_true[8] =', y_true[8])
    print('y_pred[8] =', y_pred[8])
    print(np.sum((y_true[8] - y_pred[8])**2))