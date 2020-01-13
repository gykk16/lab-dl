import numpy as np
import matplotlib.pyplot as plt

from ch05.e10_twolayer import TwoLayerNetwork
from ch06.e05_Adam import Adam
from dataset.mnist import load_mnist

if __name__ == '__main__':
    (X_train, Y_train), (X_test, Y_test) = load_mnist()

    neural_net = TwoLayerNetwork(input_size = 784,
                                 hidden_size = 32,
                                 output_size = 10)
    train_losses = []

    iterations = 2_000
    batch_size = 128
    train_size = X_train.shape[0]

    np.random.seed(111)
    adam = Adam()
    for i in range(iterations):
        batch_mask = np.random.choice(train_size, batch_size)
        X_batch = X_train[batch_mask]
        Y_batch = Y_train[batch_mask]

        gradients = neural_net.gradient(X_batch, Y_batch)
        adam.update(neural_net.params, gradients)

        loss = neural_net.loss(X_batch, Y_batch)
        train_losses.append(loss)

        if i % 100 == 0:
            print()
            print(f'====== training # {i} =======')
            print(train_losses[-1])
