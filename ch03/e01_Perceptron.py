'''
Perceptron:
    - 입력: (x1, x2)
    - 출력:
        a = x1 * w1 + x2 * w2 + b 계산 (w1 w2 가중치, b bias)
        y = 1, a > 임계값
          = 0 a <= 임계값

신경망의 뉴런(neuron)에서는 입력 신호의 가중치 합을 출력값으로 변환해 주는 함수가 존재
-> 활성화 함수(activation function)

'''
import math

import numpy as np
import matplotlib.pyplot as plt


def step_function(x):
    ''' Step Function.

        :param x: numpy.ndarray
        :return: step(계단) 함수 출력(0 또는 1)로 이루어진 numpy.ndarray
        '''

    # r = [1 if i > 0 else 0 for i in x]
    # return np.array(r)

    y = x > 0  # [False False False False  True  True  True]
    return y.astype(np.int)  # [0 0 0 0 1 1 1]


def step_function2(x):
    y = x > .446  # [False False False False  True  True  True]
    return y.astype(np.int)  # [0 0 0 0 1 1 1]


def sigmoid(x):
    ''' sigmoid = 1 / (1 + exp(-x))
        '''

    return 1 / (1 + np.exp(-x))


def relu(x):
    ''' ReLU(Rectified Linear Unit)
        y = x, if x > 0
          = 0, otherwise
        '''

    return np.maximum(0, x)


if __name__ == '__main__':
    x = np.arange(-3, 4)
    print('x =', x)
    # for x_i in x:
    #     print(step_function(x_i), end = ' ')
    # print()

    print('y =', step_function(x))  # [0 0 0 0 1 1 1]
    print('y =', sigmoid(x))

    # step 함수, sigmoid 함수를 하나의 그래프에 출력
    x = np.linspace(-5, 5, 100)
    steps = step_function(x)
    sigmoid = sigmoid(x)

    plt.plot(x, steps, label = 'Step function')
    plt.plot(x, sigmoid, label = 'Sigmoid function')
    plt.legend()
    plt.show()

    x = np.arange(-3, 4)
    print('x =', x)
    relus = relu(x)
    print('relu =', relus)
    plt.plot(x, relus)
    plt.title('ReLU')
    plt.show()
