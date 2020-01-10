import numpy as np


def and_gate(x):
    # x는 [0, 0], [0, 1], [1, 0], [1, 1] 중 하나인 numpy.ndarray 타입
    # w = [w1, w2]인 numpy.ndarray 가중치와 bias b를 찾음
    w = np.array([1, 1])  # weight
    b = 1  # bias
    test = x.dot(w) + b  # np.sum(x * w) + b
    if test > 2:
        return 1
    else:
        return 0


def nand_gate(x):
    w = np.array([1, 1])
    b = 1
    test = x.dot(w) + b
    if test <= 2:
        return 1
    else:
        return 0


def or_gate(x):
    w = np.array([1, 1])
    b = 1
    test = x.dot(w) + b
    if test >= 2:
        return 1
    else:
        return 0


def test_perceptron(perceptron):
    for x1 in (0, 1):
        for x2 in (0, 1):
            x = np.array([x1, x2])
            result = perceptron(x)
            print(x, '->', result)


if __name__ == '__main__':
    print('\nAND:')
    test_perceptron(and_gate)
    print('\nNAND:')
    test_perceptron(nand_gate)
    print('\nOR:')
    test_perceptron(or_gate)
