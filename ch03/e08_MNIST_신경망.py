'''
MNIST 숫자 손글씨 데이터 신경망 구현

'''
import pickle

import numpy as np


from ch03.e01_Perceptron import sigmoid, step_function2
from ch03.e11_minibatch import softmax
from dataset.mnist import load_mnist


def init_network():
    ''' 가중치 행렬들(W1, W2, W3, b1, b2, b3)을 생성 '''

    # 교재의 저자가 만든 가중치 행렬(sample_weight.pkl)을 읽어 옴
    with open('sample_weight.pkl', mode = 'rb') as file:
        network = pickle.load(file)
    print(network.keys())
    # W1, W2,W3 ,b1, b2, b3 shape 확인
    return network


def predict(network, X_test):
    ''' 가중치 행렬들(W1, W2, W3, b1, b2, b3)을 생성
        '''

    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = X_test.dot(W1) + b1
    z1 = sigmoid(a1)
    z2 = sigmoid(z1.dot(W2) + b2)
    y = z2.dot(W3) + b3
    y = softmax(y)

    # print('y after softmax:', y)
    y = step_function2(y)
    # print('y after step:', y[0])

    return y


def accuracy(y_test, y_pred):
    ''' 신경망에서 사용되는 가중치 행렬들과 테스트 데이터를 파라미터로 전달받아서,
        테스트 데이터의 예측값(배열)을 리턴.
        '''

    # r = 0
    # for i, j in zip(y_test, y_pred):
    #     if np.array_equal(i, j):
    #         r += 1

    r = [1 for i, j in zip(y_test, y_pred) if np.array_equal(i, j)]

    return sum(r) / len(y_pred)


if __name__ == '__main__':
    network = init_network()
    (X_train, y_train), (X_test, y_test) = load_mnist(normalize = True,
                                                      flatten = True,
                                                      one_hot_label = True)

    # print(X_train[0])
    # print(y_train[0])

    # 신경망 가중치(와 편향, bias) 행렬들 생성
    network = init_network()
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    # print(f'W1: {W1.shape}, W2: {W2.shape}, W3: {W3.shape}')
    # print(f'b1: {b1.shape}, b2: {b2.shape}, b3: {b3.shape}')

    # 테스트 이미지들의 예측값
    y_pred = predict(network, X_test)
    print('y_pred[0] =', y_pred[0])
    print('y_test[0] =', y_test[0])

    # acc = accuracy(y_test, y_pred)
    acc = accuracy(y_test, y_pred)
    print('Accuracy :', acc)
