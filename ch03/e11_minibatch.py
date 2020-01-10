'''
mini-batch

'''
import pickle
import time

import numpy as np

from ch03.e01_Perceptron import sigmoid
from ch03.e08_MNIST_신경망_teacher import accuracy
from dataset.mnist import load_mnist



def softmax(X):
    '''
    1) X 가 1차원 : [x_1, x_2, ... , x_n]
    2) X 가 2차원 : [[x_11, x_12, ... , x_1n],
                    [x_21, x_22, ... , x_2n],
                     ... ]

    '''

    dimension = X.ndim
    if dimension == 1:
        m = np.max(X)  # 1차원 배열의 최댓값을 찾음
        X = X - m  # 0 이하의 숫자로 변환 <- exp 함수의 overflow 를 방지하기 위해
        y = np.exp(X) / np.sum(np.exp(X))

    elif dimension == 2:
        # m = np.max(X, axis = 1).reshape(len(X), 1)
        # X = X - m
        # y = np.exp(X) / np.sum(np.exp(X), axis = 1).reshape(len(X), 1)

        X_t = X.T  # X의 전치 행렬(transpose)
        m = np.max(X_t, axis=0)
        X_t = X_t - m
        y = np.exp(X_t) / np.sum(np.exp(X_t), axis=0)
        y = y.T

    return y


def init_network():
    ''' 가중치 행렬들(W1, W2, W3, b1, b2, b3)을 생성 '''

    # 교재의 저자가 만든 가중치 행렬(sample_weight.pkl)을 읽어 옴
    with open('sample_weight.pkl', mode = 'rb') as file:
        network = pickle.load(file)
    # print(network.keys())
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

    y_pred = np.argmax(y, axis = 1)

    return y_pred


def mini_batch(network, X_test, batch_size):
    # batch_starts = [i for i in range(0, len(X_test), batch_size)]
    # print('batch_starts :', batch_starts)

    r = np.empty(0)
    # print(r)

    for i in range(0, len(X_test), batch_size):
        X_batch = X_test[i:i + batch_size]
        X_batch_pred = predict(network, X_batch)
        # print(X_batch_pred)
        r = np.r_[r, X_batch_pred]

    print(r)
    return r


if __name__ == '__main__':
    np.random.seed(2020)

    # 1차원 softmax 테스트
    a = np.random.randint(10, size = 5)
    print(a)
    print(softmax(a))

    # 2차원 softmax 테스트
    A = np.random.randint(10, size = (2, 3))
    print(A)
    print(softmax(A))

    #############################
    print('\n\n')
    #############################

    # (Train/Test) 데이터 세트 로드.
    (X_train, y_train), (X_test, y_test) = load_mnist()
    # print(X_test[0])

    # 신경망 생성 (W1, b1, ...)
    network = init_network()
    batch_size = 100

    start_time = time.time()

    y_pred = mini_batch(network, X_test, batch_size)
    print('예측값 :', y_pred.shape)

    # 정확도(accuracy) 출력
    acc = accuracy(y_test, y_pred)
    print('Accuracy :', acc)








print('time elapsed: {:.2f}s'.format(time.time() - start_time))