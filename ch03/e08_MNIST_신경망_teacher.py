'''
MNIST 숫자 손글씨 데이터 신경망 구현

'''
import pickle
import time

import numpy as np
from PIL import Image

from ch03.e01_Perceptron import sigmoid
from ch03.e05_neuralnetwork import softmax
from dataset.mnist import load_mnist




def init_network():
    ''' 가중치 행렬들(W1, W2, W3, b1, b2, b3)을 생성 '''

    # 교재의 저자가 만든 가중치 행렬(sample_weight.pkl)을 읽어 옴
    with open('sample_weight.pkl', mode = 'rb') as file:
        network = pickle.load(file)
    print(network.keys())
    # W1, W2,W3 ,b1, b2, b3 shape 확인
    return network


def forward(network, x):
    """
    순방향 전파(forward propagation).
    파라미터 x: 이미지 한 개의 정보를 가지고 있는 배열. (784,)
    """
    # 가중치 행렬(Weight Matrices)
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    # bias matrices
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    # 첫번째 은닉층
    a1 = x.dot(W1) + b1
    z1 = sigmoid(a1)

    # 두번째 은닉층
    a2 = z1.dot(W2) + b2
    z2 = sigmoid(a2)

    # 출력층
    a3 = z2.dot(W3) + b3
    y = softmax(a3)

    return y


def predict(network, X_test):
    ''' 신경망에서 사용되는 가중치 행렬들과 테스트 데이터를 파라미터로 전달받아서,
        테스트 데이터의 예측값(배열)을 리턴.
        파라미터 X_test: 10,000개의 테스트 이미지들의 정보를 가지고 있는 배열
        '''

    y_pred = []
    for sample in X_test:  # 테스트 세트의 각 이미지들에 대해서 반복
        # 이미지를 신경망에 전파(통과)시켜서 어떤 숫자가 될 지 확률을 계산.
        sample_hat = forward(network, sample)
        # 가장 큰 확률의 인덱스(-> 예측값)를 찾음.
        sample_pred = np.argmax(sample_hat)
        y_pred.append(sample_pred)  # 예측값을 결과 리스트에 추가

    return np.array(y_pred)


def accuracy(y_test, y_pred):
    ''' 테스트 데이터 레이블(y_true)과
        테스트 데이터 예측값(y_pred)을 파라미터로 전달받아서,
        정확도(accuracy) = (정답 개수)/(테스트 데이터 개수) 를 리턴.
        '''

    result = (y_test == y_pred)  # 정답과 예측값의 비교(bool) 결과를 저장한 배열
    print(result[:10])  # [True, True, ..., False, ... ]

    return np.mean(result)  # True=1, False=0으로 대체된 후 평균 계산됨.
    # (1 + 1 + ... + 0 + ...) / 전체 개수
    # return np.mean(y_test == y_pred)


if __name__ == '__main__':
    network = init_network()
    (X_train, y_train), (X_test, y_test) = load_mnist(normalize = True,
                                                      flatten = True,
                                                      one_hot_label = False)

    print(X_train[0])
    print(y_train[0])

    # 신경망 가중치(와 편향, bias) 행렬들 생성
    network = init_network()
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    print(f'W1: {W1.shape}, W2: {W2.shape}, W3: {W3.shape}')
    print(f'b1: {b1.shape}, b2: {b2.shape}, b3: {b3.shape}')

    start_time = time.time()

    # 테스트 이미지들의 예측값
    y_pred = predict(network, X_test)
    print('예측값 :', y_pred.shape)
    print(y_test[:10])
    print(y_pred[:10])

    # acc = accuracy(y_test, y_pred)

    acc = accuracy(y_test, y_pred)
    print('Accuracy :', acc)

    # 예측이 틀린 첫번째 이미지: X_test[8]
    # normalize(0 ~ 1) 되어 있고, 1차원 배열로 flatten 된 데이터
    # img = X_test[8] * 255  # 0 ~ 1 -> 0 ~ 255
    # img = img.reshape(28, 28)
    # img = Image.fromarray(img)
    # img.show()




    print('time elapsed: {:.2f}s'.format(time.time() - start_time))