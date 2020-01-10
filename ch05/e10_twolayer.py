'''
오차 역전파를 사용한 2층 신경망 (은닉 1 + 출력 1)

'''
from collections import OrderedDict

import numpy as np

from ch05.e05_relu import Relu
from ch05.e07_affine import Affine
from ch05.e08_softmaxwithloss import SoftmaxWithLoss
from dataset.mnist import load_mnist


class TwoLayerNetwork:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = .01):
        ''' 신경망의 구조(모양) 결정 '''
        np.random.seed(106)

        # 가중치/편향 행렬들을 초기화
        self.params = dict()
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        # layer 생성/초기화
        self.layers = OrderedDict()
        # dict 에 데이터가 추가된 순서가 유지되는 dict
        self.layers['affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['relu'] = Relu()
        self.layers['affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.last_layer = SoftmaxWithLoss()

    def predict(self, X):
        ''' 입력 데이터 X의 예측값을 리턴. '''
        # Y = self.layers['affine1'].forward(X)
        # Y = self.layers['relu'].forward(Y)
        # Y = self.layers['affine2'].forward(Y)

        for layer in self.layers.values():
            X = layer.forward(X)
        return X

    def loss(self, X, Y_true):
        ''' 입력 데이터 X와 실제 값(레이블) Y_true가 주어졌을 때,
            손실(cross-entropy)를 계산해서 리턴.
            '''
        Y_pred = self.predict(X)
        loss = self.last_layer.forward(Y_pred, Y_true)
        return loss

    def accuracy(self, X, Y_true):
        ''' 입력 데이터 X와 실제 값(레이블) Y_true 가 주어졌을때,
            예측 값들의 정확도를 계삲서 리턴
            accuracy = 에측이 실제값과 일치하는 개수 / 전체 입력 데이터 개수
            X, Y_true 는 모두 2차원 배열(행렬)이라 가정
            '''
        Y_pred = self.predict(X)
        predictions = np.argmax(Y_pred, axis = 1)
        trues = np.argmax(Y_true, axis = 1)
        acc = np.mean(predictions == trues)
        return acc

    def gradient(self, X, Y_true):
        ''' 입력 데이터 X와 실게 값(레이블) Y_true 가 주어졌을 때,
            모든 레이어에 대해서 forward propagation 을 수행한 후,
            오차 역전파 방법을 이용해서 dW1, db1, dW2, db2 를 계산 후 리턴
            '''
        gradients = dict()
        # 가중치/편향 행렬에 대한 gradient 들을 저장할 dict

        # forward propagation
        self.loss(X, Y_true)

        # back propagation
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())  # [affine1, ReLU, affine2]
        layers.reverse()  # 리스트를 역순으로 바꿈  [affine2, ReLU, affine1]
        for layer in layers:
            dout = layer.backward(dout)

        # 모든 레이어에 대해 역전파가 끝나며,
        gradients['W1'] = self.layers['affine1'].dW
        gradients['b1'] = self.layers['affine1'].db
        gradients['W2'] = self.layers['affine2'].dW
        gradients['b2'] = self.layers['affine2'].db

        return gradients


if __name__ == '__main__':
    # MNIST 데이터 로드
    (X_train, Y_train), (X_test, Y_test) = load_mnist(one_hot_label = True)
    # 데이터 확인
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

    # 신경망 객체 생성
    neural_net = TwoLayerNetwork(input_size = 784, hidden_size = 32, output_size = 10)

    for key in neural_net.params:
        print(key, ':', neural_net.params[key].shape)

    for key in neural_net.layers:
        print(key, ':', neural_net.layers[key])

    print('last layer :', neural_net.last_layer)

    # predict 메소드 테스트
    Y_pred = neural_net.predict(X_train[0])
    print(Y_pred)
    print(np.argmax(Y_pred))

    loss1 = neural_net.loss(X_train[0], Y_train[0])
    print('loss 1 =', loss1)

    Y_pred = neural_net.predict(X_train[:3])
    print(Y_pred)
    print(np.argmax(Y_pred, axis = 1))

    loss2 = neural_net.loss(X_train[:3], Y_train[:3])
    print('loss 2 =', loss2)

    # accuracy 메소드 테스트
    print(Y_train[:10])
    print('accuracy =', neural_net.accuracy(X_train[:10], Y_train[:10]))

    # gradients 메소드 테스트
    gradiants = neural_net.gradient(X_train[:100], Y_train[:100])
    for key in gradiants:
        print(key, ':', gradiants[key].shape)


