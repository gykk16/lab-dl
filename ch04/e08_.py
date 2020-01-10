'''
weight 행렬에 경사 하강법(gradient descent) 적용

'''

import numpy as np

from ch03.e11_teacher import softmax
from ch04.e03_cross_entropy import cross_entropy
from ch04.e05_미분 import numerical_gradient


class SimpleNetwork:
    def __init__(self):
        np.random.seed(1230)
        self.W = np.random.randn(2, 3)
        # 가중치 행렬의 초기 값들을 임의로 설정

    def predict(self, x):
        z = x.dot(self.W)
        y = softmax(z)
        return y

    def loss(self, x, y_true):
        ''' 손실 함수 (ㅣoss function) - cross entropy '''
        y_pred = self.predict(x)  # 입력이 x 일때 출력 y의 예측값 계산
        ce = cross_entropy(y_pred, y_true)  # cross entropy 계산
        return ce

    def gradient(self, x, t):
        ''' x: 입력, t: 출력 실제 값 (정답 레이블) '''
        fn = lambda W: self.loss(x, t)  # 오차 최소값을 찾기 위해 손실함수 사용
        return numerical_gradient(fn, self.W)


if __name__ == '__main__':
    # SimpleNetwork 클래스 객체를 생성
    network = SimpleNetwork()  # 생성자 호출 -> __init__() 메소드 호출
    print('w =', network.W)

    # x = [.6, .9] 일때 y_true = [0, 0, 1] 라고 가정
    x = np.array([.6, .9])
    y_true = np.array([.0, .0, 1.])
    print('x =', x)
    print('y_true =', y_true)

    y_pred = network.predict(x)
    print('y_pred =', y_pred)

    ce = network.loss(x, y_true)
    print('cross entropy =', ce)

    g1 = network.gradient(x, y_true)
    print('g1 =', g1)

    lr = .1  # learning rate
    network.W -= lr * g1
    print('W =', network.W)
    print('y_pred =', y_pred)
    print('ce =', network.loss(x, y_true))

    lr = 0.1
    for i in range(100):
        grad = network.gradient(x, y_true)
        network.W -= lr * grad
        print(f'y_pred = {network.predict(x)}, '
              f'cross entropy = {network.loss(x, y_true)}')
