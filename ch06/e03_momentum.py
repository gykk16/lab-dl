'''
파라미터 최적회 알고리즘 2) Momentum 알고리즘

v: 속도 (velocity)
m: 모멘텀 상수 (momentum constant)
lr: 학습률
W: 파라미터

v = m * v - lr * dL/dW
W = W + v = W + m * v - lr * dL/dW

'''
import numpy as np
import matplotlib.pyplot as plt

from ch06.e01_matplot3d import fn_derivative, fn


class Momentum:
    def __init__(self, lr = 0.01, m = 0.9):
        self.lr = lr  # 학습률
        self.m = m  # 모멘텀 상수(속도 v에 곱해줄 상수)
        self.v = dict()  # 속도

    def update(self, params, gradients):
        if not self.v:
            for key in params:
                # 파라미터(x, y 등)와 동일한 shape의 0 으로 패워진 배열 생성
                self.v[key] = np.zeros_like(params[key])

        for key in params:
            # v = m * v - lr * dL/dW
            # self.v[key] = self.m * self.v[key] - self.lr * gradients[key]
            self.v[key] *= self.m
            self.v[key] -= self.lr * gradients[key]
            # W = W + v
            params[key] += self.v[key]


if __name__ == '__main__':
    # Momentum 클래스의 인스턴스를 생성

    momentum = Momentum(lr = .1)

    # update 메소드 테스트
    params = {'x': -7., 'y': 2.}  # 파라미터 초기값
    gradients = {'x': 0., 'y': 0.}  # gradient 초기값
    x_history = []  # params['x'] 가 갱신되는 과정을 저장할 리스트
    y_history = []  # params['y'] 가 갱신되는 과정을 저장할 리스트

    for i in range(30):
        x_history.append(params['x'])
        y_history.append(params['y'])
        gradients['x'], gradients['y'] = fn_derivative(params['x'], params['y'])
        momentum.update(params, gradients)

    for x, y in zip(x_history, y_history):
        print(f'({x}, {y})')

    # contour 그래프에 파라미터의 갱신 값 그래프를 추가

    # f(x,y) 함수를 등고선으로 표현
    x = np.linspace(-10, 10, 2000)
    y = np.linspace(-5, 5, 1000)
    X, Y = np.meshgrid(x, y)
    Z = fn(X, Y)

    mask = Z > 7
    Z[mask] = 0

    plt.contour(X, Y, Z, 10)
    plt.title('Momentum')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')

    # 등고선 그래프에 파라미터(x, y)들이 갱신되는 과정을 추가
    plt.plot(x_history, y_history, 'o-', color = 'r')
    plt.show()
