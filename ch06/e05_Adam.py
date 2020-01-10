'''
https://github.com/WegraLee/deep-learning-from-scratch/blob/master/common/optimizer.py

파라미터 최적화 알고리즘 4) Adam(Adaptive Moment estimate)
    AdaGrad + Momentum 알고리즘
    학습률 변화 + 속도(모멘텀) 개념 도입
    W: 파라미터
    lr: 학습률(learning rate)
    t: timestamp. 반복할 때마다 증가하는 숫자. update 메소드가 호출될 때마다 +1
    beta1, beta2: 모멘텀을 변화시킬 때 사용하는 상수들. 0<= beta1,2 <1
    m: 1st momentum ~ gradient(dL/dW) -> SGD의 gradient를 수정함.
    v: 2nd momentum ~ gradient**2 ((dL/dW)**2) -> SGD의 학습률을 수정함.
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad * grad
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    W = W - lr * m_hat / sqrt(v_hat)


'''
import numpy as np
import matplotlib.pyplot as plt

from ch06.e01_matplot3d import fn_derivative, fn


class Adam:
    def __init__(self, lr = 0.01, beta1 = 0.9, beta2 = 0.99):
        self.lr = lr  # learning rate(학습률)
        self.beta1 = beta1
        self.beta2 = beta2
        self.t = 0  # timestamp
        self.m = dict()  # 1st momentum
        self.v = dict()  # 2nd momentum

    def update(self, params, gradients):
        self.t += 1  # update 가 호출될 때마다 timestamp 를 1씩 증가
        if not self.m:  # m 이 비어 있는 dict() 때
            for key in params:
                self.m[key] = np.zeros_like(params)
                self.v[key] = np.zeros_like(params)

        epsilon = 1e-8  # 0으로 나누는 경우를 방지하기 위해서 사용할 상수
        for key in params:
            # m = beta1 * m + (1 - beta1) * dL/dW
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * gradients[key]
            # v = beta2 * v + (1 - beta2) * (dL/dW)**2
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * gradients[key] ** 2
            # m_hat = m / (1 - beta1 ** t)
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            # v_hat = v / (1 - beta2 ** t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            # W = W - [lr / (sqrt(v_hat))] * m_hat
            params[key] -= (self.lr / (np.sqrt(v_hat) + epsilon)) * m_hat


if __name__ == '__main__':
    params = {'x': -7., 'y': 2.}  # 파라미터 초기값
    gradients = {'x': 0., 'y': 0.}  # gradient 초기값

    # Adam 클래스의 인스턴스 생성
    adam = Adam(lr = .3)  # lr = .01

    # 학습 하면서 파라미터(x, y)들이 업데이트 되는 내용을 저장하기 위한 리스트
    x_history = []
    y_history = []
    for i in range(30):
        x_history.append(params['x'])
        y_history.append(params['y'])
        gradients['x'], gradients['y'] = fn_derivative(params['x'], params['y'])
        adam.update(params, gradients)
        # 파라미터 값 출력
        print(f'({params["x"]}, {params["y"]})')

    # 등고선(contour) 그래프
    x = np.linspace(-10, 10, 2000)
    y = np.linspace(-5, 5, 1000)
    X, Y = np.meshgrid(x, y)
    Z = fn(X, Y)

    mask = Z > 8
    Z[mask] = 0

    plt.contour(X, Y, Z, 10)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Adam')
    plt.axis('equal')
    # x_history, y_history 를 plot
    plt.plot(x_history, y_history, 'o-', c = 'r')
    plt.show()
