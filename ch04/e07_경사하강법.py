'''
경사 하강법 (gradient descent)

x_new = x - lr * df/dx
위 과정을 반복 -> f(x)의 최소값을 찾음

'''
import numpy as np
import matplotlib.pyplot as plt

from ch04.e05_미분 import numerical_gradient


def gradient_method(fn, x_init, lr = 0.01, step = 100):
    '''

    :param fn:
    :param x_init:
    :param lr: learning rate
    :param step: 반복 횟수
    :return:
    '''

    x = x_init  # 점진적으로 변화시킬 변수
    x_history = []  # x가 변화되는 과정을 저장할 배열, 로그를 확인하기 위해

    for i in range(step):  # step 횟수만큼 반복하면서
        x_history.append(x.copy())  # x 의 복사본을 x 변화 과정에 기록, .copy() 사용하는 이유: x 가 벼열이기 때문에 값이 아닌 주소를 저장한다
        grad = numerical_gradient(fn, x)  # x 에서의 gradient 를 계산
        x -= lr * grad  # x_new = x_init - lr * grad: x 를 변경

    return x, np.array(x_history)


def fn(x):
    if x.ndim == 1:
        return np.sum(x ** 2)
    else:
        return np.sum(x ** 2, axis = 1)


if __name__ == '__main__':
    init_x = np.array([4.])
    x, x_hist = gradient_method(fn, init_x, lr = .1)
    print('x =', x)
    print('x_hist =', x_hist)

    # 학습률(learning rate: lr) 이 너무 작으면,
    # 최소값을 찾아가는 시간이 오래 걸리고,
    # 학습률이 너무 크면, 최소값을 찾지 못하고 발산하는 경우가 생길 수 있다.

    init_x = np.array([4., -3.])
    x, x_hist = gradient_method(fn, init_x, lr = .1, step = 100)
    print('x =', x)
    print('x_hist =', x_hist)

    # x_hist(최소값을 찾아가는 과정)을 산점도 그래프
    plt.axvline(color = '.8')
    plt.axhline(color = '.8')
    plt.scatter(x_hist[:, 0], x_hist[:, 1])
    plt.xlim([-5, 5])
    plt.ylim([-5, 5])
    plt.show()

    #
    # 동심원: x**2 + y**2 = r**2 -> y**2 = r**2 - x**2
    for r in range(1, 5):
        r = float(r)  # 정수 -> 실수 변환
        x_pts = np.linspace(-r, r, 100)
        y_pts1 = np.sqrt(r ** 2 - x_pts ** 2)
        y_pts2 = -np.sqrt(r ** 2 - x_pts ** 2)
        plt.plot(x_pts, y_pts1, ':', color = 'gray')
        plt.plot(x_pts, y_pts2, ':', color = 'gray')

    plt.scatter(x_hist[:, 0], x_hist[:, 1])
    plt.xlim([-5, 5])
    plt.ylim([-5, 5])
    plt.axvline(color = '0.8')
    plt.axhline(color = '0.8')
    plt.show()
