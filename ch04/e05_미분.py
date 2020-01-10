import numpy as np


def numerical_diff(fn, x):
    '''
    Numerical Differential
    함수 fn와 점 x가 주어졌을 때, x 에서의 함수 fn의 미분(도함수) 값

    '''

    h = 1e-4  # 0.0001
    return (fn(x + h) - fn(x - h)) / (2 * h)


def f1(x):
    return 0.001 * x ** 2 + 0.01 * x


def f1_prime(x):
    ''' 근사값을 사용하지 않은 함수 f1의 도함수 '''

    return 0.002 * x + 0.01


def f2(x):
    ''' x = [x0, x1, ... ] '''

    return np.sum(x ** 2)  # x0 ** 2 + x1 ** 2 + ...


def f3(x):
    ''' f3 = x0 + x1**2 + x2**3 '''

    return x[0] + x[1] ** 2 + x[2] ** 3


def f4(x):
    ''' f4 = x0**2 + x0 * x1 + x1**2 '''

    return x[0] ** 2 + x[0] * x[1] + x[1] ** 2


def _numerical_gradient(fn, x):
    ''' 점 x=[x0, x1, ..., xn] 에서의
        함수 fn = fn(x0, x1, ..., xn)의
        각 편미분(partial differential) 값들의 배열을 리턴.
    '''

    # x = x.astype(np.float)  # 실수 타입
    gradient = np.zeros_like(x)  # np.zeros(shape=x.shape)
    h = 1e-4  # 0.0001
    for i in range(x.size):
        ith_value = x[i]
        x[i] = ith_value + h
        fh1 = fn(x)
        x[i] = ith_value - h
        fh2 = fn(x)
        gradient[i] = (fh1 - fh2) / (2 * h)
        x[i] = ith_value
    return gradient


def numerical_gradient(fn, x):
    ''' x = [
        [x11 x12 x13, ...]
        [x21 x22 x23 ...]
        ...
        ] '''

    if x.ndim == 1:
        return _numerical_gradient(fn, x)
    else:
        grads = np.zeros_like(x)
        for i, x_i in enumerate(x):
            grads[i] = _numerical_gradient(fn, x_i)
        return grads


if __name__ == '__main__':
    estimate = numerical_diff(f1, 3)
    print('근사값 =', estimate)
    real = f1_prime(3)
    print('실제값 =', real)

    # f2 함수의 점 (3, 4)에서의 편미분 값
    estimate_1 = numerical_diff(lambda x: x ** 2 + 4 ** 2, 3)
    print(estimate_1)
    estimate_2 = numerical_diff(lambda x: 3 ** 2 + x ** 2, 4)
    print(estimate_2)

    gradient = numerical_gradient(f2, np.array([3, 4]))
    print(gradient)

    # f3 = x0 + x1**2 + x2**3
    # 점 (1, 1, 1) 에서의 각 편미분들의 값
    # df/dx0 = 1, df/dx1 = 2, df/dx = 3
    gradient = numerical_gradient(f3, np.array([1, 1, 1]))
    print(gradient)

    # f4 = x0**2 + x0 * x1 + x1**2
    # 점 (1, 2) 에서의 df/dx0 = 4, df/dx1 = 5
    gradient = numerical_gradient(f4, np.array([1, 2]))
    print(gradient)
