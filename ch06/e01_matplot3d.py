'''

'''

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d  # 3D 그래프를 그리기 위해서 반드시 import
import numpy as np


def fn(x, y):
    '''
    f(x, y) = (1/20) * x ** 2 + y ** 2
    '''

    return x ** 2 / 20 + y ** 2


def fn_derivative(x, y):
    # derivative : 도함수(미분)
    ''' 편미분 df/dx, df/dy 를 리턴하는 함수 '''
    return x / 10, 2 * y


if __name__ == '__main__':
    # x 좌표들
    x = np.linspace(-10, 10, 1000)  # x 좌표들
    y = np.linspace(-10, 10, 1000)  # y 좌표들
    # 3 차원 드래프를 그리기 위해서
    X, Y = np.meshgrid(x, y)
    Z = fn(X, Y)

    fig = plt.figure()
    ax = plt.axes(projection = '3d')
    # projection 파라미터를 사용하려면 mpl_toolkits.mplt3d 패키지가 필요
    ax.contour3D(X, Y, Z, 100,  # 등고선 갯수
                 cmap = 'binary')  # 등고선 색상 맵(color map)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    # 등고선(contour) 그래프
    plt.contour(X, Y, Z, 50)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.show()
