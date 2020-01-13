'''
2차원 Convolution(합성곱) 연산

'''
import numpy as np


def _convolution_2d(x, w):
    '''
    x, w : 2d ndarray
    x.shape >= w.shape
    x 와 w 의 교차 상관(cross-correlation) 연산 결과를 리턴
    '''

    xh, xw = x.shape[0], x.shape[1]
    wh, ww = w.shape[0], w.shape[1]

    rows = xh - wh + 1
    cols = xw - ww + 1

    cc = np.zeros(shape = (rows, cols))
    for i in range(rows):
        for j in range(cols):
            x_sub = x[i:wh + i, j:ww + j]
            cc[i, j] = np.sum(x_sub * w)
    return cc


def convolution_2d(x, w):  # 선생님 답안
    # convolution 결과 행렬(2d ndarray)의 shape
    rows = x.shape[0] - w.shape[0] + 1
    cols = x.shape[1] - w.shape[1] + 1
    conv = []  # 결과를 저장할 list
    for i in range(rows):
        for j in range(cols):
            x_sub = x[i:i + w.shape[0], j:j + w.shape[1]]
            fma = np.sum(x_sub * w)
            conv.append(fma)
    conv = np.array(conv).reshape(rows, cols)
    return conv


if __name__ == '__main__':
    np.random.seed(113)

    x = np.arange(1, 10).reshape(3, 3)
    print('x =', x)

    w = np.array([[2, 0],
                  [0, 0]])
    print('w =', w)

    # 2d 배열 x의 가로 (width), 세로 (height)
    xh, xw = x.shape[0], x.shape[1]
    # 2d 배열 w의 가로 (width), 세로 (height)
    wh, ww = w.shape[0], w.shape[1]

    x_sub1 = x[0:wh, 0:ww]  # w 의 크기와 맞춰야 한다   : x[0:2, 0:2]
    x_sub2 = x[0:wh, 1:1 + ww]  # x[0:2, 1:3]
    x_sub3 = x[1:wh + 1, 0:ww]  # x[1:3, 0:2]
    x_sub4 = x[1:wh + 1, 1:1 + ww]  # x[1:3, 1:3]
    print(x_sub1)
    print(x_sub2)
    print(x_sub3)
    print(x_sub4)
    fma1 = np.sum(x_sub1 * w)
    fma2 = np.sum(x_sub2 * w)
    fma3 = np.sum(x_sub3 * w)
    fma4 = np.sum(x_sub4 * w)
    print('fma1 =', fma1)
    print('fma2 =', fma2)
    print('fma3 =', fma3)
    print('fma4 =', fma4)
    conv = np.array([fma1, fma2, fma3, fma4]).reshape(2, 2)
    print('conv =', conv)

    conv = convolution_2d(x, w)
    print(conv)

    x = np.random.randint(10, size = (5, 5))  # 0 ~ 9 사이의 정수
    w = np.random.randint(5, size = (3, 3))  # 0 ~ 4 사이의 정수
    print(x)
    print(w)
    conv = convolution_2d(x, w)
    print(conv)
