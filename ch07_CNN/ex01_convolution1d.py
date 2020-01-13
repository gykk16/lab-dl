'''
1차원 Convolution, Cross-Correlation 연산

'''
import numpy as np


def __convolution_1d(x, w):
    '''
    x, w: 1d ndarray
    len(x) >= len(w)

    '''
    w_r = np.flip(w)
    move = len(x) - len(w) + 1
    conv = []
    for i in range(move):
        x_sub = x[i:i + len(w)]
        fma = x_sub.dot(w_r)
        conv.append(fma)
    conv = np.array(conv)

    return conv


def _convolution_1d(x, w):  # 선생님 ver
    w_r = np.flip(w)
    nx = len(x)  # x 원소의 개수
    nw = len(w)  # w 원소의 개수
    n = nx - nw + 1  # convolution 연산 결과의 원소 개수
    conv = []
    for i in range(n):
        x_sub = x[i:i + nw]
        fma = np.sum(x_sub * w_r)  # fused multiply-add
        conv.append(fma)
    conv = np.array(conv)
    return conv


def convolution_1d(x, w):
    w_r = np.flip(w)
    conv = cross_correlation_1d(x, w_r)
    return conv


def cross_correlation_1d(x, w):
    '''
    x, w: 1d ndarray, len(x) >= len(w)
    x와 w의 교차 상관(cross-correlation) 연산 결과를 리턴
    '''
    nx = len(x)
    nw = len(w)
    n = nx - nw + 1
    cc = []
    for i in range(n):
        x_sub = x[i:i + nw]
        fma = x_sub.dot(w)
        cc.append(fma)
    return np.array(cc)


if __name__ == '__main__':
    np.random.seed(113)
    x = np.arange(1, 6)
    print('x =', x)
    w = np.array([2, 1])
    print('w =', w)

    # Convolution(합성곱) 연산
    # 1) w 를 반전
    # w_r = np.array([1, 2])
    w_r = np.flip(w)
    print('w_r =', w_r)

    # 2) FMA(Fused Multiply-Add)
    conv = []
    for i in range(4):
        x_sub = x[i:i + 2]  # (0, 1), (1, 2), (2, 3), (3, 4)
        # fma = np.sum(x_sub * w_r)
        fma = np.dot(x_sub, w_r)
        conv.append(fma)
    conv = np.array(conv)
    print(conv)

    # 1 차원 convolution 연산 결과의 크기(원소의 개수)
    # = len(x)- len(w) + 1
    #   ex) x = [1 2 3 4 5], w = [2 1]
    #       convolution 연산 결과 = [ 5  8 11 14] : 5 - 2 + 1 = 4

    # convolution_1d 함수 테스트
    conv = convolution_1d(x, w)
    print(conv)

    x = np.arange(1, 6)
    w = np.array([2, 0, 1])
    conv = convolution_1d(x, w)
    print(conv)

    # 교차 상관(Cross-Correlation)연산
    # 합성곱 연산과 다른 점은 w 를 반전시키지 않는다는 것
    # CNN(Convolutional Neural Network, 합성곱 신경망) 에서는
    # 대부분의 경우 교차 상관을 사용함 -> 가중치 행렬을 난수로 생성하기 때문 (w 를 반전해도 난수임)
    # 가중치 행렬을 난수로 생성한 후 Gradient Descent 등을 이용해서 갱신하기 때문에,
    # 대부분의 경우 합성곱 연산 대신 교차 상관 연산을 사용함

    cross_corr = cross_correlation_1d(x, w)
    print(cross_corr)