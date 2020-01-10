import numpy as np

from ch03.e01_Perceptron import sigmoid


def init_network():
    ''' 신경망(neural network) 에서 사용되는 가중치 행렬과 bias 행렬을 생성
        교재 p. 88 그림 3-20
        입력층 : 입력값 (x1, x2) -> 1 x 2 행렬
        은닉층 :
            - 1st 은닉층 : 뉴런 3개 (x @ W1 + b1)
            - 2nd 은닉층 : 뉴런 2개
        출력층 : 출력 값 (y1, y2) -> 1 x 2 행렬
        W1, W2, W3, b1, b2, b3를 난수로 생성

        '''

    np.random.seed(1224)
    network = dict()  # 가중치/bias 행렬을 저장하기 위한 딕셔너리 -> 리턴 값

    # x @ W1 + b1: 1 x 3 행렬
    # (1 x 2) @ (2 x 3) + (1 x 3)

    network['W1'] = np.random.random(size = (2, 3)).round(2)
    network['b1'] = np.random.random(3).round(2)

    # z1 @ W2 + b2: 1 x 2
    # (1 x 3) @ (3 x 2) + (1 x 2)
    network['W2'] = np.random.random((3, 2)).round(2)
    network['b2'] = np.random.random(2).round(2)

    # z2 @ W3 + b3: 1 x 2
    # (1 x 2) @ (2 x 2) + (1 x 2)
    network['W3'] = np.random.random((2, 2)).round(2)
    network['b3'] = np.random.random(2).round(2)

    return network


def forward(network, x):
    '''
    순방향 전파(forward propagation). 입력 -> 은닉층 -> 출력.

    :param network: 신경망에서 사용되는 가중치/bias 행렬들을 저장한 dict
    :param x: 입력 값을 가지고 있는 (1차원) 리스트. [x1, x2]
    :return: 2개의 은닉층과 출력층을 거친 후 계산된 출력 값. [y1, y2]
    '''

    # 가중치 행렬:
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    # bias 행렬:
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    # 은닉층에서 활성화 함수: sigmoid 함수
    a1 = x.dot(W1) + b1
    z1 = sigmoid(a1)  # 첫번째 은닉층 전파
    z2 = sigmoid(z1.dot(W2) + b2)  # 두번째 은닉층 전파
    # 출력층 : z2 2 W3 + b3 값을 그대로 출력
    y = z2.dot(W3) + b3
    # return identity_function(y)
    return softmax(y)  # 출력층의 활성화 함수로 softmax 함수를 적용


# 출력층의 활성화 함수 1 - 항등 함수: 회귀(regression) 문제
def identity_function(x):
    return x


# 출력축의 활성화 함수 2 - softmax: 분류(classification) 문제
def softmax(X):
    '''
    1) X 가 1차원 : [x_1, x_2, ... , x_n]
    2) X 가 2차원 : [[x_11, x_12, ... , x_1n],
                    [x_21, x_22, ... , x_2n],
                     ... ]

    '''

    dimension = X.ndim
    if dimension == 1:
        m = np.max(X)  # 1차원 배열의 최댓값을 찾음
        X = X - m  # 0 이하의 숫자로 변환 <- exp 함수의 overflow 를 방지하기 위해
        y = np.exp(X) / np.sum(np.exp(X))

    elif dimension == 2:
        m = np.max(X, axis = 1).reshape(len(X), 1)
        X = X - m
        y = np.exp(X) / np.sum(np.exp(X), axis = 1).reshape(len(X), 1)

    return y

    # 0~1의 실수 값을 출력하는 소프트맥스는 그 성질 때문에 분류하는 결과에 대한 확률값으로 해석되기도 하여 분류 문제를 확률적으로 풀어볼 수 있도록 한다.
    # 소프트맥스 함수 구현시 주의할 점은 소프트맥스는 지수함수를 사용하기 때문에 값이 급격하게 증가하여 오버플로(overflow) 문제가 발생한다.
    # 컴퓨터는 계산할 수 있는 또는 표현할 수 있는 값 이상의 값은 계산/표현이 안되기 때문에 문제가 생긴다.


if __name__ == '__main__':
    # init_network() 함수 테스트
    network = init_network()
    print('W1 =', network['W1'], sep = '\n')
    print('b1 =', network['b1'])
    print(network.keys())

    # forward() 함수 테스트
    x = np.array([1, 2])
    y = forward(network, x)
    print('y =', y)

    # softmax() 함수 테스트
    print('x =', x)
    print('softmax =', softmax(x))

    x = [1, 2, 3]
    print('softmax =', softmax(x))

    x = [-10, 1e1, 1e2, 1e3]  # [1, 10, 100, 1000]
    print('x =', x)
    print('softmax =', softmax(x))

