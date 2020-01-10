import numpy as np


class Relu:
    '''
    ReLU(Rectified Linear Unit)
    relu(x) = x (if x > 0), 0 (otherwise) = max(0, x) : forward
    relu_prime(x) = 1 (if x > 0), 0 (otherwise) : backward
    '''

    def __init__(self):
        # relu 함수의 input 값(x)가 0보다 큰 지 작은 지를 저장할 field
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        return np.maximum(0, x)

    def backward(self, dout):
        # print('masking 전:', dout)
        dout[self.mask] = 0
        # print('masking 후:', dout)
        dx = dout
        return dx


if __name__ == '__main__':
    # ReLU 객체를 생성
    relu_gate = Relu()

    # x = 1일 때 relu 함수의 리턴값 y
    y = relu_gate.forward(1)
    print('y =', y)

    np.random.seed(103)
    x = np.random.randn(5)
    print('x =', x)
    y = relu_gate.forward(x)
    print('y =', y)  # relu 함수의 리턴 값
    print('mask =', relu_gate.mask)  # relu_gate의 필드 mask

    # back propagation(역전파)
    delta = np.random.randn(5)
    dx = relu_gate.backward(delta)
