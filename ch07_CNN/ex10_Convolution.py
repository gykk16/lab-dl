'''
Convolution 클래스

'''
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from common.util import im2col
from dataset.mnist import load_mnist


class Convolution:
    def __init__(self, W, b, stride = 1, pad = 0):
        self.W = W  # weight -> filter 역할
        self.b = b  # bias
        self.stride = stride
        self.pad = pad

        # 중간 데이터: forward 에서 생성되는 데이터 -> backward 에서 사용되기 때문에 저장
        self.x = None
        self.x_col = None
        self.W_col = None

        # gradients
        self.dW = None
        self.db = None

    def _forward(self, x):
        '''

        :param x: 4차원 이미지 (mini-batch) 데이터
        '''

        self.x = x
        n, c, h, w = x.shape
        fn, c, fh, fw = self.W.shape
        oh = (h - fh + 2 * self.pad) // self.stride + 1
        ow = (w - fw + 2 * self.pad) // self.stride + 1

        # 이미지 데이터 x 를 im2col 함수를 사용해서 x_col 로 변환
        self.x_col = im2col(self.x, fh, fw, self.stride, self.pad)

        # 필터 w 를 x_col 과 dot 연산을 할 수 있도록 reshape & transpose
        self.W_col = self.W.reshape(fn, -1).T
        # W(fn, c, fh, fw) --> W_col(fn, c*fh*fw) --> W_col(c*fh*fw, fn)

        # x_col @ w_col + bias
        out = np.dot(self.x_col, self.W_col) + self.b  # self.x_col.dot(self.W_col)

        # @ 연산의 결과를 reshape & transpose
        out = out.reshape(n, oh, ow, fn)
        out = out.transpose(0, 3, 1, 2)

        return out

    def forward(self, x):  # 선생님 답안
        self.x = x
        n, c, h, w = x.shape
        fn, c, fh, fw = self.W.shape
        oh = (h - fh + 2 * self.pad) // self.stride + 1  # output height
        ow = (w - fw + 2 * self.pad) // self.stride + 1  # output width

        self.x_col = im2col(self.x, fh, fw, self.stride, self.pad)
        self.W_col = self.W.reshape(fn, -1).T
        # W(fn,c,fh,fw) --> W_col(fn, c*fh*fw) --> (c*fh*fw, fn)

        out = np.dot(self.x_col, self.W_col) + self.b
        # self.x_col.dot(self.W_col)

        out = out.reshape(n, oh, ow, -1).transpose(0, 3, 1, 2)

        return out

    def backward(self):
        pass


if __name__ == '__main__':
    np.random.seed(115)

    # 연습
    # Convolution 을 생성
    # filter: (fn, c, fh, fw)
    W = np.zeros(shape = (1, 1, 4, 4), dtype = np.uint8)  # dtype: 8bit 부호 없는 정수
    W[0, 0, 1, 1] = 1
    print('W =\n', W)

    b = np.zeros(1)

    print('W shape = ', W.shape)
    print('b shape = ', b.shape)

    conv = Convolution(W, b)  # Convolution 클래스의 생성자 호출

    # MNIST 데이터를 forward
    (x_train, y_train), (x_test, y_test) = load_mnist(normalize = False, flatten = False)
    # input = x_train[0]
    # print('input shape =', input.shape)
    # x = input.reshape(1, 1, 28, 28)
    # print('x shape =', x.shape)
    input = x_train[0:1]
    print('input shape =', input.shape)
    out = conv.forward(input)
    print('out shape =', out.shape)

    # 이미지 확인
    img = out.squeeze()  # 차원 축소
    print('img :', img.shape)
    plt.imshow(img, cmap = 'gray')
    plt.show()

    #####
    #####
    #####
    print('\n====================\n')
    #
    W = np.random.randint(10, size = (10, 3, 10, 10))
    b = np.zeros(shape = (1, 10))
    print('W shape =', W.shape)
    print('b shape =', b.shape)

    # Convolution 을 생성
    convolution = Convolution(W, b, stride = 5, pad = 0)

    # 다운로드 받은 이미지 파일을 ndarray 로 변환해서 forward
    img = Image.open('sample.jpg')
    img = np.array(img)
    # print(img.shape)
    x = img.reshape(1, 1075, 1920, 3)
    x = x.transpose(0, 3, 1, 2)
    print('x shape =', x.shape)
    conv = convolution.forward(x)
    print('conv shape =', conv.shape)

# W shape =  (10, 3, 10, 10)
# b shape =  (1, 10)
# x shape = (1, 3, 1075, 1920)
# conv shape = (1, 10, 214, 383)

# W shape =  (10, 3, 10, 10)
# b shape =  (1, 10)
# x shape = (1, 3, 1075, 1920)
# conv shape = (1, 10, 214, 383)
