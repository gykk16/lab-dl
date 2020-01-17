'''
Simple Convolutional Neural Network
p.228 그림7-2

'''
from collections import OrderedDict

import numpy as np

from common.layers import Convolution


class SimpleConvNet:
    '''
    1st hidden layer: Convolution -> ReLU -> Pooling
    2nd hidden layer: Affine -> ReLU (fully-connected network, 완전연결층)
    출력층: Affine -> SoftmaxWithLoss

    '''

    def __init__(self,
                 input_dim = (1, 28, 28),
                 conv_param = {'filter_num:': 30,
                               'filter_size': 5,
                               'pad': 0,
                               'stride': 1},
                 hidden_size = 100,
                 output_size = 10,
                 weight_init_std = 0.01):
        ''' 
        인스턴스 초기화 - CNN 구성, 변수들 초기화
        input_dim: 입력 데이터 차원. MNIST인 경우 (1, 28, 28)
        conv_param: Convolution 레이어의 파라미터(filter, bias)를 생성하기 위해
        필요한 값들
            필터 개수(filter_num),
            필터 크기(filter_size = filter_height = filter_width),
            패딩 개수(pad),
            보폭(stride)
        hidden_size: Affine 계층에서 사용할 뉴런의 개수
        output_size: 출력값의 원소의 개수. MNIST인 경우 10
        weight_init_std: 가중치(weight) 행렬을 난수로 초기화할 때 사용할 표준편차
        '''

        c, h, w = input_dim
        fn, fh, fw = conv_param['filter_num'], conv_param['filter_size'], conv_param['filter_size']
        s, p = conv_param['stride'], conv_param['pad']
        oh = (h - fh + 2 * p) // s + 1
        ow = (w - fw + 2 * p) // s + 1

        W1 = np.random.randn(fn, c, fh, fw)
        b1 = np.zeros(shape = (fn,))

        # CNN layer(계층) 생성, 연결
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(W1, b1)

        # CNN layer 에서 필요한 파라미터들
        self.params = dict()
        self.params['W1'] = W1
        self.params['b1'] = b1

        def predict(self):
            pass

        def loss(self):
            pass

        def accuracy(self):
            pass

        def gradient(self):
            pass

    if __name__ == '__main__':
        pass
        # MNIST 데이터 로드
        # SimpleConvNet 생성
        # 학습 -> 테스트
