'''
CNN 사용하는 파라미터(filter W, bias b)의 초기값과 학습 끝난 후의 값 비교

'''
import numpy as np
import matplotlib.pyplot as plt

from ch07_CNN.simple_convnet import SimpleConvNet
from common.layers import Convolution


def show_filters(filters, num_filters, ncols = 8):
    '''
    CNN 필터를 그래프로 출력

    :param filters:
    :param num_filters:
    :param ncols:
    '''
    nrows = np.ceil(num_filters / ncols)  # ceil(천장) 무조건 위로 반올림 3.12 -> 4
    for i in range(num_filters):  # 필터 갯수만큼 반복
        plt.subplot(nrows, ncols, i + 1, xticks = [], yticks = [])
        plt.imshow(filters[i, 0], cmap = 'gray')

    plt.show()


if __name__ == '__main__':
    # Simple CNN 생성
    cnn = SimpleConvNet()

    # 학습 시키기 전 파라미터 - 임의의 값들로 초기화 된 필터
    before_filters = cnn.params['W1']
    print('before_filters shape =', before_filters.shape)  # (30, 1, 5, 5)

    show_filters(before_filters, num_filters = 16, ncols = 4)

    # 학습이 끝난 후 파라미터
    cnn.load_params('cnn_params.pkl')
    after_filters = cnn.params['W1']

    # 학습 끝난 후 갱신된 파라미터 그래프로 출력
    show_filters(after_filters, 16, 4)

    # 학습 끝난 후 갱신 된 파라미터를 실제 이미지 파일에 적용
    lena = plt.imread('lena_gray.png')  # numpy array 로 이미지 열어진다. 단, png 파일만. jpeg 의 경우 외부 패키지 필요
    print('lena shape =', lena.shape)  # (256, 256) ndarray

    # 이미지 데이터를 Convolution 레이어의 forward() 메소드에 전달하기 위해서 2차원 배열을 4차원 배열로 변환
    lena = lena.reshape(1, 1, *lena.shape)  # * : 튜플 내용을 하나씩 꺼내준다. (256, 256)
    print('lena shape =', lena.shape)
    for i in range(16):  # 필터 16개에 대해 반복

        w = cnn.params['W1'][i]  # 갱신된 필터
        b = 0  # 바이어스 사용 안함
        w = w.reshape(1, *w.shape)  # 3차원 -> 4차원
        conv = Convolution(w, b)  # Convolution 레이어 생성
        out = conv.forward(lena)  # 이미지에 필터 적용
        # pyplot 을 사용하기 위해거 4차원을 2차원으로 변환
        out = out.reshape(out.shape[2], out.shape[3])
        plt.subplot(4, 4, i + 1, xticks = [], yticks = [])
        plt.imshow(out, cmap = 'gray')
    plt.show()
