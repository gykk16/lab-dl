import numpy as np
from scipy.signal import convolve, correlate, convolve2d, correlate2d

from ch07_CNN.ex01_convolution1d import convolution_1d

if __name__ == '__main__':
    x = np.arange(1, 6)
    w = np.array([2, 0])
    print(convolution_1d(x, w))

    # 일반적인 convolution(x, w) 결과의 shape는 (4,)
    # convolution 연산에서 x 원소 중 1과 5는 연산에 1번만 기여.
    # 다른 원소들은 2번씩 기여.
    # x의 모든 원소가 convolution 연산에서 동일한 기여를 할 수 있도록 padding.
    x_pad = np.pad(x, pad_width = 1, mode = 'constant', constant_values = 0)
    print(convolution_1d(x_pad, w))  # [ 2  4  6  8 10  0]

    # convolution 결과의 크기가 입력 데이터 x와 동일한 크기가 되도록 padding
    x_pad = np.pad(x, pad_width = (1, 0), mode = 'constant', constant_values = 0)
    print(convolution_1d(x_pad, w))  # [ 2  4  6  8 10]

    x_pad = np.pad(x, pad_width = (0, 1), mode = 'constant', constant_values = 0)
    print(convolution_1d(x_pad, w))  # [ 4  6  8 10  0]

    # scipy.signal.convolve() 함수    , w 반전 시킴
    # 차원 관계 없이 함수 사용 가능
    conv = convolve(x, w, mode = 'valid')
    print('colvolve, mode = valid :', conv)
    conv_full = convolve(x, w, mode = 'full')
    print('colvolve, mode = full:', conv_full)  # x의 모든 원소가 동일하게 연산에 기여
    conv_same = convolve(x, w, mode = 'same')
    print('colvolve, mode = same:', conv_same)  # x의 크기와 동일한 리턴

    # scipy.signal.correlate() 함수   , w 반전 안함
    # 차원 관계 없이 함수 사용 가능
    cross_corr = correlate(x, w, mode = 'valid')
    print('correlate, mode = valid', cross_corr)
    cross_corr_full = correlate(x, w, mode = 'full')
    print('correlate, mode = full', cross_corr_full)
    cross_corr_same = correlate(x, w, mode = 'same')
    print('correlate, mode = same', cross_corr_same)

    # scipy.signal.convolve2d(), scipy.signal.correlate2d 함수
    # 2차원에 한해서만 사용 가능
    # (4, 4) 2d ndarray
    # p.232
    x = np.array([[1, 2, 3, 0],
                  [0, 1, 2, 3],
                  [3, 0, 1, 2],
                  [2, 3, 0, 1]])
    w = np.array([[2, 0, 1],
                  [0, 1, 2],
                  [1, 0, 2]])
    cross_corr = correlate2d(x, w, mode = 'valid')
    print('correlate2d , mode = valid \n', cross_corr)
    cross_corr_full = correlate2d(x, w, mode = 'full')
    print('correlate2d , mode = full \n', cross_corr_full)
    cross_corr_same = correlate2d(x, w, mode = 'same')
    print('correlate2d , mode = same \n', cross_corr_same)

