'''
im2col 함수를 사용한 Convolution 구현

'''

import numpy as np

from common.util import im2col

if __name__ == '__main__':
    np.random.seed(115)

    # p.238 그림 7-11 참조

    # 가상의 이미지 1개 생성

    # !!중요!! 이미지, 필터의 n, h, w 구분!
    # (n, c, h, w) = (이미지 개수, color-depth, heigth, width)
    x = np.random.randint(10, size = (1, 3, 7, 7))  # rgb, height 7, width 7 인 이미지 1장
    print('x =\n', x)
    print('x shape =', x.shape)

    # (3, 5, 5) 크기의 필터 1개 생성
    # (fn, c, fh, fw) = (필터 개수, color-depth, 필터 height, 필터 width)
    w = np.random.randint(5, size = (1, 3, 5, 5))
    print('w =\n', w)
    print('w shape =', w.shape)

    #
    # 필터를 stride = 1, padding = 0 으로 해서 convolution 연산
    # output 예상해보기,
    #   oh = (h - fh + 2 * p) / s + 1 = (7 - 5 + 2 * 0) / 1 + 1 = 2 + 1 = 3
    #   ow = (h - fw + 2 * p) / s + 1 = 3

    #
    # 4차원 배열 필터 x를 2차원으로 변환
    # 이미지 데이터 x를 함수 im2col에 전달
    x_col = im2col(x, filter_h = 5, filter_w = 5, stride = 1, pad = 0)
    print('x_col =', x_col.shape)  # x_col = (9, 75)
    # x_col = (9, 75) = (oh * ow, c * fh * fw)
    # 필터를 1차원으로 펼침 -> c * fh * fw = 3 * 5 * 5 = 75

    #
    # 4차원 배열 필터 w를 2차원으로 변환
    w_col = w.reshape(1, -1)  # row 의 개수가 1, 모든 원소들은 col 으로 (1, 75)
    print('w_col shape =', w_col.shape)  # (1, 75)
    # x_col과 dot 연산을 하기 위해 w 를 transpose (1, 75) -> (75, 1)
    w_col = w_col.T
    print('w_col transposed shape =', w_col.shape)

    #
    # 2차원으로 변환되 이미지와 필터를 행렬 dot product 연산
    out = x_col.dot(w_col)
    print('out shape =', out.shape)

    #
    # dot product 의 결과를 (n, oh, ow, ?) 형태로 reshape
    out = out.reshape(1, 3, 3, -1)
    print('out shape =', out.shape)  # (1, 3, 3, 1) = (n, oh, ow, fn) 이미지 개수, oh, ow, fn
    # 필요한 shape 는 (n, fn, oh, ow)
    out = out.transpose(0, 3, 1, 2)  # 원하는 모양으로 axis(축) 을 옮긴다
    print('out shape =', out.shape)

    #####
    #####
    #####
    print('\n====================\n')
    # 연습
    # p.238 그림 7-12, p.244 그림 7-19 참조

    # 1.
    # 가상으로 생성한 이미지 데이터 x 와 2차원으로 변환한 x_col 을 사용
    print('x_col shape =', x_col.shape)
    # (3, 5, 5) 필터를 10개 생성 -> w.shape = (10, 3, 5, 5)
    w = np.random.randint(5, size = (10, 3, 5, 5))

    # 2.
    # w 를 변형(reshape): (fn, c * fh * fw)
    w_col = w.reshape(10, -1)  # 필터를 1차원으로 펼친다
    print('w_col shape =', w_col.shape)  # (10, 75)

    # 3.
    # x_col @ w.T
    w_col = w_col.T  # x_col 과 dot 하기 위해 transpose
    print('w_col shape =', w_col.shape)  # (75, 10)

    out = x_col.dot(w_col)
    print('x_col @ w.T out shape =', out.shape)  # (9, 10)

    # 4.
    # dot 연산 결과를 reshape: (n, oh, ow, fn)
    out = out.reshape(1, 3, 3, -1)
    print('out shape (n, oh, ow, c) =', out.shape)  # (1, 3, 3, 10)

    # 5.
    # reshape 된 결과에서 네번째 axis 이 두번째 axis 이 되도록 전치(transpose)
    out = out.transpose(0, 3, 1, 2)  # (n, fn, oh, ow) 로 axis 변경
    print('out shape (n, c, oh, ow) =', out.shape)  # (1, 10, 3, 3)

    #####
    #####
    #####
    print('\n====================\n')
    #
    # (3, 7, 7) shape 의 이미지 12개를 난수로 생성 -> (n, c, h, w) = (12, 3, 7, 7)
    x = np.random.randint(10, size = (12, 3, 7, 7))
    print('x shape =', x.shape)

    # (3, 5, 5) shape 의 필터 10개를 난수로 생성 -> (fn, c, fh, fw)
    w = np.random.randint(5, size = (10, 3, 5, 5))
    print('w shape =', w.shape)

    # stride = 1, padding = 0 일때 output height, output width = ? (3, 3)
    # oh = (h - fh + 2 * p) / s + 1 = 3
    # ow = (w - fw + 2 * p) / s + 1 = 3

    # 이미지 데이터 x 를 im2col 함수를 사용해서 x_col 로 변환 -> shape ? (108, 75)
    x_col = im2col(x, filter_h = 5, filter_w = 5, stride = 1, pad = 0)
    print('x_col shape =', x_col.shape)

    # 필터 w 를 x_col 과 dot 연산을 할 수 있도록 reshape & transpose: w_col -> shape ? (75, 10)
    # (fn, c * fh * fw)
    w_col = w.reshape(10, -1) # (10, 75) = (fn, c * fh * fw)
    w_col = w_col.T # (75, 10)
    print('w_col T shape =', w_col.shape)

    # x_col @ w_col = (108, 75) @ (75, 10) = (108, 10)
    out = x_col.dot(w_col)
    print('out shape =', out.shape)  # (108, 10)

    # @ 연산의 결과를 reshape & transpose
    # (n * oh * ow, fn)
    out = out.reshape(12, 3, 3, -1)  # (n, oh, ow, fn)
    print('out shape =', out.shape)  # (12, 3, 3, 10)

    out = out.transpose(0, 3, 1, 2)  # (n, fn, oh, ow)
    print('out shape =', out.shape)  # (12, 10, 3, 3)
