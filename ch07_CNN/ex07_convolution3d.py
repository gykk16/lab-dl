import numpy as np
from scipy.signal import correlate


def _convolution3d(x, w):
    # 들어오는 x 의 shape 는 (c, h, w)
    x_depth, x_row, x_col = x.shape[0], x.shape[1], x.shape[2]
    w_depth, w_row, w_col = w.shape[0], w.shape[1], w.shape[2]

    # 결과 ndarray
    row = x_row - w_row + 1
    col = x_col - w_row + 1
    depth = x_depth - w_depth + 1

    cc = np.zeros(shape = (depth, row, col))  # 비어 있는 결과 ndarray 생성

    for k in range(depth):
        for i in range(row):
            for j in range(col):
                x_sub = x[k:k + w_depth, i:i + w_row, j:j + w_col]
                # x_sub = x[:, i:i + w_row, j:j + w_col]
                cc[k, i, j] = np.sum(x_sub * w)
    return cc


def convolution3d(x, y):  # 선생님 답안
    '''
    x.shape = (c, h, w), y.shape = (c, fh, fw)
    h >= fh, w >= fw 라고 가정.

    '''
    h, w = x.shape[1], x.shape[2]  # source의 height/width
    fh, fw = y.shape[1], y.shape[2]  # 필터의 height/width
    oh = h - fh + 1  # 결과 행렬(output)의 height(row 개수)
    ow = w - fw + 1  # 결과 행렬(output)의 width(column 개수)
    # result = []
    result = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            x_sub = x[:, i:(i + fh), j:(j + fw)]
            # fma = np.sum(x_sub * y)
            # result.append(fma)
            result[i, j] = np.sum(x_sub * y)
    return result


if __name__ == '__main__':
    np.random.seed(114)

    # x : (3, 4, 4) shape의 3차원 ndarray
    x = np.random.randint(10, size = (3, 4, 4))
    print('x =\n', x)

    # w : (3, 3, 3) shape의 3차원 ndarray
    w = np.random.randint(10, size = (3, 3, 3))
    print('w =\n', w)

    conv1 = correlate(x, w, mode = 'valid')
    print('conv1 =\n', conv1)

    # 위와 동일한 결과를 함수로 작성
    conv2 = convolution3d(x, w)
    print('conv2 =\n', conv2)

    #####
    x = np.random.randint(10, size = (3, 28, 28))
    w = np.random.rand(3, 16, 16)

    conv1 = correlate(x, w, mode = 'valid')
    conv2 = convolution3d(x, w)

    print('conv1 =\n', conv1)
    print('conv2 =\n', conv2)

    print('conv1 shape =', conv1.shape)
    print('conv2 shape =', conv2.shape)
