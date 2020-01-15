import numpy as np


def cross_correlation_1d(x, w):
    x_len = len(x)
    w_len = len(w)

    n = x_len - w_len + 1
    cc = np.zeros(n)
    for i in range(n):
        x_sub = x[i:i + w_len]
        cc[i] = np.sum(x_sub * w)
    return cc


def cross_correlation_2d(x, w):
    x_row, x_col = x.shape[0], x.shape[1]
    w_row, w_col = w.shape[0], w.shape[1]
    row = x_row - w_row + 1
    col = x_col - w_row + 1
    cc = np.zeros((row, col))

    for i in range(row):
        for j in range(col):
            x_sub = x[i:i + w_row, j:j + w_col]
            cc[i, j] = np.sum(x_sub * w)
    return cc


def cross_correlation_3d(x, w):
    # (c, h, w)
    c, h, w = x.shape
    c, fh, fw = w.shape
    oh = h - fh + 1
    ow = w - fw + 1
    oc = 1

    cc = np.zeros(shape = (c, oh, ow))

    for k in range(oc):
        for i in range(oh):
            for j in range(ow):
                x_sub = x[:, i:i + fh, j:j + fw]
                cc[k, i, j] = np.sum(x_sub * w)
    return cc



if __name__ == '__main__':
    x = np.arange(1, 6)
    w = np.array([2, 0, 1])

    cc = cross_correlation_1d(x, w)
    print(cc)

    x = np.arange(1, 10).reshape(3, 3)
    w = np.array([[2, 0],
                  [0, 0]])

    cc = cross_correlation_2d(x, w)
    print(cc)


    x = np.arange()
