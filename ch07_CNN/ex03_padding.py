'''
Padding

    - pad_width = 1 : 패딩 크기, ( befor(앞), after(뒤) ) 값을 따로 줄수고 있다
    - mode = 'constant' : 패딩 숫자 타입 (상수로 padding 하겠다)
    - constant_values = 0 : 지정할 값


'''

import numpy as np

if __name__ == '__main__':
    np.random.seed(113)

    # 1차원 ndarray
    x = np.arange(1, 6)
    print(x)

    x_pad = np.pad(x,
                   pad_width = 1,
                   mode = 'constant',
                   constant_values = 0)
    print(x_pad)

    x_pad = np.pad(x, pad_width = (2, 3), mode = 'constant', constant_values = 0)  # 앞은 2, 뒤는 3 으로 패딩 하겠다
    print(x_pad)

    x_pad = np.pad(x, pad_width = 2, mode = 'minimum')  # 데이터 중에서 최소값으로 패딩 하겠다
    print(x_pad)

    # 2차원 ndarray
    x = np.arange(1, 10).reshape(3, 3)
    x_pad = np.pad(x, pad_width = 1, mode = 'constant', constant_values = 0)
    print(x_pad)

    x = np.arange(1, 10).reshape(3, 3)
    # axis = 0 방향 before-padding = 1
    # axis = 0 방향 after-padding = 2
    # axis = 1 방향 before-padding = 1
    # axis = 1 방향 after-padding = 2
    x_pad = np.pad(x, pad_width = (1, 2), mode = 'constant', constant_values = 0)
    print(x_pad)

    # (1, 2) = (axis = 0 before, axis = 0 after)
    # (3, 4) = (axis = 1 before, axis = 1 after)
    x_pad = np.pad(x, pad_width = ((1, 2), (3, 4)),
                   mode = 'constant', constant_values = 0)
    print(x_pad)
