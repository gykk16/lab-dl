'''
4차원 데이터를 2차원으로 변화 후에 max pooling을 구현

'''
import numpy as np

from common.util import im2col

if __name__ == '__main__':
    np.random.seed(116)

    # 가상의 이미지 데이터 (c, h, w) = (3, 4, 4) 1개를 난수로 생성 -> (1, 3, 4, 4)
    x = np.random.randint(10, size = (1, 3, 4, 4))
    print(x, '\nshape :', x.shape)

    col = im2col(x, filter_h = 2, filter_w = 2, stride = 2, pad = 0)
    print(col, '\nshape :', col.shape)  # (4, 12) = (n*oh*ow, c*fh*fw)

    # max pooling: 채널별로 최댓값을 찾음
    # 채널별 최댓값을 쉽게 찾기 위해서 2차원 배열의 shape 을 변환
    # -> 현환된 행렬의 각 행에는, 채널별로 윈도우에 포함된 값들로만
    col = col.reshape(-1, 2 * 2)  # (-1, fh*fw)
    print(col, '\nshape :', col.shape)

    # 각 행(row)에서 최댓값을 찾음
    out = np.max(col, axis = 1)
    print(out, '\nshape :', out.shape)

    # 1차원 pooling 의 결과를 4차원으로 변환: (n, oh, ow, c) -> (n, c, oh, ow)
    # 채널(color depth) 축이 가장 마지막 축이 되도록 변환 후,
    # 채널 축이 2번째 축이 되도록 transpose 함수를 사용해서 축의 위치를 변경해줌.
    out = out.reshape(1, 2, 2, 3).transpose(0, 3, 1, 2)
    print(out)
    print(out.shape)
