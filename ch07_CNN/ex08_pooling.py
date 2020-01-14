import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from dataset.mnist import load_mnist


def pooling1d(x, pool_size, stride = 1):
    n = x.shape[0]  # len(x)
    result_size = (n - pool_size) // stride + 1
    result = np.zeros(result_size)
    for i in range(result_size):
        x_sub = x[(i * stride):(i * stride) + pool_size]
        result[i] = np.max(x_sub)
    return result


def pooling2d(x, pool_h, pool_w, stride = 1):  # 선생님 답안
    '''

    :param x: 2-dim ndarray
    :param pool_h: pooling window height
    :param pool_w: pooling window width
    :param stride: 보폭
    :return: max-pooling
    '''

    h, w = x.shape[0], x.shape[1]  # 원본 데이터의 height/widtgh
    oh = (h - pool_h) // stride + 1  # 출력 배열의 height
    ow = (w - pool_w) // stride + 1  # 출력 배열의 width
    output = np.zeros((oh, ow))  # 출력 배열 초기화
    for i in range(oh):
        for j in range(ow):
            x_sub = x[(i * stride):(i * stride) + pool_h, (j * stride):(j * stride) + pool_w]
            output[i, j] = np.max(x_sub)
    return output


if __name__ == '__main__':
    np.random.seed(114)
    x = np.random.randint(10, size = 10)
    print(x)

    pooled = pooling1d(x, pool_size = 4, stride = 2)
    print('pooled =', pooled)

    pooled = pooling1d(x, pool_size = 4, stride = 3)
    print('pooled =', pooled)

    #####
    x = np.random.randint(100, size = (8, 8))
    print('x =\n', x)

    pooled = pooling2d(x, pool_h = 2, pool_w = 2, stride = 2)
    print('pooled =\n', pooled)

    pooled = pooling2d(x, pool_h = 4, pool_w = 4, stride = 4)
    print('pooled =\n', pooled)

    #####
    x = np.random.randint(100, size = (5, 5))
    print('x =\n', x)
    pooled = pooling2d(x, pool_h = 3, pool_w = 3, stride = 2)
    print('pooled =\n', pooled)

    pooled = pooling2d(x, pool_h = 2, pool_w = 2, stride = 2)
    print('pooled =\n', pooled)  # 경계에 있는 값들은 버려짐 (마지막 row, col 무시)

    #####

    # MNIST 데이터 세트를 로드
    (x_train, y_train), (x_test, y_test) = load_mnist(normalize = False,
                                                      flatten = False)
    # 손글씨 이미지를 하나를 선택: shape=(1, 28, 28) -> (28, 28) 변환
    img = x_train[0]
    print('img.shape:', img.shape)
    # img_2d = img.reshape((28, 28))
    img_2d = img[0, :, :]
    print('img_2d.shape:', img_2d.shape)

    # 선택된 이미지를 pyplot을 사용해서 출력
    plt.imshow(img_2d, cmap = 'gray')
    plt.show()

    # window shape=(4, 4), stride=4 pooling -> output shape=(7,7)
    img_pooled = pooling2d(img_2d, 4, 4, 4)
    print('img_pooled.shape:', img_pooled.shape)

    # pyplot으로 출력
    plt.imshow(img_pooled, cmap = 'gray')
    plt.show()

    #####
    # 이미지 파일을 오픈: (height, width, color)
    img = Image.open('sample.jpg')  # 1920 x 1075
    img_pixel = np.array(img)
    print('img_pixel shape =', img_pixel.shape)  # (1075, 1920, 3)

    # Red, Green, Blue에 해당하는 2차원 배열들을 추출
    img_r = img_pixel[:, :, 0]  # Red panel
    img_g = img_pixel[:, :, 1]  # Green panel
    img_b = img_pixel[:, :, 2]  # Blue panel

    # 각각의 2차원 배열을 window shape=(16, 16), stride=16으로 pooling
    img_r_pooled = pooling2d(img_r, 16, 16, 16)
    img_g_pooled = pooling2d(img_g, 16, 16, 16)
    img_b_pooled = pooling2d(img_b, 16, 16, 16)

    # pooling된 결과(shape)를 확인, pyplot
    print('img_r_pooled shape:', img_r_pooled.shape)  # (67, 120)
    plt.imshow(img_r_pooled)
    plt.show()

    img_pooled = np.array([img_r_pooled, img_g_pooled, img_b_pooled]).astype(np.uint8)
    print('img_pooled shape =', img_pooled.shape)
    img_pooled = np.moveaxis(img_pooled, 0, 2)  # 0번 축과 2번축과 바꾸겠다
    print('img_pooled shape =', img_pooled.shape)

    plt.imshow(img_pooled)
    plt.show()
