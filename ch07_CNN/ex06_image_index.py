import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from dataset.mnist import load_mnist

if __name__ == '__main__':
    # 이미지 파일 오픈
    img = Image.open('sample.jpg')
    # 이미지 객체를 numpy 배열 형태(3차원 배열)로 변환
    img_pixel = np.array(img)
    print('img_pixel:', img_pixel.shape)  # (height, width, color-depth)
    # print(img_pixel)

    (x_train, y_train), (x_test, y_test) = load_mnist(normalize = False, flatten = False)
    print('x_train =', x_train.shape)

    print('x_train', x_train.shape)  # (samples, color, height, width)
    print('x_train[0] shape =', x_train[0].shape)  # (color, height, width)
    # plt.imshow(x_train[0])
    # (c, h, w) 형식의 이미지 데이터는 matplotlib에서 사용 못함
    # (h, w, c) 형식으로 변환 필요
    num_img = np.moveaxis(x_train[0], 0, 2)
    print('num_img shape =', num_img.shape)  # (h, w, c)
    num_img = num_img.reshape((28, 28))  # 단색인 경우 2차원으로 변환
    plt.imshow(num_img)
    plt.show()
