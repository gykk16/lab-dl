'''
CNN(Convolutional Neural Network, 합성곱 신경망)
원래 convolution 연산은 영상, 음성 처리(image/audio processing)에서 신호를 변환하기 위한 연산으로 사용


'''

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from scipy.signal import convolve, correlate

# jpg 파일 오픈
img = Image.open('sample.jpg')
img_pixel = np.array(img)  # 3차원
print(img_pixel.shape)

# $$$
# (1075, 1920, 3) : (height, width, color-depth) => (row, col, RGB)  24-bit color
# 머신 러닝 라이브러리에 따라서 color 표기의 위치가 다르다
# TenserFlow: channel-last 방식. color-depth 가 3차원 배열의 마지막 차원
# Theano: channel-first 방식. color-depth 가 3차원 배열의 첫번째 차원 (color, h, w)
# Keras: 두가지 모두 지원


plt.imshow(img_pixel)
plt.show()

# 이미지의 RED 값 정보 => 2차원
print(img_pixel[:, :, 0])  # height 전체, width 전체, 첫전째 장 출력

# (3, 3, 3) 필터
filter = np.zeros((3, 3, 3))
filter[1, 1, 2] = 1.0
print(filter)
transformed = convolve(img_pixel, filter, mode = 'same') / 255
plt.imshow(transformed.astype(np.uint8))    # np.uint8 : 부호가 없는 8-bit int
plt.show()

