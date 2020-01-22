# -*- coding: utf-8 -*-
"""mnist_cnn.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1hjayANaZvDUc1UmpxFy-4kt57OVbxcrU

Colab에서 GPU 사용하기:<br/>
메뉴 > 수정 > 노트 설정 > 하드웨어 가속기 > GPU 선택 > Save
"""

# Commented out IPython magic to ensure Python compatibility.
# TensorFlow 버전 선택
# %tensorflow_version 2.x

# TensorFlow 버전과 GPU 사용 여부 확인
import tensorflow as tf

print(tf.__version__)  # 2.1.x
print(tf.test.gpu_device_name())  # GPU:0

# 클래스 및 모듈 import
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# MNIST 데이터 로드
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
print(f'X_train: {X_train.shape}, Y_train: {Y_train.shape}')
print(f'X_test: {X_test.shape}, Y_test: {Y_test.shape}')

# MNIST 데이터를 CNN의 입력 데이터로 사용할 수 있도록 shape을 변환
# X_train, X_test를 (samples, h, w, 1) shape으로 변환
n, h, w = X_train.shape
X_train = X_train.reshape(*X_train.shape, 1)
X_test = X_test.reshape(*X_test.shape, 1)

# Y_train, Y_test를 one-hot-encoding 변환
Y_train = to_categorical(Y_train, 10, dtype = 'float16')
Y_test = to_categorical(Y_test, 10, dtype = 'float16')

# X_train, X_test를 0. ~ 1. 사이의 값으로 정규화(normalization)
X_train = X_train.astype('float16') / 255
X_test = X_test.astype('float16') / 255

print(f'X_train: {X_train.shape}, Y_train: {Y_train.shape}')
print(f'X_test: {X_test.shape}, Y_test: {Y_test.shape}')

# 신경망 모델 생성 - Sequential 클래스 인스턴스 생성
model = Sequential()

# 신경망 모델에 은닉층, 출력층 계층(layers)들을 추가
# Conv2D -> MaxPool2D -> Flatten -> Dense -> Dense
# Conv2D 활성화 함수: ReLU
# Dense 활성화 함수: ReLU, Softmax
model.add(Conv2D(filters = 32,  # 필터 개수
                 kernel_size = (3, 3),  # 필터 height/width
                 activation = 'relu',  # 활성화 함수
                 input_shape = (28, 28, 1)))  # 입력 데이터 shape
model.add(MaxPool2D(pool_size = 2))
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))  # 완전 연결 은닉층
model.add(Dense(10, activation = 'softmax'))  # 출력층

# 신경망 모델 컴파일
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])

# 신경망 모델의 성능 향상이 없는 경우 중간에 epoch를 빨리 중지시키기 위해서
early_stop = EarlyStopping(monitor = 'val_loss',
                           verbose = 1,
                           patience = 10)

# 신경망 학습
history = model.fit(X_train, Y_train,
                    batch_size = 200,
                    epochs = 50,
                    verbose = 1,
                    callbacks = [early_stop],
                    validation_data = (X_test, Y_test))

# 테스트 데이터를 사용해서 신경망 모델을 평가
# 테스트 데이터의 Loss, Accuracy
eval = model.evaluate(X_test, Y_test)
print(f'Test loss: {eval[0]}, accuracy: {eval[1]}')

# 학습 데이터와 테스트 데이터의 Loss 그래프
train_loss = history.history['loss']
test_loss = history.history['val_loss']

x = range(len(train_loss))
plt.plot(x, train_loss, marker = '.', color = 'red', label = 'Train loss')
plt.plot(x, test_loss, marker = '.', color = 'blue', label = 'Test loss')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

# 학습 데이터, 테스트 데이터의 정확도 그래프
train_acc = history.history['accuracy']
test_acc = history.history['val_accuracy']

plt.plot(x, train_acc, marker = '.', c = 'red', label = 'Train Acc.')
plt.plot(x, test_acc, marker = '.', c = 'blue', label = 'Test Acc.')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()
