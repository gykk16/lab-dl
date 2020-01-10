'''
Affine, ReLU, SoftmaxWithLoss 클래스들을 이용한 신경망 구현

'''
import numpy as np

from ch05.e05_relu import Relu
from ch05.e07_affine import Affine
from ch05.e08_softmaxwithloss import SoftmaxWithLoss

np.random.seed(106)

# 입력 데이터: (1, 2) shape의 ndarray
X = np.random.rand(2).reshape((1, 2))
print('X =', X)
# 실제 레이블(정답):
Y_true = np.array([1, 0, 0])
print('Y =', Y_true)

# 첫번째 은닉층(hidden layer)에서 사용할 가중치/편향 행렬
# 첫번째 은닉층의 뉴런 개수 : 3개
# W1 shape: (2, 3), b1 shape : (3,)
W1 = np.random.randn(2, 3)
b1 = np.random.rand(3)
print('W1 =', W1)
print('b1 =', b1)

affine1 = Affine(W1, b1)
relu = Relu()

# 출력층의 뉴런 개수 : 3개
# W shape : (3, 3), b shape : (3,)
W2 = np.random.randn(3, 3)
b2 = np.random.rand(3)
print('W2 =', W2)
print('b2 =', b2)

affine2 = Affine(W2, b2)
last_layer = SoftmaxWithLoss()

# 각 레이어들을 연결 : forward propagation
Y = affine1.forward(X)
print('Y shape:', Y.shape)

Y = relu.forward(Y)
print('Y shape:', Y.shape)

Y = affine2.forward(Y)
print('Y shape:', Y.shape)

loss = last_layer.forward(Y, Y_true)
print('loss =', loss)  # cross-entropy
print('y_pred =', last_layer.y_pred)  # [[0.22573711 0.2607098  0.51355308]]

# gradient 를 계산하기 위해서 역전파 (back propagation)
learning_rate = .1

dout = last_layer.backward(1)
print('dout 1 =', dout)

dout = affine2.backward(dout)
print('dout 2 =', dout)
print('dW2 =', affine2.dW)
print('db2 =', affine2.db)

dout = relu.backward(dout)
print('dout 3 =', dout)

dout = affine1.backward(dout)
print('dout 4 =', dout)
print('dW1 =', affine1.dW)
print('db1 =', affine1.db)

#####
# 가중치/편향 행렬을 학습률과 gradient 를 이용해서 수정
W1 -= learning_rate * affine1.dW
b1 -= learning_rate * affine1.db
W2 -= learning_rate * affine2.dW
b2 -= learning_rate * affine2.db

#####
# 수정된 가중치/편향 행렬들을 이용해서 다시 forward propagation
Y = affine1.forward(X)
Y = relu.forward(Y)
Y = affine2.forward(Y)
loss = last_layer.forward(Y, Y_true)
print('loss =', loss)  # cross-entropy
print('y_pred =', last_layer.y_pred)  # [[0.24797233 0.25787792 0.49414975]]

#####
print('#########')
# 미니 배치 (mini-batch)
X = np.random.rand(3, 2)
Y_true = np.identity(3)  # [[1. 0. 0.] [0. 1. 0.] [0. 0. 1.]]
# forward -> backward -> W, b 수정 -> forward

Y = affine1.forward(X)
Y = relu.forward(Y)
Y = affine2.forward(Y)
loss = last_layer.forward(Y, Y_true)
print('loss =', loss)
# print('y_pred =', last_layer.y_pred)
#
dout = 1
dout = last_layer.backward(dout)
dout = affine2.backward(dout)
dout = relu.backward(dout)
dout = affine1.backward(dout)
print('dout =', dout)
#
W1 -= learning_rate * affine1.dW
b1 -= learning_rate * affine1.db
W2 -= learning_rate * affine2.dW
b2 -= learning_rate * affine2.db
#
Y = affine1.forward(X)
Y = relu.forward(Y)
Y = affine2.forward(Y)
loss = last_layer.forward(Y, Y_true)
print('loss =', loss)
# print('y_pred =', last_layer.y_pred)

