import matplotlib.pyplot as plt
import numpy as np

from ch06.e02_sgd import Sgd
from common.multi_layer_net_extend import MultiLayerNetExtend
from dataset.mnist import load_mnist

np.random.seed(110)

# x = np.random.rand(20)  # 0.0 ~ 0.999999... 균등 분포에서 뽑은 난수
# print(x)
# mask = x > 0.5
# print(mask)
# print(x * mask)

# 데이터 준비
(X_train, Y_train), (X_test, Y_test) = load_mnist(one_hot_label = True)

# 신경망 생성
dropout_ratio = 0.1
neural_net = MultiLayerNetExtend(input_size = 784,
                                 hidden_size_list = [100, 100, 100, 100, 100],
                                 output_size = 10,
                                 use_dropout = True,
                                 dropout_ration = dropout_ratio)

X_train = X_train[:500]
Y_train = Y_train[:500]
X_test = X_test[:500]
Y_test = Y_test[:500]

epochs = 200  # 1 에포크: 모든 학습 데이터가 1번씩 학습된 경우
mini_batch_size = 100  # 1번 forward에 보낼 데이터 샘플 개수
train_size = X_train.shape[0]
iter_per_epoch = int(max(train_size / mini_batch_size, 1))
# 학습하면서 학습/테스트 데이터의 정확도를 각 에포크마다 기록
train_accuracies = []
test_accuracies = []

optimizer = Sgd(learning_rate = 0.01)  # optimizer

for epoch in range(epochs):
    indices = np.arange(train_size)
    np.random.shuffle(indices)
    for i in range(iter_per_epoch):
        iter_idx = indices[(i * mini_batch_size):((i + 1) * mini_batch_size)]
        x_batch = X_train[iter_idx]
        y_batch = Y_train[iter_idx]
        gradients = neural_net.gradient(x_batch, y_batch)
        optimizer.update(neural_net.params, gradients)

    train_acc = neural_net.accuracy(X_train, Y_train)
    train_accuracies.append(train_acc)
    test_acc = neural_net.accuracy(X_test, Y_test)
    test_accuracies.append(test_acc)
    print(f'epoch #{epoch}: train={train_acc}, test={test_acc}')

x = np.arange(epochs)
plt.plot(x, train_accuracies, label = 'Train')
plt.plot(x, test_accuracies, label = 'Test')
plt.legend()
plt.title(f'Dropout (ratio={dropout_ratio})')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()
