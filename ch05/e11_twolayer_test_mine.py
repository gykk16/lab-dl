'''
2층 신경망 테스트

'''
import numpy as np
import matplotlib.pyplot as plt

from ch05.e10_twolayer import TwoLayerNetwork
from dataset.mnist import load_mnist

if __name__ == '__main__':
    np.random.seed(106)
    # MNIST 데이터를 로드
    (X_train, Y_train), (X_test, Y_test) = load_mnist(one_hot_label = True)

    # 2층 신경망 생성
    neural_net = TwoLayerNetwork(input_size = 784, hidden_size = 32, output_size = 10)

    epochs = 100  # 100번 반복
    batch_size = 100  # 한번에 학습시키는 입력 데이터 개수
    learning_rate = .1  #

    iter_size = max(X_train.shape[0] // batch_size, 1)
    print(iter_size)

    # 학습 데이터를 랜덤하게 썪음
    idx = np.arange(len(X_train))
    np.random.shuffle(idx)

    loss_data = []
    acc_data = []

    acc_test1 = neural_net.accuracy(X_test, Y_test)

    for epoch in range(100):
        for i in range(0, len(X_train), batch_size):
            # for j in range(iter_size):
            # 처음 batch_size 개수 만큼의 데이터를 입력으로 해서 gradient 계산
            gradients = neural_net.gradient(X_train[i:i + batch_size], Y_train[i:i + batch_size])
            # 가중치/편향 행렬들을 수정
            for key in gradients:
                neural_net.params[key] -= learning_rate * gradients[key]
        # loss 를 계산해서 출력
        print('epoch =', epoch)
        loss = neural_net.loss(X_train, Y_train)
        loss_data.append(loss)
        print('loss =', loss)
        acc = neural_net.accuracy(X_train, Y_train)
        acc_data.append(acc)
        print('accuracy =', acc)

    acc_test2 = neural_net.accuracy(X_test, Y_test)
    print('X_test accuracy 1 =', acc_test1)
    print('X_test accuracy 2 =', acc_test2)

    # line 23 ~ 28까지의 과정을 100회(epochs)만큼 반복
    # 반복할 때마다 학습 데이터 세트를 무작위로 섞는(shuffle) 코드를 추가
    # 각 epoch마다 테스트 데이터로 테스트를 해서 accuracy를 계산
    # 100번의 epoch가 끝났을 때, epochs-loss, epochs-accuracy 그래프를 그림.

    epoch_data = list(range(0, 100))
    plt.plot(epoch_data, acc_data, label = 'accuracy')
    plt.plot(epoch_data, loss_data, label = 'loss')
    plt.legend()
    plt.show()
