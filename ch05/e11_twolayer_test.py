'''
2층 신경망 테스트

'''
import pickle

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
    batch_size = 128  # 한번에 학습시키는 입력 데이터 개수
    learning_rate = .1  #

    # iter_size:한번의 epoch(학습이 한번 완료되는 주기)에 필요한 반복 회수
    # -> 가중치/편향 행렬들이 한번의 학습 주기(epoch)에서 변경되는 회수
    iter_size = max(X_train.shape[0] // batch_size, 1)
    print(iter_size)

    train_losses = []  # 각 epoch마다 학습 데이터의 손실을 저장할 리스트.
    train_accuracies = []  # 각 epoch마다 학습 데이터의 정확도를 저장할 리스트.
    test_accuracies = []  # 각 epoch마다 테스트 데이터의 정확도를 저장할 리스트.
    for epoch in range(epochs):
        # 학습 데이터를 랜덤하게 섞음(shuffle).
        # 인덱스 0 ~ 59,999를 랜덤하게 섞은 후, 섞인 인덱스로 X_train과 Y_train을 선택.
        idx = np.arange(len(X_train))  # [0, 1, 2, ..., 59999]
        np.random.shuffle(idx)
        for i in range(iter_size):
            # batch_size 개수만큼의 학습 데이터를 입력으로 해서 gradient 계산
            X_batch = X_train[idx[i * batch_size:(i + 1) * batch_size]]
            Y_batch = Y_train[idx[i * batch_size:(i + 1) * batch_size]]
            gradients = neural_net.gradient(X_batch, Y_batch)
            # 가중치/편향 행렬들을 수정
            for key in neural_net.params:
                neural_net.params[key] -= learning_rate * gradients[key]

        # loss를 계산해서 출력
        train_loss = neural_net.loss(X_train, Y_train)
        train_losses.append(train_loss)
        print('train_loss:', train_loss)
        # 학습 데이터 accuracy를 계산해서 출력
        train_acc = neural_net.accuracy(X_train, Y_train)
        train_accuracies.append(train_acc)
        print('train_acc:', train_acc)
        # 테스트 데이터 accuracy를 계산해서 출력
        test_acc = neural_net.accuracy(X_test, Y_test)
        test_accuracies.append(test_acc)
        print('test_acc:', test_acc)

    # epoch ~ loss 그래프
    x = range(epochs)
    plt.plot(x, train_losses, label = 'train loss')
    plt.plot(x, train_accuracies, label = 'train accuracy')
    plt.plot(x, test_accuracies, label = 'test accuracy')
    plt.title('Loss - Cross Entropy')
    plt.legend()
    plt.show()

    # 신경망에서 학습이 모든 끝난 후, 파라미터(가중치/편향 행렬들)들을 파일에 저장
    # pickle 이용
    with open('params.pickle', mode = 'wb') as f:
        pickle.dump(neural_net.params, f)

    # line 23 ~ 28까지의 과정을 100회(epochs)만큼 반복
    # 반복할 때마다 학습 데이터 세트를 무작위로 섞는(shuffle) 코드를 추가
    # 각 epoch마다 테스트 데이터로 테스트를 해서 accuracy를 계산
    # 100번의 epoch가 끝났을 때, epochs-loss, epochs-accuracy 그래프를 그림.
