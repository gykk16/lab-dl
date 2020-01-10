# p. 137

import numpy as np

from dataset.mnist import load_mnist


class TwoLayerNetwork:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = .01):
        '''
        입력 784(28x28)개
            천번째 층(Layer)의 뉴련 개수: 32개
            출력 층(Layer)의 뉴련 개수: 10개

            가중치 행렬(W1, w2), bias 행렬(b1, b2)을 난수로 생성

        :param input_size:
        :param hidden_layer:
        :param output_layer:
        :param weight_init_std:
        '''

        # self.input_size = input_size
        # self.hidden_size = hidden_layer
        # self.output_size = output_layer

        np.random.seed(1231)

        self.params = dict()  # weight/bias 행렬들을 저장하는 dict

        # x(1, 784) @ W1(784, 32) + (1, 32)
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)

        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        if x.ndim == 1:
            max_x = np.max(x)
            x -= max_x  # overflow 를 방지하기 위해서
            result = np.exp(x) / np.sum(np.exp(x))
        else:  # ndim == 2
            x_t = x.T  # 행렬 x의 전치행렬(transpose)를 만듦.
            max_x = np.max(x_t, axis = 0)
            x_t -= max_x
            result = np.exp(x_t) / np.sum(np.exp(x_t), axis = 0)
            result = result.T
        return result

    def predict(self, x):
        '''
        data -> 은닉층 -> 출력층 -> 예측값
        sigmoid(data @ W1 + b1) = z
        softmax(z @ W2 + b2) = y

        :param x: y_test
        :return: y
        '''

        a1 = x.dot(self.params['W1']) + self.params['b1']
        z = self.sigmoid(a1)
        a2 = z.dot(self.params['W2']) + self.params['b2']
        y = self.softmax(z.dot(self.params['W2']) + self.params['b2'])

        return y

    def accuracy(self, x, y_true):
        '''

        :param x: 예측값을 구하고 싶은 데이터, 2차원 배열
        :param y_true: 실제 레이블, 2차원 배열
        :return: 정확도
        '''

        y_pred = self.predict(x)
        predictions = np.argmax(y_pred, axis = 1)
        true_vals = np.argmax(y_true, axis = 1)

        return np.mean(true_vals == predictions)

    def loss(self, x, y_true):
        y_pred = self.predict(x)
        ce = self.cross_entropy(y_true, y_pred)

        return ce

    def cross_entropy(self, y_true, y_pred):
        delta = 1e-7
        if y_pred.ndim == 1:
            # 1차원 배열인 경우, 행의 개수가 1개인 2차원 배열로 변환,
            y_pred = y_pred.reshape((1, y_pred.size))
            y_true = y_true.reshape((1, y_true.size))
            # y_true 는 one-hot-encoding 되어 있다고 가정.
        # y_true 에서 1이 있는 컬럼 위치(인덱스)를 찾음.
        true_vals = np.argmax(y_true, axis = 1)
        n = y_pred.shape[0]  # 2차원 배열의 shape: (row, col)
        rows = np.arange(n)  # row index, [0, 1, 2, 3, ... ]
        # y_pred[[0, 1, 2], [3, 3, 9]]   [[row index], [해당 row에 1이 있는 위치]]
        log_p = np.log(y_pred[rows, true_vals])
        entropy = -np.sum(log_p) / n
        return entropy

    def gradients(self, x, y_true):
        loss_fn = lambda w: self.loss(x, y_true)
        gradients = dict()  # W1, b1, W2, b2의 gradient 를 저장할 dict
        for key in self.params:
            gradients[key] = self.numerical_gradient(loss_fn, self.params[key])
        return gradients

    def numerical_gradient(self, fn, x):
        h = 1e-4  # 0.0001
        gradient = np.zeros_like(x)
        with np.nditer(x, flags = ['c_index', 'multi_index'], op_flags = ['readwrite']) as it:
            while not it.finished:
                i = it.multi_index
                ith_value = it[0]  # 원본 데이터를 임시 변수에 저장
                it[0] = ith_value + h  # 원본 값을 h 만큼 증가
                fh1 = fn(x)  # f(x + h)
                it[0] = ith_value - h  # 원본 값을 h 만큼 감소
                fh2 = fn(x)  # f(x - h)
                gradient[i] = (fh1 - fh2) / (2 * h)
                it[0] = ith_value
                it.iternext()

        return gradient


if __name__ == '__main__':
    # 신경망 생성
    neural_net = TwoLayerNetwork(input_size = 784,
                                 hidden_size = 32,
                                 output_size = 10)

    # W1, W2, b1, b2의 shape 를 확인
    print(f'W1 shape = {neural_net.params["W1"].shape}, b1 shape = {neural_net.params["b1"].shape}')
    print(f'W2 shape = {neural_net.params["W2"].shape}, b2 shape = {neural_net.params["b2"].shape}')

    # 신경망 클래스의 predict() 메소드 테스트
    # mnist 데이터 세트를 로드(dataset.load_mnist 사용)
    # X_train[0]를 신경망에 전파(propagate)시켜서 예측값 확인
    # X_train[:5]를 신경망에 전파시켜서 예측값 확인

    (X_train, y_train), (X_test, y_test) = load_mnist(one_hot_label = True)
    y_pred0 = neural_net.predict(X_train[:5])
    print('y_pred0 =', y_pred0)
    print('y_true0 =', y_train[:5])

    acc = neural_net.accuracy(X_train[:5], y_train[:5])
    print('acc =', acc)

    ce = neural_net.loss(X_train[:5], y_train[:5])
    print('cross entropy =', ce)

    # gradient 메소드 테스트
    gradients = neural_net.gradients(X_train[:100], y_train[:100])
    for key in gradients:
        print(key, np.sum(gradients[key]))

    # 찾은 gradient 를 이용해서 weight/bias 행렬들을 업데이트
    lr = .1  # 학습률(learning rate)
    for key in gradients:
        neural_net.params[key] -= lr * gradients[key]

    epoch = 10
    for i in range(epoch):
        for j in range(600):
            gradients = neural_net.gradients(X_train[j * 100:(j + 1) * 100], y_train[j * 100:(j + 1) * 100])
            for key in gradients:
                neural_net.params[key] -= lr * gradients[key]
