'''
교차 엔트로피(Cross-Entropy):
    entropy = -true_value * log (exptected_value)
    entropy = -sum i [t_i * log (y_i)]


'''
import pickle

import numpy as np

from ch03.e11_teacher import forward
from dataset.mnist import load_mnist


def _cross_entropy(y_pred, y_true):
    delta = 1e-7  # log0 = -inf 가 되는 것은 방지하기 위해서 더해줄 것
    return -np.sum(y_true * np.log(y_pred + delta))


def cross_entropy(y_pred, y_true):
    if y_pred.ndim == 1:
        ce = _cross_entropy(y_pred, y_true)
    elif y_pred.ndim == 2:
        ce = _cross_entropy(y_pred, y_true) / len(y_pred)

    return ce


if __name__ == '__main__':
    (T_train, y_train), (T_test, y_test) = load_mnist(one_hot_label = True)

    y_true = y_test[:10]

    with open('../ch03/sample_weight.pkl', mode = 'rb') as f:
        network = pickle.load(f)

    y_pred = forward(network, T_test[:10])

    print(y_true[0])  # 숫자 7 이미지
    print(y_pred[0])  # 이미지가 7이 될 확률이 가장 큼

    print('ce =', cross_entropy(y_pred[0], y_true[0]))  # 0.0029

    print(y_true[8])  # 숫자 5 이미지
    print(y_pred[8])  # 이미지가 6이 될 확률이 가장 큼

    print('ce =', cross_entropy(y_pred[8], y_true[8]))  # 4.9094

    print('ce 평균 =', cross_entropy(y_pred, y_true))  # 0.5206

    # 만약 y_true 또는 y_pred가 one-hot-encoding이 사용되어 있지 않으면,
    # one-hot-encoding 형태로 변환해서 Cross-Entropy를 계산
    np.random.seed(1227)
    y_true = np.random.randint(10, size = 10)
    print('y_true =', y_true)
    y_true_2 = np.zeros((y_true.size, 10))
    for i in range(y_true.size):
        y_true_2[i][y_true[i]] = 1
    print(y_true_2)
