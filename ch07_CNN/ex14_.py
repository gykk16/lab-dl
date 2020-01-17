'''
SimpleConvNet (간단한 CNN)을 활용한 MNIST 손글씨 이미지 데이터 분류

'''
import numpy as np
import matplotlib.pyplot as plt

from ch07_CNN.simple_convnet import SimpleConvNet
from common.trainer import Trainer
from dataset.mnist import load_mnist

# 데이터 로드
(X_train, Y_train), (X_test, Y_test) = load_mnist(flatten = False)


# 테스트 시간을 줄이기 위해서 데이터 사이즈를 줄임
# X_train, Y_train = X_train[:5000], Y_train[:5000]
# X_test, Y_test = X_test[:1000], Y_test[:1000]

# CNN 생성
cnn = SimpleConvNet()  # 파라미터 기본 값 사용

# 테스트 도우미 클래스
trainer = Trainer(network = cnn,
                  x_train = X_train,
                  t_train = Y_train,
                  x_test = X_test,
                  t_test = Y_test,
                  epochs = 20,
                  mini_batch_size = 100,
                  optimizer = 'Adam',
                  optimizer_param = {'lr': .01},
                  evaluate_sample_num_per_epoch = 1000)

# 테스트 실행
trainer.train()

# 학습이 끝난 후 파라미터들을 파일에 저장
cnn.save_params(file_name = 'cnn_params.pkl')

# 그래프 (x축 : epoch, y축 : accuracy)
x = np.arange(20)
plt.plot(x, trainer.train_acc_list, label = 'train accuracy')
plt.plot(x, trainer.test_acc_list, label = 'test accuracy')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Accuracy')
plt.show()