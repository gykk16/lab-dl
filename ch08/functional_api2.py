"""
GoogLeNet: p.271 그림 8-11
ResNet: p.272 그림 8-12
"""
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Add, concatenate

##### GoogLeNet

# 입력 텐서 생성
input_tensor = Input(shape = (784,))
# 은닉층 생성
x1 = Dense(64, activation = 'relu')(input_tensor)
x2 = Dense(64, activation = 'relu')(input_tensor)
# 두 개의 output 텐서를 연결
concat = concatenate([x1, x2])
# 연결된 텐서를 다음 계층으로 전달
x = Dense(32, activation = 'relu')(concat)
# 출력층 생성
output_tensor = Dense(10, activation = 'softmax')(x)

# 모델 생성
model = Model(input_tensor, output_tensor)
model.summary()


##### ResNet

print()
input_tensor = Input(shape = (784,))
sc = Dense(32, activation = 'relu')(input_tensor)
x = Dense(32, activation = 'relu')(sc)
x = Dense(32, activation = 'relu')(x)
x = Add()([x, sc])
output_tensor = Dense(10, 'softmax')(x)

model = Model(input_tensor, output_tensor)
model.summary()
