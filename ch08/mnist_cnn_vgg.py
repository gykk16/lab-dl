import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
# from tensorflow_core.python.keras import Sequential
from tensorflow_core.python.keras.callbacks import EarlyStopping
from tensorflow_core.python.keras.datasets import mnist
from tensorflow_core.python.keras.layers import Conv2D, MaxPool2D, Flatten, Dropout, Dense
from tensorflow_core.python.keras.utils.np_utils import to_categorical

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
print(f'X_train: {X_train.shape}, Y_train: {Y_train.shape}')
print(f'X_test: {X_test.shape}, Y_test: {Y_test.shape}')

X_train = X_train.reshape(*X_train.shape, 1).astype('float16') / 255
X_test = X_test.reshape(*X_test.shape, 1).astype('float16') / 255
Y_train = to_categorical(Y_train, 10, dtype = 'float16')
Y_test = to_categorical(Y_test, 10, dtype = 'float16')

print(f'X_train: {X_train.shape}, Y_train: {Y_train.shape}')
print(f'X_test: {X_test.shape}, Y_test: {Y_test.shape}')

model = Sequential()

model.add(Conv2D(filters = 32,
                 kernel_size = (3, 3),
                 activation = 'relu',
                 input_shape = (28, 28, 1)))
model.add(Conv2D(filters = 64,
                 kernel_size = (3, 3),
                 activation = 'relu'))
model.add(MaxPool2D(pool_size = 2))
model.add(Dropout(rate = 0.25))
model.add(Flatten())
model.add(Dense(units = 128, activation = 'relu'))
model.add(Dropout(rate = 0.5))
model.add(Dense(units = 10, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])

early_stop = EarlyStopping(monitor = 'val_loss',
                           patience = 10,
                           verbose = 0)

history = model.fit(X_train, Y_train,
                    batch_size = 200,
                    epochs = 50,
                    verbose = 1,
                    callbacks = [early_stop],
                    validation_data = (X_test, Y_test))

eval = model.evaluate(X_test, Y_test)
print(f'Test loss={round(eval[0], 4)}, acc={round(eval[1], 4)}')

train_loss = history.history['loss']
test_loss = history.history['val_loss']

train_acc = history.history['accuracy']
test_acc = history.history['val_accuracy']

x = range(len(train_loss))
plt.plot(x, train_loss, marker = '.', color = 'red', label = 'Train loss')
plt.plot(x, test_loss, marker = '.', color = 'blue', label = 'Test loss')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Loss during epochs')
plt.show()

plt.plot(x, train_acc, marker = '.', c = 'red', label = 'Train Acc.')
plt.plot(x, test_acc, marker = '.', c = 'blue', label = 'Test Acc.')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Accuracy during epochs')
plt.show()
