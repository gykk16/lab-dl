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
