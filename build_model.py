import warnings 
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=FutureWarning)
    import tensorflow as tf
    import pandas as pd
    from matplotlib import pyplot as plt
print('Warnings suppressed, review code to adjust')

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
import time

NAME = f'Cats-vs-dog-cnn-64x2-nodense {int(time.time())}'
TensorBoard = TensorBoard(log_dir=f'logs/{NAME}')

X = pickle.load(open('X.pickle','rb'))
Y = pickle.load(open('Y.pickle','rb'))

X = X/255.0

model = Sequential()
model.add(Conv2D(64, (3,3), input_shape = X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size =(2,2)))

model.add(Conv2D(64, (3,3), input_shape = X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size =(2,2)))

model.add(Flatten())

# model.add(Dense(64))
# model.add(Activation('relu'))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X, Y, batch_size=32, epochs=5, validation_split=0.3, callbacks=[TensorBoard])


"""Train on 22451 samples, validate on 2495 samples
2019-09-30 17:30:14.895016: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
Epoch 1/3
22451/22451 [==============================] - 39s 2ms/sample - loss: 0.6288 - acc: 0.6486 - val_loss: 0.5367 - val_acc: 0.7214
Epoch 2/3
22451/22451 [==============================] - 43s 2ms/sample - loss: 0.5320 - acc: 0.7366 - val_loss: 0.5133 - val_acc: 0.7455
Epoch 3/3
22451/22451 [==============================] - 44s 2ms/sample - loss: 0.4843 - acc: 0.7671 - val_loss: 0.5489 - val_acc: 0.7238

tensorboard --logdir='logs/'
"""