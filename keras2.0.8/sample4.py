from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM

model = Sequential()
model.add(Embedding(100, 256, input_length=256))
model.add(LSTM(output_dim=128, activation='sigmoid', inner_activation='hard_sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# generate dummy data                                                                                                                        
import numpy as np
X_train = np.random.random((10, 256))
Y_train = np.random.randint(2, size=(10, 1))

print(X_train.shape)
print(Y_train.shape)

model.fit(X_train, Y_train, batch_size=16, nb_epoch=1)