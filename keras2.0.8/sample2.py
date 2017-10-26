from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Merge, LSTM, Dense

# for a multi-input model with 10 classes:                                                                                                   

left_branch = Sequential()
left_branch.add(Dense(32, input_dim=784))

right_branch = Sequential()
right_branch.add(Dense(32, input_dim=784))

merged = Merge([left_branch, right_branch], mode='concat')

model = Sequential()
model.add(merged)
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# generate dummy data                                                                                                                        
import numpy as np
from keras.utils.np_utils import to_categorical
data_1 = np.random.random((1000, 784))
data_2 = np.random.random((1000, 784))

# these are integers between 0 and 9                                                                                                         
labels = np.random.randint(10, size=(1000, 1))
# we convert the labels to a binary matrix of size (1000, 10)                                                                                
# for use with categorical_crossentropy                                                                                                      
labels = to_categorical(labels, 10)

# train the model                                                                                                                            
# note that we are passing a list of Numpy arrays as training data                                                                           
# since the model has 2 inputs                                                                                                               
model.fit([data_1, data_2], labels, nb_epoch=10, batch_size=32)

