#importing libraries

import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

#loading txt file
filename = "alice.txt"
raw_text = open(filename).read()

#converting to lowercase
raw_text = raw_text.lower()

#finding unique characters used in txt file in sorted order in the form of list
chars = sorted(list(set(raw_text)))

#mapping characters to int
char_to_int = dict((c,i) for i, c in enumerate(chars))

#summarizing dataset

n_chars = len(raw_text)
n_vocab = len(chars)

print "Total Characters: ", n_chars
print "Total Vocab: ", n_vocab

#preapare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
	seq_in = raw_text[i:i + seq_length]
	seq_out = raw_text[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])

n_patterns = len(dataX)
print "Total Patterns: ", n_patterns

#reshape X to be [samples, timesteps, features] required by LSTM type of RNN

X = numpy.reshape(dataX, (n_patterns, seq_length, 1))

X = X / float(n_vocab)

y = np_utils.to_categorical(dataY)


# define the LSTM model

model = Sequential()
# print model
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
# load the network weights
model.compile(loss='categorical_crossentropy', optimizer='adam')

#storing checkpoints after each epoch of model
filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

#training the model
model.fit(X, y, epochs=20, batch_size=128, callbacks=callbacks_list)