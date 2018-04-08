import csv

from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np

data_dim = 14
timesteps = 4
num_classes = 2

# expected input data shape: (batch_size, timesteps, data_dim)
model = Sequential()
model.add(LSTM(32, return_sequences=True,
               input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
model.add(LSTM(32))  # return a single vector of dimension 32
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# Generate dummy training data
# x_train = np.random.random((1000, timesteps, data_dim))
# y_train = np.random.random((1000, num_classes))



csvFile = open('train.csv', 'r')
reader = csv.reader(csvFile)
dataset = []
for i in reader:
    dataset.append(i)

train_in = np.array(dataset)[1:, 1]
train_out = np.array(dataset)[1:, 2]

x_train = np.zeros((2000, 4, 14))

for k in range(0, 2000):
    seq = train_in[k]

    for i in range(0, 14):
        if seq[i] == 'A':
            x_train[k][0][i] = 1
        if seq[i] == 'C':
            x_train[k][1][i] = 1
        if seq[i] == 'G':
            x_train[k][2][i] = 1
        if seq[i] == 'T':
            x_train[k][3][i] = 1


y_train = np.zeros((2000, 2))
for k in range(0, 2000):
    res = train_out[k]
    if res == '0':
        y_train[k][0] = 1
    else:
        y_train[k][1] = 1





csvFile = open('test.csv', 'r')
reader = csv.reader(csvFile)
dataset = []
for i in reader:
    dataset.append(i)

test_in = np.array(dataset)[1:, 1]

x_test = np.zeros((400, 4, 14))

for k in range(0, 400):
    seq = train_in[k]

    for i in range(0, 14):
        if seq[i] == 'A':
            x_test[k][0][i] = 1
        if seq[i] == 'C':
            x_test[k][1][i] = 1
        if seq[i] == 'G':
            x_test[k][2][i] = 1
        if seq[i] == 'T':
            x_test[k][3][i] = 1

print(x_train)

# Generate dummy validation data
# x_val = np.random.random((100, timesteps, data_dim))
# y_val = np.random.random((100, num_classes))

model.fit(x_train, y_train,
          batch_size=64, epochs=100)
y_test = model.predict_classes(x_test)
#
print(y_test)

res = np.zeros((400, 2))
for i in range(0, 400):
    res[i][0] = i
    res[i][1] = y_test[i]

res = np.array(res, dtype='int')

with open("output3.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(res)

#
# print(train_out)
#
# print(y_train)
