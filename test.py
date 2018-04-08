import csv
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout



csvFile = open('train.csv', 'r')
reader = csv.reader(csvFile)
dataset = []
for i in reader:
    dataset.append(i)

train_in = np.array(dataset)[1:, 1]
train_out = np.array(dataset)[1:, 2]

y_train = np.array(train_out, dtype='int')

x_train = np.zeros((2000, 14))

for k in range(0, 2000):
    seq = train_in[k]

    for i in range(0, 14):
        if seq[i] == 'A':
            x_train[k][i] = 0
        if seq[i] == 'C':
            x_train[k][i] = 1
        if seq[i] == 'G':
            x_train[k][i] = 2
        if seq[i] == 'T':
            x_train[k][i] = 3

x_train = np.array(x_train, dtype='int')



# y_train = np.zeros((2000, 1))
# for k in range(0, 2000):
#     res = train_out[k]
#     if res == '0':
#         y_train[k][0] = 1
#     else:
#         y_train[k][1] = 1


csvFile = open('test.csv', 'r')
reader = csv.reader(csvFile)
dataset = []
for i in reader:
    dataset.append(i)

test_in = np.array(dataset)[1:, 1]
x_test = np.zeros((400, 14))

for k in range(0, 400):
    seq = train_in[k]
    for i in range(0, 14):
        if seq[i] == 'A':
            x_test[k][i] = 0
        if seq[i] == 'C':
            x_test[k][i] = 1
        if seq[i] == 'G':
            x_test[k][i] = 2
        if seq[i] == 'T':
            x_test[k][i] = 3

x_test = np.array(x_test, dtype='int')

model = Sequential()
model.add(Dense(64, input_dim=14, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])



y_train = y_train.reshape((2000, 1))


model.fit(x_train, y_train,
          epochs=222,
          batch_size=32)
y_test = model.predict_classes(x_test)
print(y_test)


res = np.zeros((400, 2))

for i in range(0, 400):
    res[i][0] = i
    res[i][1] = y_test[i]

res = np.array(res, dtype='int')

with open("output2.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(res)