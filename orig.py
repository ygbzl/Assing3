import csv

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout

# Generate dummy data
x_train = np.random.randint(4, size=(1000, 20))
y_train = np.random.randint(2, size=(1000, 1))
x_test = np.random.randint(4, size=(100, 20))
y_test = np.random.randint(2, size=(100, 1))
#
# model = Sequential()
# model.add(Dense(64, input_dim=20, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(1, activation='sigmoid'))
#
# model.compile(loss='binary_crossentropy',
#               optimizer='rmsprop',
#               metrics=['accuracy'])
#
# model.fit(x_train, y_train,
#           epochs=20,
#           batch_size=128)
# score = model.evaluate(x_test, y_test, batch_size=128)
# y = model.predict_classes(x_test)
# print(y)

res = np.zeros((100, 2))
for i in range(0, 100):
    res[i][0] = i
    res[i][1] = y_test[i]

res = np.array(res, dtype='int')


with open("output.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(res)