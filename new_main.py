from Gaussian import Gaussian
import numpy as np
from numpy.random import random
import matplotlib.pyplot as plt

import datetime

from sys import stdout

time = datetime.datetime.now

def rand_range(a, b):
    return np.random.random() * (b - a) + a


def rand_range_normal(a, b):
    tmp = -1
    while tmp < 0 or tmp > 1:
        tmp = np.random.normal(0.5, 0.25)
    return tmp * (b-a) + a


def features(y):

    y_deriv = np.gradient(y, x[1] - x[0])

    indexs = np.zeros((7), dtype=np.int16)
    index = 0
    sign = np.sign(y_deriv[10])
    for i in range(11, n):
    
        if np.sign(y_deriv[i]) != sign:
            indexs[index] = i
            index += 1
            sign *= -1

    indexes = []
    for i in range(7):
        if indexs[i] > 0:
            indexes.append(indexs[i])

    if np.random.random() < 0.0001:
        for i in range(len(indexes)):
            plt.plot(x[indexes[i]], y[indexes[i]], "ro")
        plt.plot(x, y)
        plt.show()

    x_min = x[indexes[0]]
    x_2_creu = x[indexes[-2]]
    delta_x = x_2_creu - x_min
    I_min_I_max = y[indexes[-1]] / y[indexes[0]]
    I_cr_I_max = y[indexes[1]] / y[indexes[0]] 
    
    return B, T1, T2, delta_x, I_min_I_max, I_cr_I_max


def percent_error(predict, true):
    return (np.abs(true - predict) / true) * 100


#Physics quantities
percent_H = 0.25
B = 2
percent_temp1 = 0.55
Temp1 = 23208
Temp2 = 174060

#Numerical quantities
n = 1000
N_train = 100000
N_test = 10
data_train = np.zeros((N_train, 6))  # 6 = number of features
target_train = np.zeros((N_train))
data_test = np.zeros((N_test, 6))  # 6 = number of features
target_test = np.zeros((N_test))

n_epochs = 10

x = np.linspace(6558, 6565, n)

for i in range(N_train):

    stdout.write("\r%3d/%4d" % (i+1, N_train))
    stdout.flush()

    percent_H = rand_range_normal(0.1, 0.45)
    B = rand_range_normal(1.5, 4)
    percent_temp1 = rand_range_normal(0.4, 0.7)
    T1 = rand_range_normal(Temp1 - Temp1 * 0.1, Temp1 + Temp1 * 0.1)
    T2 = rand_range_normal(Temp2 - Temp2 * 0.1, Temp2 + Temp2 * 0.1)

    y, _ = Gaussian(1-percent_H, B, percent_temp1, T1, T2, n)

    data_train[i, :] = features(y)
    target_train[i] = percent_H

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers


def build_and_compile_model(norm, n, n_points):
    model = keras.Sequential([
        # norm,
        layers.Dense(n_points, activation='relu'),
        layers.Dense(1.5*n_points, activation='relu'),
        layers.Dense(3*n_points, activation='relu'),
        layers.Dense(3*n_points, activation='relu'),
        layers.Dense(1.5*n_points, activation='relu'),
        # layers.Dense(128, activation='relu'),
        layers.Dense(0.5*n_points, activation='relu'),
        layers.Dense(n)
    ])

    model.compile(loss='mean_squared_error',
                  optimizer=tf.keras.optimizers.Adam(0.001))
    return model


normalizer = tf.keras.layers.Normalization(axis=-1)
# normalizer.adapt(np.array(data))

# model = build_and_compile_model(normalizer, 3)
model_NN = build_and_compile_model(normalizer, 1, 150)

t1 = time()

model_NN.fit(data_train, target_train, epochs=n_epochs)#, validation_split=0.2)

t2 = time()
print("fini fit NN, in", t2-t1)

for i in range(N_test):

    percent_H = rand_range_normal(0.1, 0.45)
    B = rand_range_normal(1.5, 4)
    percent_temp1 = rand_range_normal(0.4, 0.7)
    T1 = rand_range_normal(Temp1 - Temp1 * 0.1, Temp1 + Temp1 * 0.1)
    T2 = rand_range_normal(Temp2 - Temp2 * 0.1, Temp2 + Temp2 * 0.1)

    y, _ = Gaussian(1-percent_H, B, percent_temp1, T1, T2, n)

    data_test[i, :] = features(y)
    target_test[i] = percent_H

error = np.zeros((N_test))
for i in range(N_test):
    prediction = model_NN.predict(data_test[i].reshape(1, -1))[0, 0]
    
    error[i] = percent_error(prediction, target_test[i])
    print("l'algorithme a prédit", prediction*100, "% d'Hydrogène, il y en a en réalité,", target_test[i]*100, "%, ça fait", error[i], "% d'erreur.")

print("L'erreur moyenne est de", np.mean(error), "%")
