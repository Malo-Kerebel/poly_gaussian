from Gaussian import Gaussian
import numpy as np
from numpy.random import random
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

import datetime

from sys import stdout

time = datetime.datetime.now

def rand_range(a, b):
    """
    returns a random value, uniformely distributed between 0 and 1
    """
    return np.random.random() * (b - a) + a


def rand_range_normal(a, b):
    """
    Returns a random value between a and b, but the middle values are prefered
    """
    tmp = -1
    # np.random.normal returns a value between -inf and +inf but we only want
    # between 0 and 1, so we discard the value outside that range and take a
    # new value until we have a value between 0 and 1.
    while tmp < 0 or tmp > 1:
        tmp = np.random.normal(0.5, 0.25)
    return tmp * (b-a) + a


def features(y, show_extremum):
    """
    returns all the features we want for the training of the algorithm
    """

    y_deriv = np.gradient(y, x[1] - x[0])

    threshold = 2*np.max(np.abs(y_deriv[:n_moving_average]))

    # plt.plot(x, y)
    # plt.plot(x, y_deriv)
    # plt.show()

    # 7 is the maximum number of extremum if we have 4 local maximum
    # and 3 local minimum
    indexs = np.zeros((7), dtype=np.int16)
    index = 0
    sign = np.sign(y_deriv[0])
    for i in range(1, n - n_moving_average - 1):

        # We determine when the derivative changes sign to detect
        # the local extremum, values oscillating arround 0 but
        if sign * y_deriv[i] < -sign * threshold:
            try:
                indexs[index] = i
            except BaseException:
                # plt.plot(x, y)
                # plt.plot(x[indexs], y[indexs], "ro")
                # plt.show()
                pass
            index += 1
            sign *= -1

    # If the gaussian are too close between the right peak of D and
    # the left peak of H, it may not detect the extremum, so we have
    # to remove those values from the indexs array
    indexes = []
    for i in range(7):
        if y[indexs[i]] > 0.015:
            indexes.append(indexs[i])

    # For small percentages of hydrogen, we may not find the extremum of
    # their spectra because they are too small compared to the deuterium.
    # in this case, we take the second derivative
    if len(indexes) < 4:
        y_second = np.gradient(y_deriv, x[1] - x[0])

        indexs_second = np.zeros((9), dtype=np.int16)
        threshold = 2*np.max(np.abs(y_second[:50]))
        index = 0
        sign = np.sign(y_second[0])
        for i in range(1, n - n_moving_average - 1):

            # We determine when the derivative changes sign to detect
            # the local extremum, values oscillating arround 0 but
            if sign * np.sign(y_second[i]) < -sign * threshold:
                indexs_second[index] = i
                index += 1
                sign *= -1

        indexes_second = []
        for i in range(7):
            if indexs_second[i] > 0:
                indexes_second.append(indexs_second[i])

        try:

            difference = indexes_second[-1] - indexes_second[-2]
            # if the second derivative change of sign, that means that we see the
            # beginning, or the end, of the last peak of the hydrogen, therefore,
            # the peak is in the middle of the last two change of sign.
            indexes.append(int(indexes_second[-2] - difference / 2))
            indexes.append(int(indexes_second[-2] + difference / 2))
        except BaseException:
            pass
    
    try:
        x_min = x[indexes[0]]       # x of the first peak
        x_2_dip = x[indexes[-2]]    # x of the firts dip
    except BaseException:
        return 0, 0, 0
    delta_x = x_2_dip - x_min   # difference between the x of the
    # first dip and the x of the first peak
    I_min_I_max = y[indexes[-1]] / y[indexes[0]] # ratio between the intensity
    # of the first peak(D), and the the last peak (H)
    I_dip_I_max = y[indexes[1]] / y[indexes[0]]

    # Plot a graph of the spectrum with the position of the detected
    # extremum to verify that it is correct
    # the threshold value is arbitrary
    if show_extremum and np.random.random() < 1:
        plt.plot(x, y)
        for i in range(len(indexes)):
            plt.plot(x[indexes[i]], y[indexes[i]], "ro")
        plt.plot([x_min, x_2_dip], [y[indexes[0]], y[indexes[0]]], "k")
        plt.plot([x_2_dip, x_2_dip], [y[indexes[-2]], y[indexes[0]]], "k--")
        plt.text((x_min + x_2_dip) / 2, y[indexes[0]] - 0.03*y[indexes[0]], r"\(\Delta \lambda\)", fontsize="xx-large")

        plt.plot([x_min, x_min], [y[indexes[0]], y[indexes[-1]]], "r")
        plt.plot([x_min, x[indexes[-1]]], [y[indexes[-1]], y[indexes[-1]]], "r--")
        plt.text(x_min - 75 * (x[1] - x[0]), (y[indexes[0]] +  y[indexes[-1]]) / 2, r"\( \frac{I_{min}}{I_{max}} \)", fontsize="xx-large")

        plt.plot([x[indexes[1]], x[indexes[1]]], [y[indexes[0]], y[indexes[1]]], "g")
        plt.plot([x_min, x[indexes[1]]], [y[indexes[0]], y[indexes[0]]], "g--")
        plt.text(x[indexes[1]] + 10 * (x[1] - x[0]), (y[indexes[0]] +  y[indexes[1]]) / 2, r"\( \frac{I_{dip}}{I_{max}} \)", fontsize="xx-large")

        plt.ylabel(r"Normalized intensity", fontsize="x-large")
        plt.xlabel(r"Wave length (in \AA)", fontsize="x-large")

        plt.show()    

    return delta_x, I_min_I_max, I_dip_I_max


def percent_error(predict, true):
    """
    Returns the percentage of error between the true value and the predicted value
    """
    return (np.abs(true - predict) / true) * 100


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


#Physics quantities
min_percent = 0.01
max_percent = 0.25
min_B = 1
max_B = 6
percent_temp1 = 0.55
Temp1 = 23208
Temp2 = 174060

# Tore Supra data
data = np.loadtxt("TS28270.txt")
x = data[:, 0]
y_tore = data[:, 1]

#Numerical quantities
n = len(x)
N_train = 250
N_test = 2000
n_features = 3

n_moving_average = 6

# Array for data and target
data_train = np.zeros((N_train, n_features))
target_train = np.zeros((N_train, 5))
data_test = np.zeros((N_test, n_features))
target_test = np.zeros((N_test))

data_tore = np.zeros((1, n_features))

n_epochs = 32

noise_train = True
noise_test = True

# QoL features
show_verif_extremum = False

x_tore = np.linspace(6558, 6565, n)
x_noise = x_tore
x = x_tore[n_moving_average - 1:]

data_tore[0, :] = features(moving_average(y_tore, n_moving_average), False)
print(data_tore)

# Generating training values
for i in range(N_train):

    stdout.write("\r%3d/%4d" % (i+1, N_train))
    stdout.flush()

    while np.sum(data_train[i, :]) == 0:
    # Randomizing values for training
        percent_H = rand_range(min_percent, max_percent)
        B = rand_range_normal(min_B, max_B)
        percent_temp1 = rand_range_normal(0.1, 0.9)
        # Generating temperatures between T +/- 10%
        T1 = rand_range_normal(Temp1 - Temp1 * 0.2, Temp1 + Temp1 * 0.2)
        T2 = rand_range_normal(Temp2 - Temp2 * 0.2, Temp2 + Temp2 * 0.2)

        # We discard in _ the expected value returned for the approach
        y_noise, _ = Gaussian(1-percent_H, B, percent_temp1, T1, T2, n, noise=noise_train)

        if noise_train:
            y = moving_average(y_noise, n_moving_average)


        data_train[i, :] = features(y, show_verif_extremum)


        target_train[i, 0] = percent_H
        target_train[i, 1] = T1
        target_train[i, 2] = T2
        target_train[i, 3] = percent_temp1
        target_train[i, 4] = B

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers


def build_and_compile_model(norm, n):
    """
    Build and compile a model for regression, norm being a normalizer
    and n being the number of output that we want
    """
    model = keras.Sequential([
        norm,
        layers.Dense(150, activation='relu'),
        layers.Dense(200, activation='relu'),
        layers.Dense(400, activation='relu'),
        layers.Dense(400, activation='relu'),
        layers.Dense(200, activation='relu'),
        layers.Dense(100, activation='relu'),
        layers.Dense(n)
    ])

    model.compile(loss='mean_squared_error',
                  optimizer=tf.keras.optimizers.Adam(0.001))
    return model


normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(data_train))

model_NN = build_and_compile_model(normalizer, 1)

# Training of the NN model
t1 = time()
model_NN.fit(data_train, target_train[:, 0], epochs=n_epochs)#, validation_split=0.2)
t2 = time()
print(f"fit NN, in {t2-t1}")

# Generating test value

prediction_H = model_NN.predict(data_tore.reshape(1, -1))[0, 0]
print(f"Pourcentage d'hydrogène prédit {prediction_H*100}%")
# prediction_H = 0.07553751766681671

# Training of the NN model
model_NN = build_and_compile_model(normalizer, 1)
t1 = time()
model_NN.fit(data_train, target_train[:, 1], epochs=n_epochs)#, validation_split=0.2)
t2 = time()
print(f"fit NN, in {t2-t1}")

# Generating test value

prediction_T1 = model_NN.predict(data_tore.reshape(1, -1))[0, 0]
print(f"Température 1 prédite {prediction_T1}K")
# prediction_T1 = 22929.09765625

# Training of the NN model
model_NN = build_and_compile_model(normalizer, 1)
t1 = time()
model_NN.fit(data_train, target_train[:, 2], epochs=n_epochs)#, validation_split=0.2)
t2 = time()
print(f"fit NN, in {t2-t1}")

# Generating test value

prediction_T2 = model_NN.predict(data_tore.reshape(1, -1))[0, 0]
print(f"Température 2 prédite {prediction_T2}K")
# prediction_T2 = 174572.90625

# Training of the NN model
model_NN = build_and_compile_model(normalizer, 1)
t1 = time()
model_NN.fit(data_train, target_train[:, 3], epochs=n_epochs)#, validation_split=0.2)
t2 = time()
print(f"fit NN, in {t2-t1}")

# Generating test value

prediction_percent = model_NN.predict(data_tore.reshape(1, -1))[0, 0]
print(f"Pourcentage à la température 1 prédit {prediction_percent*100}%")
# prediction_percent = 0.8020308017730713

# Training of the NN model
model_NN = build_and_compile_model(normalizer, 1)
t1 = time()
model_NN.fit(data_train, target_train[:, 4], epochs=n_epochs)#, validation_split=0.2)
t2 = time()
print(f"fit NN, in {t2-t1}")

# Generating test value

prediction_B = model_NN.predict(data_tore.reshape(1, -1))[0, 0]
# prediction_B = 1.9945340156555176

print(f"Pourcentage d'hydrogène prédit {prediction_H*100}%")
print(f"Température 1 d'hydrogène prédite {prediction_T1}K")
print(f"Température 2 prédite {prediction_T2}K")
print(f"Pourcentage à la température 1 prédit {prediction_percent*100}%")
print(f"Champs magnétique B prédit {prediction_B}T")

plt.plot(x_tore, y_tore, label="Expérimental")
y_pred, _ = Gaussian(1-prediction_H, prediction_B, prediction_percent, prediction_T1, prediction_T2, n, noise=False)
plt.plot(x_tore, y_pred, label="Prédit")
plt.legend()
plt.show()
