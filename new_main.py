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

    threshold = 2*np.max(np.abs(y_deriv[:50]))

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
        return 0, 0, 0, 0, 0, 0
    delta_x = x_2_dip - x_min   # difference between the x of the
    # first dip and the x of the first peak
    I_min_I_max = y[indexes[-1]] / y[indexes[0]] # ratio between the intensity
    # of the first peak(D), and the the last peak (H)
    I_dip_I_max = y[indexes[1]] / y[indexes[0]]

    # Plot a graph of the spectrum with the position of the detected
    # extremum to verify that it is correct
    # the threshold value is arbitrary
    if show_extremum and np.random.random() < 0.01:
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

    return B, T1, T2, delta_x, I_min_I_max, I_dip_I_max


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
max_B = 4
percent_temp1 = 0.55
Temp1 = 23208
Temp2 = 174060

#Numerical quantities
n = 1000
N_train = 150000
N_test = 2000
n_features = 6

n_moving_average = 75

# Array for data and target
data_train = np.zeros((N_train, n_features))
target_train = np.zeros((N_train))
data_test = np.zeros((N_test, n_features))
target_test = np.zeros((N_test))

n_epochs = 32

noise_train = True
noise_test = True

# QoL features
show_verif_extremum = False

x = np.linspace(6558, 6565, n)
x_noise = x
x = x[n_moving_average - 1:]

# Generating training values
for i in range(N_train):

    stdout.write("\r%3d/%4d" % (i+1, N_train))
    stdout.flush()

    while np.sum(data_train[i, :]) == 0:
    # Randomizing values for training
        percent_H = rand_range(min_percent, max_percent)
        B = rand_range_normal(min_B, max_B)
        percent_temp1 = rand_range_normal(0.25, 0.75)
        # Generating temperatures between T +/- 10%
        T1 = rand_range_normal(Temp1 - Temp1 * 0.1, Temp1 + Temp1 * 0.1)
        T2 = rand_range_normal(Temp2 - Temp2 * 0.1, Temp2 + Temp2 * 0.1)

        # We discard in _ the expected value returned for the approach
        y_noise, _ = Gaussian(1-percent_H, B, percent_temp1, T1, T2, n, noise=noise_train)

        if noise_train:
            y = moving_average(y_noise, n_moving_average)


        data_train[i, :] = features(y, show_verif_extremum)

    target_train[i] = percent_H

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
model_NN.fit(data_train, target_train, epochs=n_epochs)#, validation_split=0.2)
t2 = time()
print(f"fit NN, in {t2-t1}")

# Generating test value
for i in range(N_test):

    stdout.write("\r%3d/%4d" % (i+1, N_test))
    stdout.flush()

    # Same as for train values we generate test value randomly
    percent_H = rand_range(min_percent, max_percent)
    B = rand_range_normal(min_B, max_B)
    percent_temp1 = rand_range_normal(0.25, 0.75)
    T1 = rand_range_normal(Temp1 - Temp1 * 0.1, Temp1 + Temp1 * 0.1)
    T2 = rand_range_normal(Temp2 - Temp2 * 0.1, Temp2 + Temp2 * 0.1)

    y, _ = Gaussian(1-percent_H, B, percent_temp1, T1, T2, n, noise=noise_test)

    if noise_test:
        y = moving_average(y, n_moving_average)

    data_test[i, :] = features(y, False)
    target_test[i] = percent_H

error = np.zeros((N_test))
predict = np.zeros((N_test))

for i in range(N_test):

    stdout.write("\r%3d/%4d" % (i+1, N_test))
    stdout.flush()

    # the predict method will output a 1×1 matrix, but we want only the number
    # so we extract it with the [0, 0]
    prediction = model_NN.predict(data_test[i].reshape(1, -1))[0, 0]

    error[i] = percent_error(prediction, target_test[i])
    # print(f"The algorithm predicted {prediction*100}% of Hydrogen, in reality there is , {target_test[i]*100}%, we have {error[i]}% of error.")

    predict[i] = prediction

print(f"The average error is of {np.mean(error)}%")
print(f"The median error is of {np.median(error)}%")

plt.plot(target_test*100, predict*100, "bo")
plt.plot(target_test*100, target_test*100, "k", label="theorical")
plt.xlabel("True percentage")
plt.ylabel("Predicted percentage")
plt.show()

plt.plot(x_noise, y_noise)
plt.show()
