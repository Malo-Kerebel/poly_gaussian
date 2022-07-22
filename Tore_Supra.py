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


def extremum(y):
    """
    Détermine les extremum de la courbe en entrée
    """

    y_deriv = np.gradient(y, x[1] - x[0])

    # The threshold is there so that the fluctuation of the noise have less impact
    threshold = np.max(np.abs(y_deriv[:n_moving_average]))

    indexs = np.zeros((20), dtype=np.int16)
    index = 0
    sign = np.sign(y_deriv[0])
    first = 0
    second = 0
    for i in range(n_moving_average + 3, n - n_moving_average - 1):

        # We determine if the value when the absolute value of the derivative
        # goes below the threshold and when it goes above the threshold
        # We approximate that the 0 of the derivative is thus the average of the
        # two point where it crossed the derivative
        # We also make sure that the two value are separated so that the fluctuation
        # of the noise have less impact
        if i > second + 15 and (first == 0 and (np.abs(y_deriv[i]) < threshold) or
           (first != 0 and (np.abs(y_deriv[i]) > threshold))):
            if first == 0:
                first = i
                sign *= -1
            elif i > first + 10:
                try:
                    second = i
                    indexs[index] = int((first+second)/2)
                    index += 1
                    first = 0
                    sign *= -1
                except BaseException:
                    # plt.plot(x, y)
                    # plt.plot(x[indexs], y[indexs], "ro")
                    # plt.show()
                    pass

    # plot_extremum(indexs, y, y_deriv, threshold)
    
    return indexs, threshold


def plot_extremum(indexs, y, y_deriv=None, threshold=None):
    """
    plot un graph montrant la position des extremums
    """

    if y_deriv is not None:
        plt.plot(x, y_deriv)
    plt.plot(x, y)
    plt.plot(x[indexs], y[indexs], "r.")
    if threshold is not None:
        plt.plot(x, [threshold for i in range(len(x))], "g--")
        if threshold != 0:
            plt.plot(x, [-threshold for i in range(len(x))], "g--")
    plt.show()


def features(y, show_extremum):
    """
    returns all the features we want for the training of the algorithm
    """

    indexs, threshold = extremum(y)
    indexes = []   

    # If the gaussian are too close between the right peak of D and
    # the left peak of H, it may not detect the extremum, so we have
    # to remove those values from the indexs array
    indexes = []
    for i in range(7):
        if y[indexs[i]] > threshold:
            indexes.append(indexs[i])

    # indexes = indexes[1:]       # We remove the first extremum as it is a bug

    # For small percentages of hydrogen, we may not find the extremum of
    # their spectra because they are too small compared to the deuterium.
    # in this case, we take the second derivative
    if len(indexes) < 5:
        indexs_second, threshold = extremum(np.gradient(y, x[1]-x[0]))
        
        indexes_second = []
        for i in range(len(indexs_second)):
            if y[indexs_second[i]] > threshold/2:
                indexes_second.append(int((indexs_second[i+1] + indexs_second[i+1]) / 2))

        try:

            difference = indexes_second[-1] - indexes_second[-2]
            # if the second derivative change of sign, that means that we see the
            # beginning, or the end, of the last peak of the hydrogen, therefore,
            # the peak is in the middle of the last two change of sign.
            indexes.append(int(indexes_second[-2] - difference / 2))
            indexes.append(int(indexes_second[-2] + difference / 2))
        except BaseException:
            pass
    
    if len(indexes) > 4:
        x_min = x[indexes[0]]       # x of the first peak
        x_2_dip = x[indexes[-2]]    # x of the firts dip
    else:
        # The algorithm might be unable to process the spectrum
        # in this case we discard the spectrum
        return 0

    if len(indexes) < 5:
        return 0, 0, 0, 0, 0, 0
    delta_x = x_2_dip - x_min   # difference between the x of the
    # first dip and the x of the first peak
    I_min_I_max = y[indexes[-1]] / y[indexes[0]] # ratio between the intensity
    # of the first peak(D), and the the last peak (H)
    I_dip_I_max = y[indexes[1]] / y[indexes[0]]

    # Plot a graph of the spectrum with the position of the detected
    # extremum to verify that it is correct
    # the threshold value is arbitrary
    if show_extremum and (np.random.random() < 0.0001 or tore):
        plt.plot(x, y)
        for i in range(len(indexes)):
            plt.plot(x[indexes[i]], y[indexes[i]], "ro")
        plt.plot([x_min, x_2_dip], [y[indexes[0]], y[indexes[0]]], "k")
        plt.plot([x_2_dip, x_2_dip], [y[indexes[-2]], y[indexes[0]]], "k--")
        plt.text((x_min + x_2_dip) / 2, y[indexes[0]] - 0.03*y[indexes[0]], r"\(\Delta \lambda\)", fontsize="xx-large")

        plt.plot([x_min, x_min], [y[indexes[0]], y[indexes[-1]]], "r")
        plt.plot([x_min, x[indexes[-1]]], [y[indexes[-1]], y[indexes[-1]]], "r--")
        plt.text(x_min - 5 * (x[1] - x[0]), (y[indexes[0]] +  y[indexes[-1]]) / 2, r"\( \frac{I_{min}}{I_{max}} \)", fontsize="xx-large")

        plt.plot([x[indexes[1]], x[indexes[1]]], [y[indexes[0]], y[indexes[1]]], "g")
        plt.plot([x_min, x[indexes[1]]], [y[indexes[0]], y[indexes[0]]], "g--")
        plt.text(x[indexes[1]] + (x[1] - x[0]), (y[indexes[0]] +  y[indexes[1]]) / 2, r"\( \frac{I_{dip}}{I_{max}} \)", fontsize="xx-large")

        plt.ylabel(r"Normalized intensity", fontsize="x-large")
        plt.xlabel(r"Wave length (in \AA)", fontsize="x-large")

        plt.show()    

    if not tore:
        return delta_x, I_min_I_max, I_dip_I_max, B, T1, T2
    else:
        return delta_x, I_min_I_max, I_dip_I_max, 0, 0, 0

def percent_error(predict, true):
    """
    Returns the percentage of error between the true value and the predicted value
    """
    return (np.abs(true - predict) / true) * 100


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


#Physics quantities
min_percent = 0.02
max_percent = 0.25
min_B = 1
max_B = 5
Temp1 = 2.0
Temp2 = 20

# Tore Supra data
data = np.loadtxt("TS28270.txt")
x = data[:, 0]
y_tore = data[:, 1]

#Numerical quantities
n = len(x)
N_train = 100000
N_test = 2000
n_features = 6

n_moving_average = 6

# Array for data and target
data_train = np.zeros((N_train, n_features))
target_train = np.zeros((N_train, 5))

data_tore = np.zeros((1, n_features))

n_epochs = 32

noise_train = True
noise_test = True

# QoL features
show_verif_extremum = False

x_tore = np.linspace(6558, 6565, n)
x_noise = x_tore
x = x_tore[n_moving_average - 1:]

tore = True
data_tore[0, :] = [2.08333333, 0.067299, 0.66948312, 0., 0., 0.]
# features(moving_average(y_tore, n_moving_average), False) # formerly calculated with this
print(data_tore)
tore = False

n = 1000
n_moving_average = n//25
x = np.linspace(6558, 6565, n - n_moving_average + 1)

# Pourcentage d'hydrogène prédit 7.0618972182273865%
# Température 1 d'hydrogène prédite 2.184262275695801eV
# Température 2 prédite 16.41613006591797eV
# Pourcentage à la température 1 prédit 60.52653193473816%
# Champs magnétique B prédit 1.8975647687911987T

# Generating training values
for i in range(N_train):

    stdout.write("\r%3d/%4d" % (i+1, N_train))
    stdout.flush()

    while np.sum(data_train[i, :]) == 0:
        # Randomizing values for training
        percent_H = rand_range(min_percent, max_percent)
        B = rand_range_normal(min_B, max_B)
        percent_temp1 = rand_range_normal(0.2, 0.8)
        # Generating temperatures between T +/- 40%
        T1 = rand_range_normal(Temp1 - Temp1 * 0.4, Temp1 + Temp1 * 0.4)
        T2 = rand_range_normal(Temp2 - Temp2 * 0.4, Temp2 + Temp2 * 0.4)

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
normalizer.adapt(np.array(data_train[:, :3]))

# Training of the NN model
model_NN = build_and_compile_model(normalizer, 1)
t1 = time()
model_NN.fit(data_train[:, :3], target_train[:, 4], epochs=n_epochs)#, validation_split=0.2)
t2 = time()
print(f"fit NN, in {t2-t1}")

# Making prediction
prediction_B = model_NN.predict(data_tore[:, :3].reshape(1, -1))[0, 0]
print(f"Champs magnétique B prédit {prediction_B}T")
data_tore[0, 3] = prediction_B
# prediction_B = 1.9945340156555176

normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(data_train[:, :4]))

model_NN = build_and_compile_model(normalizer, 1)

# Training of the NN model
t1 = time()
model_NN.fit(data_train[:, :4], target_train[:, 0], epochs=n_epochs)#, validation_split=0.2)
t2 = time()
print(f"fit NN, in {t2-t1}")

# Making prediction
prediction_H = model_NN.predict(data_tore[:, :4].reshape(1, -1))[0, 0]
print(f"Pourcentage d'hydrogène prédit {prediction_H*100}%")
# prediction_H = 0.07553751766681671

# Training of the NN model
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(data_train[:, :4]))
model_NN = build_and_compile_model(normalizer, 1)

t1 = time()
model_NN.fit(data_train[:, :4], target_train[:, 1], epochs=n_epochs)#, validation_split=0.2)
t2 = time()
print(f"fit NN, in {t2-t1}")

# Making prediction
prediction_T1 = model_NN.predict(data_tore[:, :4].reshape(1, -1))[0, 0]
print(f"Température 1 prédite {prediction_T1}eV")
data_tore[0, 4] = prediction_T1
# prediction_T1 = 22929.09765625

# Training of the NN model
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(data_train[:, :5]))
model_NN = build_and_compile_model(normalizer, 1)

t1 = time()
model_NN.fit(data_train[:, :5], target_train[:, 2], epochs=n_epochs)#, validation_split=0.2)
t2 = time()
print(f"fit NN, in {t2-t1}")

# Making prediction
prediction_T2 = model_NN.predict(data_tore[:, :5].reshape(1, -1))[0, 0]
print(f"Température 2 prédite {prediction_T2}eV")
data_tore[0, 5] = prediction_T2
# prediction_T2 = 174572.90625

# Training of the NN model
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(data_train))
model_NN = build_and_compile_model(normalizer, 1)
t1 = time()
model_NN.fit(data_train, target_train[:, 3], epochs=n_epochs)#, validation_split=0.2)
t2 = time()
print(f"fit NN, in {t2-t1}")

# Making prediction
prediction_percent = model_NN.predict(data_tore.reshape(1, -1))[0, 0]
print(f"Pourcentage à la température 1 prédit {prediction_percent*100}%")
# prediction_percent = 0.8020308017730713

print(f"Pourcentage d'hydrogène prédit {prediction_H*100}%")
print(f"Température 1 d'hydrogène prédite {prediction_T1}eV")
print(f"Température 2 prédite {prediction_T2}eV")
print(f"Pourcentage à la température 1 prédit {prediction_percent*100}%")
print(f"Champs magnétique B prédit {prediction_B}T")

plt.plot(x_tore, y_tore, label="Expérimental")
y_pred, _ = Gaussian(1-prediction_H, prediction_B, prediction_percent, prediction_T1, prediction_T2, n-n_moving_average+1, noise=False)
plt.plot(x, y_pred, label="Prédit")
plt.legend()
plt.show()
