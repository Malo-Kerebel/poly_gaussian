#!/usr/bin/python

from double_gaussienne import poly_gauss
from Gaussian import Gaussian
import numpy as np
from numpy.random import random
import matplotlib.pyplot as plt

import datetime

from sys import stdout

time = datetime.datetime.now


def change_x(x):
    """
    Change les ordonnées entre x_min et x_max à entre lambda_min et lambda_max
    """
    return (x - x_min) / (x_max - x_min) * (lambda_max - lambda_min) + lambda_min


def rand_range(a, b):
    """
    retourne un nombre aléatoire dans l'intervalle [a, b[
    """
    return random() * (b - a) + a


def gaussienne(x, mu, sigma, coeff):
    """
    retourne les ordonnées d'une gaussienne
    """
    return coeff * np.exp(-(x-mu)*(x-mu)/sigma)


def merge_array(mu, sigma, coeff):
    """
    Retourne la concaténation des 3 arrays en entrée
    """
    ret = np.zeros(len(mu) + len(sigma) + len(coeff))

    for i in range(len(mu)):
        ret[i] = mu[i]
    for i in range(len(sigma)):
        ret[i+len(mu)] = sigma[i]
    for i in range(len(coeff)):
        ret[i+len(mu)+len(sigma)] = coeff[i]
    return ret


sk_learn = False  # Est-ce que du ML doit être utilisé
train_random = False  # Est-ce que les valeurs de test doivent être aléatoire ou
# utilisé les contraintes supplémentaires
test_random = False   # Est-ce que la valeur de test doit être aléatoire ou
# venir de données synthétique

N = 400000  # Nombre de valeurs d'entrainement
n = 1000    # Nombre de points dans les courbes d'entrainements
n_gauss = 4  # nombre de gaussienne à additionner

n_epochs = 15  # Nombre d'épochs sur lesquel le NN doit itérer

x_min = -10  # Valeurs minimales et maximales du x des données
x_max = 10

lambda_min = 6558  # Valeurs minimales des longueurs d'ondes étudier
lambda_max = 6565

percent_D = 0.75  # Pourcentage de deutérium dans la valeur de test (uniquement si test_random == False)

x = np.linspace(x_min, x_max, n)

train_share = 1  # Portion des valeurs créer qui doit être dédié à l'entrainement
N_train = int(train_share * N)
N_test = N - N_train

data = np.zeros((N_train, n))  # Initialisation des matrices de stockage des valeurs d'entrainement
target = np.zeros((N_train, n_gauss, 3))
if train_random:
    target_full = np.zeros((N_train, 3*n_gauss))
else:
    target_full = np.zeros((N_train, n_gauss + 4))

data_test = np.zeros((N_test, n))
target_test = np.zeros((N_test, n_gauss, 3))

if train_random:
    target_test_full = np.zeros((N_test, 3*n_gauss))
else:
    target_test_full = np.zeros((N_test, n_gauss + 4))  # On peut supprimer des informations supplémentaires
    # grâce aux contraintes supplémentaires
stdout.write("generating train values ")

for i in range(N):

    stdout.write("\r%5d/%6d" % (i+1, N))
    stdout.flush()
    # print(f"generating train values {i+1}/{N}")

    mu = []
    sigma = []
    coeff = []

    if train_random:
        for j in range(n_gauss):
            # Génération des mu, en prenant attention de ne pas placer les courbes trop près des valeurs limites
            mu.append(rand_range(x_min + (x_max - x_min) / 5, x_max - (x_max - x_min) / 5))
            sigma.append(rand_range(0.1, 2.5))
            coeff.append(rand_range(0.01, 1))
        # mu = sorted(mu)
        # coeff = sorted(coeff)
        # tmp = coeff[1]
        # coeff[1] = coeff[2]
        # coeff[2] = tmp

    else:

        small_gap = rand_range(0.75, 4)
        big_gap = rand_range(4, 8)

        first_mu = rand_range(x_min + (x_max - x_min) / 5, x_max - (x_max - x_min) / 5)
        mu.append(first_mu - small_gap)
        mu.append(first_mu + small_gap)
        mu.append(first_mu + big_gap - small_gap)
        mu.append(first_mu + big_gap + small_gap)

        sigma_first = rand_range(0.05, 4)

        sigma.append(sigma_first)
        sigma.append(sigma_first)

        sigma_second = rand_range(0.05, 4)

        sigma.append(sigma_second)
        sigma.append(sigma_second)

        coeff_first = rand_range(0.001, 1)

        coeff.append(coeff_first)
        coeff.append(coeff_first)

        coeff_second = rand_range(0.001, 1)

        coeff.append(coeff_second)
        coeff.append(coeff_second)

    courbe = poly_gauss(x, mu, sigma, coeff)

    # if i%50000 == 0:
    #     courbe.plot()

    if i < N_train:
        data[i, :] = courbe.y

        target[i, :, 0] = mu
        target[i, :, 1] = sigma
        target[i, :, 2] = coeff
        if train_random:
            target_full[i, :] = merge_array(mu, sigma, coeff)
        else:
            target_full[i, :] = merge_array(mu, [sigma[0], sigma[-1]], [coeff[0], coeff[-1]])
    else:
        data_test[i-N_train, :] = courbe.y

        target_test[i-N_train, :] = mu
        target_test[i-N_train, :, 1] = sigma
        target_test[i-N_train, :, 2] = coeff
        if train_random:
            target_test_full[i-N_train, :] = merge_array(mu, sigma, coeff)
        else:
            target_full[i, :] = merge_array(mu, [sigma[0], sigma[-1]], [coeff[0], coeff[-1]])
    # if i%500 == 0:
    #     courbe.plot()

stdout.write("\n")

if sk_learn:
    # from sklearn.linear_model import LinearRegression
    # from sklearn.ensemble import RandomForestRegressor
    from sklearn.ensemble import BaggingRegressor
    # from sklearn.tree import DecisionTreeRegressor

    from sklearn.model_selection import cross_validate

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers


def build_and_compile_model(norm, n):
    model = keras.Sequential([
        norm,
        layers.Dense(32, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(n)
    ])

    model.compile(loss='mean_absolute_error',
                  optimizer=tf.keras.optimizers.Adam(0.001))
    return model


if sk_learn:
    model = BaggingRegressor()

normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(data))

# model = build_and_compile_model(normalizer, 3)
if train_random:
    model_NN = build_and_compile_model(normalizer, 3*n_gauss)
else:
    model_NN = build_and_compile_model(normalizer, n_gauss + 4)

if test_random:

    mu = []
    sigma = []
    coeff = []

    for j in range(n_gauss):
        mu.append(rand_range(-3, 3))
        sigma.append(rand_range(0.5, 2))
        coeff.append(rand_range(0.5, 3))

    mu = sorted(mu)
    coeff = sorted(coeff)
    tmp = coeff[1]
    coeff[1] = coeff[2]
    coeff[2] = tmp

    courbe_test = poly_gauss(x, mu, sigma, coeff)

else:

    y_test = Gaussian(percent_D)

t1 = time()

model_NN.fit(data, target_full, epochs=n_epochs)#, validation_split=0.2)

t2 = time()
print("fini fit NN, in", t2-t1)

if test_random:
    result_NN = model_NN.predict(courbe_test.reshape(1, -1))[0]
else:
    result_NN = model_NN.predict(y_test.reshape(1, -1))[0]

if train_random:
    mu_NN = result_NN[:n_gauss]
    sigma_NN = abs(result_NN[n_gauss:2*n_gauss])
    coeff_NN = result_NN[2*n_gauss:]
else:
    mu_NN = result_NN[:n_gauss]
    sigma_NN = abs(result_NN[n_gauss:2+n_gauss])
    coeff_NN = result_NN[2+n_gauss:]

print(result_NN)
if test_random:
    print(merge_array(courbe_test.mu, courbe_test.sigma, courbe_test.coeff))

# plt.plot(courbe_test.x, courbe_test.y, "b", label="somme des courbes originelles")
# for i in range(3):
#     plt.plot(courbe_test.x, gaussienne(courbe_test.x, courbe_test.mu[i],
#                                        courbe_test.sigma[i], courbe_test.coeff[i]),
#              "b.", label="Courbe originelles")

# for i in range(3):
#     plt.plot(courbe_test.x, gaussienne(courbe_test.x, mu_full[i],
#                                        sigma_full[i], coeff_full[i]),
#              "g.", label="Courbe predites")

# resultat = poly_gauss(courbe_test.x, mu_full, sigma_full, coeff_full)
# plt.plot(resultat.x, resultat.y, "g", label="Somme des courbes prédites")
# plt.legend()
# plt.show()

if sk_learn:

    model_sk = model

    t1 = time()

    model_sk.fit(data, target_full)#, validation_split=0.2)

    t2 = time()

    print("fini fit full, in", t2-t1)

    # result_sk = model_sk.predict(courbe_test.y.reshape(1, -1))[0]
    result_sk = model_sk.predict(y_test.reshape(1, -1))[0]

    mu_sk = result_sk[:n_gauss]
    sigma_sk = abs(result_sk[n_gauss:2*n_gauss])
    coeff_sk = result_sk[2*n_gauss:3*n_gauss]

    print(result_sk)
    if test_random:
        print(merge_array(courbe_test.mu, courbe_test.sigma, courbe_test.coeff))


# model_mu = model

# # cv_result_mu = cross_validate(model_mu, data, target[:, :, 0], return_train_score=True)

# # print(cv_result_mu)

# model_mu.fit(data, target[:, :, 0])#, epochs=n_epochs)#, validation_split=0.2)

# print("fini fit mu")

# result_mu = model_mu.predict(courbe_test.y.reshape(1, -1))

# print("fini prediction mu")
# print(result_mu)
# print(courbe_test.mu)

# # print("Linear Regression result :", result)

# model_sigma = model
# model_sigma.fit(data, target[:, :, 1])#, epochs=n_epochs)#, validation_split=0.2)

# print("fini fit sigma")

# result_sigma = model_sigma.predict(courbe_test.y.reshape(1, -1))

# print("fini prediction sigma")
# print(result_sigma)
# print(courbe_test.sigma)

# model_coeff = model
# model_coeff.fit(data, target[:, :, 2]),# epochs=n_epochs)#, validation_split=0.2)

# print("fini fit coeff")

# result_coeff = model_coeff.predict(courbe_test.y.reshape(1, -1))

# print("fini prediction coeff")
# print(result_coeff)
# print(courbe_test.coeff)

# # print("Random Forest Regressor result", result)
# # print("Expected result :", courbe_test.mu)
# # courbe_test.plot()

# result_mu = result_mu[0]
# result_sigma = result_sigma[0]
# result_coeff = result_coeff[0]

# courbe_test.plot()

if test_random:

    plt.plot(courbe_test.x, courbe_test.y, "b", label="somme des courbes originelles")
    for i in range(n_gauss):
        if i == 0:
            plt.plot(courbe_test.x, gaussienne(courbe_test.x, courbe_test.mu[i],
                                               courbe_test.sigma[i], courbe_test.coeff[i]),
                     "b,", label="Courbe originelles")
        else:
            plt.plot(courbe_test.x, gaussienne(courbe_test.x, courbe_test.mu[i],
                                               courbe_test.sigma[i], courbe_test.coeff[i]),
                     "b,")

    resultat = poly_gauss(courbe_test.x, mu_NN, sigma_NN, coeff_NN)
    plt.plot(resultat.x, resultat.y, "g--", label="Somme des courbes prédites, réseau neuronal")

    for i in range(n_gauss):
        if i > 0:
            plt.plot(courbe_test.x, gaussienne(courbe_test.x, mu_NN[i],
                                               sigma_NN[i], coeff_NN[i]),
                     "g,")
        else:
            plt.plot(courbe_test.x, gaussienne(courbe_test.x, mu_NN[i],
                                               sigma_NN[i], coeff_NN[i]),
                     "g,", label="Courbe predites, réseau neuronal")

    if sk_learn:
        resultat = poly_gauss(courbe_test.x, mu_sk, sigma_sk, coeff_sk)
        plt.plot(resultat.x, resultat.y, "r--", label="Somme des courbes prédites, scickit learn")

        for i in range(n_gauss):
            if i == 0:
                plt.plot(courbe_test.x, gaussienne(courbe_test.x, mu_sk[i],
                                                   sigma_sk[i], coeff_sk[i]),
                         "r,", label="Courbe predites, scickit learn")
            else:
                plt.plot(courbe_test.x, gaussienne(courbe_test.x, mu_sk[i],
                                                   sigma_sk[i], coeff_sk[i]),
                         "r,")

else:

    # x = change_x(x)
    # mu_full = change_x(mu_full)
    # if sk_learn:
    #     result_mu = change_x(result_mu)

    plt.plot(x, y_test, "k", label="somme des courbes originelles")

    resultat_D = poly_gauss(x, mu_NN[:2], [sigma_NN[0], sigma_NN[0]],
                            [coeff_NN[0], coeff_NN[0]])
    resultat_H = poly_gauss(x, mu_NN[2:], [sigma_NN[1], sigma_NN[1]],
                            [coeff_NN[1], coeff_NN[1]])
    
    plt.plot(resultat_D.x, resultat_D.y, "r--", label="Somme des courbes prédites pour D, réseau neuronal")
    plt.plot(resultat_H.x, resultat_H.y, "b--", label="Somme des courbes prédites pour H, réseau neuronal")

    for i in range(2):
        if i > 0:
            plt.plot(x, gaussienne(x, mu_NN[i],
                                   sigma_NN[0], coeff_NN[0]),
                     "r,")
        else:
            plt.plot(x, gaussienne(x, mu_NN[i],
                                   sigma_NN[0], coeff_NN[0]),
                     "r,", label="Courbe predites pour D, réseau neuronal")

    for i in range(2, n_gauss):
        if i > 2:
            plt.plot(x, gaussienne(x, mu_NN[i],
                                   sigma_NN[1], coeff_NN[1]),
                     "b,")
        else:
            plt.plot(x, gaussienne(x, mu_NN[i],
                                   sigma_NN[1], coeff_NN[1]),
                     "b,", label="Courbe predites pour H, réseau neuronal")

    if sk_learn:
        resultat = poly_gauss(x, mu_sk, sigma_sk, coeff_sk)
        plt.plot(resultat.x, resultat.y, "r--", label="Somme des courbes prédites, scickit learn")

        for i in range(n_gauss):
            if i == 0:
                plt.plot(x, gaussienne(x, mu_sk[i],
                                       sigma_sk[i], coeff_sk[i]),
                         "r,", label="Courbe predites, scickit learn")
            else:
                plt.plot(x, gaussienne(x, mu_sk[i],
                                       sigma_sk[i], coeff_sk[i]),
                         "r,")

    rapport_isotopique = resultat_H.int() / (resultat_H.int() + resultat_D.int()) * 100
    print("Concentration isotopique :", rapport_isotopique, "% d'H, valeur théorique ", (1-percent_D)*100, "%")

plt.legend()
plt.show()
