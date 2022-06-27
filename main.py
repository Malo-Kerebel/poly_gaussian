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
    return coeff * np.exp(-4*np.log(2.)*(x-mu)*(x-mu)/sigma)


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


def gen_test(test_random=False, rand_B=False, rand_D=False, B = 2, percent_D = 0.75,):

    if rand_B:
        B = rand_range(1.5, 3.5)  # Champ magnétique en tesla

    if rand_D:
        percent_D = rand_range(0.55, 0.95)  # Pourcentage de deutérium dans la valeur de test (uniquement si test_random == False)
    
    if test_random:

        mu = []
        sigma = []
        coeff = []

        for j in range(n_gauss):
            mu.append(rand_range(-3, 3))
            sigma.append(rand_range(0.5, 2))
            coeff.append(rand_range(0.5, 3))

        courbe_test = poly_gauss(x, mu, sigma, coeff, noise_test)
        return courbe_test.y, merge_array(mu, sigma, coeff), None, None

    else:

        y_test, expected = Gaussian(percent_D, B, noise=noise_test)
        return y_test, expected, B, percent_D


def test(y_test, expected, B=None, percent_D=None):
    
    result_NN = model_NN.predict(y_test.reshape(1, -1))[0]

    # print(result_NN)
    # print(expected)

    if sk_learn:

        model_sk = model

        t1 = time()

        model_sk.fit(data, target_full)#, validation_split=0.2)

        t2 = time()

        print("fini fit full, in", t2-t1)

        # result_sk = model_sk.predict(courbe_test.y.reshape(1, -1))[0]
        result_sk = model_sk.predict(y_test.reshape(1, -1))[0]

        print(result_sk)
        if test_random:
            print(merge_array(courbe_test.mu, courbe_test.sigma, courbe_test.coeff))

        return result_NN, result_sk

    return result_NN, None


def show(result_NN, result_sk=None, show=True):

    if train_random:
        mu_NN = result_NN[:n_gauss]
        sigma_NN = abs(result_NN[n_gauss:2*n_gauss])
        coeff_NN = result_NN[2*n_gauss:]
    else:
        mu_NN = result_NN[:n_gauss]
        sigma_NN = abs(result_NN[n_gauss:2+n_gauss])
        coeff_NN = result_NN[2+n_gauss:]

    plt.figure()

    if test_random:

        plt.plot(x_show, courbe_test.y, "b", label="somme des courbes originelles")
        for i in range(n_gauss):
            if i == 0:
                plt.plot(x_show, gaussienne(courbe_test.x, courbe_test.mu[i],
                                            courbe_test.sigma[i], courbe_test.coeff[i]),
                         "b,", label="Courbe originelles")
            else:
                plt.plot(x_show, gaussienne(courbe_test.x, courbe_test.mu[i],
                                            courbe_test.sigma[i], courbe_test.coeff[i]),
                         "b,")

        resultat = poly_gauss(courbe_test.x, mu_NN, sigma_NN, coeff_NN)
        plt.plot(resultat.x, resultat.y, "g--", label="Somme des courbes prédites, réseau neuronal")

        for i in range(n_gauss):
            if i > 0:
                plt.plot(x_show, gaussienne(courbe_test.x, mu_NN[i],
                                            sigma_NN[i], coeff_NN[i]),
                         "g,")
            else:
                plt.plot(x_show, gaussienne(courbe_test.x, mu_NN[i],
                                            sigma_NN[i], coeff_NN[i]),
                         "g,", label="Courbe predites, réseau neuronal")

        if sk_learn:
            mu_sk = result_sk[:n_gauss]
            sigma_sk = abs(result_sk[n_gauss:2*n_gauss])
            coeff_sk = result_sk[2*n_gauss:3*n_gauss]
            
            resultat = poly_gauss(courbe_test.x, mu_sk, sigma_sk, coeff_sk)
            plt.plot(x_show, resultat.y, "r--", label="Somme des courbes prédites, scickit learn")

            for i in range(n_gauss):
                if i == 0:
                    plt.plot(x_show, gaussienne(courbe_test.x, mu_sk[i],
                                                sigma_sk[i], coeff_sk[i]),
                             "r,", label="Courbe predites, scickit learn")
                else:
                    plt.plot(x_show, gaussienne(courbe_test.x, mu_sk[i],
                                                sigma_sk[i], coeff_sk[i]),
                             "r,")

    else:

        # x = change_x(x)
        # mu_full = change_x(mu_full)
        # if sk_learn:
        #     result_mu = change_x(result_mu)

        plt.plot(x_show, y_test, "k", label="somme des courbes originelles")

        resultat_D = poly_gauss(x, mu_NN[:2], [sigma_NN[0], sigma_NN[0]],
                                [coeff_NN[0], coeff_NN[0]])
        resultat_H = poly_gauss(x, mu_NN[2:], [sigma_NN[1], sigma_NN[1]],
                                [coeff_NN[1], coeff_NN[1]])

        if save_txt:
            with open("B="+str(B)+",D="+str(percent_D)+"N="+str(N)+".txt", 'w') as f:
                f.write("# y_true, y_H, y_D\n")
                for i in range(n):
                    tmp = f"{y_test[i]} {resultat_H.y[i]} {resultat_D.y[i]}\n"
                    f.write(tmp)

        plt.plot(x_show, resultat_D.y, "r--", label="Somme des courbes prédites pour D, réseau neuronal")
        plt.plot(x_show, resultat_H.y, "b--", label="Somme des courbes prédites pour H, réseau neuronal")

        for i in range(2):
            if i > 0:
                plt.plot(x_show, gaussienne(x, mu_NN[i],
                                            sigma_NN[0], coeff_NN[0]),
                         "r,")
            else:
                plt.plot(x_show, gaussienne(x, mu_NN[i],
                                            sigma_NN[0], coeff_NN[0]),
                         "r,", label="Courbe predites pour D, réseau neuronal")

        for i in range(2, n_gauss):
            if i > 2:
                plt.plot(x_show, gaussienne(x, mu_NN[i],
                                            sigma_NN[1], coeff_NN[1]),
                         "b,")
            else:
                plt.plot(x_show, gaussienne(x, mu_NN[i],
                                            sigma_NN[1], coeff_NN[1]),
                         "b,", label="Courbe predites pour H, réseau neuronal")

        if sk_learn:

            mu_sk = result_sk[:n_gauss]
            sigma_sk = abs(result_sk[n_gauss:2*n_gauss])
            coeff_sk = result_sk[2*n_gauss:3*n_gauss]
            
            resultat = poly_gauss(x, mu_sk, sigma_sk, coeff_sk)
            plt.plot(resultat.x, resultat.y, "r--", label="Somme des courbes prédites, scickit learn")

            for i in range(n_gauss):
                if i == 0:
                    plt.plot(x_show, gaussienne(x, mu_sk[i],
                                                sigma_sk[i], coeff_sk[i]),
                             "r,", label="Courbe predites, scickit learn")
                else:
                    plt.plot(x_show, gaussienne(x, mu_sk[i],
                                                sigma_sk[i], coeff_sk[i]),
                             "r,")

        rapport_isotopique = resultat_H.int() / (resultat_H.int() + resultat_D.int()) * 100
        print("Concentration isotopique :", rapport_isotopique, "% d'H, valeur théorique ", (1-percent_D)*100, "%")

        error = abs(rapport_isotopique-(1-percent_D)*100)/((1-percent_D)*100)*100
        print("Erreur de :", error, '%')

        if save_png:
            plt.savefig("B="+str(B)+",D="+str(percent_D)+",N="+str(N)+",error="+str(error)+".png")

    plt.legend()
    if show:
        plt.show()

    if test_random:
        return resultat.y, None
    else:
        return resultat_D.y + resultat_H.y, error


sk_learn = False  # Est-ce que du ML doit être utilisé
train_random = False  # Est-ce que les valeurs de test doivent être aléatoire ou
# utilisé les contraintes supplémentaires
test_random = False   # Est-ce que la valeur de test doit être aléatoire ou
# venir de données synthétique

N = 1000  # Nombre de valeurs d'entrainement
n = 1000    # Nombre de points dans les courbes d'entrainements
n_gauss = 4  # nombre de gaussienne à additionner
n_test = 10  # Nombre de test à faire pour déterminer l'erreur moyenne

n_epochs = 2   # Nombre d'épochs sur lesquel le NN doit itérer

x_min = -10  # Valeurs minimales et maximales du x des données
x_max = 10

lambda_min = 6558  # Valeurs minimales des longueurs d'ondes étudier
lambda_max = 6565

noise_train = True
noise_test = True

train_show = False
test_show = False

save_txt = False
save_png = False

x = np.linspace(x_min, x_max, n)
x_show = np.linspace(lambda_min, lambda_max, n)

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
            mu.append(rand_range(x_min + (x_max - x_min) / 6, x_max - (x_max - x_min) / 6))
            sigma.append(rand_range(0.1, 2.5))
            coeff.append(rand_range(0.01, 0.7))
        # mu = sorted(mu)
        # coeff = sorted(coeff)
        # tmp = coeff[1]
        # coeff[1] = coeff[2]
        # coeff[2] = tmp

    else:

        small_gap = rand_range(0.05, 2)
        # big_gap = rand_range(4, 8)
        big_gap = 1.8*(x_max - x_min) / (lambda_max - lambda_min)

        first_mu = rand_range(x_min + (x_max - x_min) / 6, x_max - (x_max - x_min) / 6)
        mu.append(first_mu - small_gap)
        mu.append(first_mu + small_gap)
        mu.append(first_mu + big_gap - small_gap)
        mu.append(first_mu + big_gap + small_gap)

        sigma_first = rand_range(0.5, 4)

        sigma.append(sigma_first)
        sigma.append(sigma_first)

        sigma_second = rand_range(0.5, 4)

        sigma.append(sigma_second)
        sigma.append(sigma_second)

        coeff_first = rand_range(0.1, 0.8)

        coeff.append(coeff_first)
        coeff.append(coeff_first)

        coeff_second = rand_range(0.1, 0.6)

        coeff.append(coeff_second)
        coeff.append(coeff_second)

    courbe = poly_gauss(x, mu, sigma, coeff, noise_train)

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

    if train_show and i%(N//2) == 0:
        courbe.plot()

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


if sk_learn:
    model = BaggingRegressor()

normalizer = tf.keras.layers.Normalization(axis=-1)
# normalizer.adapt(np.array(data))

# model = build_and_compile_model(normalizer, 3)
if train_random:
    model_NN = build_and_compile_model(normalizer, 3*n_gauss, n)
else:
    model_NN = build_and_compile_model(normalizer, n_gauss + 4, n)

t1 = time()

model_NN.fit(data, target_full, epochs=n_epochs)#, validation_split=0.2)

t2 = time()
print("fini fit NN, in", t2-t1)

mean_squared_error = np.zeros((n_test))
mean_absolute_error = np.zeros((n_test))
error = np.zeros((n_test))

for i in range(n_test):

    y_test, expected, B, percent_D = gen_test(False, True, True)

    result_NN, _ = test(y_test, expected, B, percent_D)

    y_predict, error[i] = show(result_NN, show=test_show)

    mean_squared_error[i] = np.mean((y_test - y_predict)*(y_test - y_predict))
    mean_absolute_error[i] = np.mean(np.abs(y_test - y_predict))

print("mean prediction error", np.mean(error))
print("mean squared error :", np.mean(mean_squared_error))
print("mean absolute error :", np.mean(mean_absolute_error))
