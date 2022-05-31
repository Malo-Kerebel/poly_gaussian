#!/usr/bin/python

from double_gaussienne import poly_gauss
import numpy as np
from numpy.random import random
import matplotlib.pyplot as plt

from sys import stdout


def rand_range(a, b):
    return random() * (b - a) + a


def gaussienne(x, mu, sigma, coeff):
    return coeff * np.exp(-(x-mu)*(x-mu)/sigma)


def merge_array(mu, sigma, coeff):
    assert len(mu) == len(sigma) == len(coeff)
    ret = np.zeros(len(mu)*3)

    for i in range(len(mu)):
        ret[i] = mu[i]
    for i in range(len(mu)):
        ret[i+3] = sigma[i]
    for i in range(len(mu)):
        ret[i+6] = coeff[i]
    return ret


N = 50000
n = 2000

x = np.linspace(-10, 10, n)

train_share = 1
N_train = int(train_share * N)
N_test = N - N_train

data = np.zeros((N_train, n))
target = np.zeros((N_train, 3, 3))
target_full = np.zeros((N_train, 9))

data_test = np.zeros((N_test, n))
target_test = np.zeros((N_test, 3, 3))
target_test_full = np.zeros((N_test, 9))

stdout.write("generating train values\n")

for i in range(N):
    
    stdout.write("\r%5d/%6d" % (i+1, N))
    stdout.flush()
    # print(f"generating train values {i+1}/{N}")

    mu = []
    sigma = []
    coeff = []
    for j in range(3):
        mu.append(rand_range(-4, 4))
        sigma.append(rand_range(0.5, 3))
        coeff.append(rand_range(0.5, 4))

    mu = sorted(mu)
    coeff = sorted(coeff)
    tmp = coeff[1]
    coeff[1] = coeff[2]
    coeff[2] = tmp
              
    courbe = poly_gauss(x, mu, sigma, coeff)

    if i < N_train:
        data[i, :] = courbe.y

        target[i, :, 0] =  mu
        target[i, :, 1] =  sigma
        target[i, :, 2] =  coeff
        target_full[i, :] = merge_array(mu, sigma, coeff)
    else:
        data_test[i-N_train, :] = courbe.y

        target_test[i-N_train, :] =  mu
        target_test[i-N_train, :, 1] =  sigma
        target_test[i-N_train, :, 2] =  coeff
        target_test_full[i-N_train, :] = merge_array(mu, sigme, coeff)
    # if i%500 == 0:
    #     courbe.plot()

stdout.write("\n")
    
from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.ensemble import BaggingRegressor
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

# model = BaggingRegressor(base_estimator=DecisionTreeRegressor(max_depth=3),
#                          n_estimators=100)

model = LinearRegression()

normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(data))

# model = build_and_compile_model(normalizer, 3)
model_full = build_and_compile_model(normalizer, 9)

n_epochs = 15

mu = []
sigma = []
coeff = []

for j in range(3):
    mu.append(rand_range(-3, 3))
    sigma.append(rand_range(0.5, 2))
    coeff.append(rand_range(0.5, 3))

mu = sorted(mu)
coeff = sorted(coeff)
tmp = coeff[1]
coeff[1] = coeff[2]
coeff[2] = tmp

courbe_test  = poly_gauss(x, mu, sigma, coeff)


model_full.fit(data, target_full, epochs=n_epochs)#, validation_split=0.2)

print("fini fit full")

result_full = model_full.predict(courbe_test.y.reshape(1, -1))[0]

mu_full = result_full[:3]
sigma_full = result_full[3:6]
coeff_full = result_full[6:9]

print(result_full)
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

model_sk = model

model_sk.fit(data, target_full)#, validation_split=0.2)

print("fini fit full")

result_sk = model_sk.predict(courbe_test.y.reshape(1, -1))[0]

result_mu = result_sk[:3]
result_sigma = result_sk[3:6]
result_coeff = result_sk[6:9]

print(result_sk)
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

plt.plot(courbe_test.x, courbe_test.y, "b", label="somme des courbes originelles")
for i in range(3):
    if i == 0:
        plt.plot(courbe_test.x, gaussienne(courbe_test.x, courbe_test.mu[i],
                                           courbe_test.sigma[i], courbe_test.coeff[i]),
                 "b,", label="Courbe originelles")
    else:
        plt.plot(courbe_test.x, gaussienne(courbe_test.x, courbe_test.mu[i],
                                           courbe_test.sigma[i], courbe_test.coeff[i]),
                 "b,")

resultat = poly_gauss(courbe_test.x, result_mu, result_sigma, result_coeff)
plt.plot(resultat.x, resultat.y, "r", label="Somme des courbes prédites, scickit learn")

for i in range(3):
    if i == 0:
        plt.plot(courbe_test.x, gaussienne(courbe_test.x, result_mu[i],
                                           result_sigma[i], result_coeff[i]),
                 "r,", label="Courbe predites, scickit learn")
    else:
        plt.plot(courbe_test.x, gaussienne(courbe_test.x, result_mu[i],
                                           result_sigma[i], result_coeff[i]),
                 "r,")
                 
resultat = poly_gauss(courbe_test.x, mu_full, sigma_full, coeff_full)
plt.plot(resultat.x, resultat.y, "g", label="Somme des courbes prédites, réseau neuronal")

for i in range(3):
    if i > 0:
        plt.plot(courbe_test.x, gaussienne(courbe_test.x, mu_full[i],
                                           sigma_full[i], coeff_full[i]),
                 "g,")
    else:
        plt.plot(courbe_test.x, gaussienne(courbe_test.x, mu_full[i],
                                           sigma_full[i], coeff_full[i]),
                 "g,", label="Courbe predites, réseau neuronal")


plt.legend()
plt.show()
