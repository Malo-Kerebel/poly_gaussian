import numpy as np
import matplotlib.pyplot as plt


class poly_gauss(object):

    def __init__(self, x, mus, sigmas, coeffs):

        self.x = x
        self.y = np.zeros(x.shape)
        self.mu = mus
        self.sigma = sigmas
        self.coeff = coeffs

        for i in range(len(mus)):
            self.y += coeffs[i] * np.exp(-(x-mus[i])*(x-mus[i])/sigmas[i])

    def __repr__(self):

        ret = f"poly gaussian object of {len(self.mu)} gaussians"
        for i in range(len(self.mu)):
            ret += "\n"
            ret += f"{i+1}: average {self.mu[i]}, std deviation {self.sigma[i]} and coefficient {self.coeff[i]}."

        return ret

    def plot(self):
        plt.plot(self.x, self.y)
        plt.show()

    def add_curve(self, mu, sigma, coeff):
        self.mu.append(mu)
        self.sigma.append(sigma)
        self.coeff.append(coeff)

        self.y += coeff * np.exp(-(self.x-mu)*(self.x-mu)/sigma)

    def add_curves(self, mus, sigmas, coeffs):

        for i in range(len(mus)):
            self.add_curve(mus[i], sigmas[i], coeffs[i])

    def get_y(self):
        return self.y

    def get_mu(self):
        return self.mu

    def get_sigma(self):
        return self.sigma

    def coeff(self):
        return self.coeff

    def int(self):
        return np.trapz(self.y, x=self.x)


def main():

    x = np.linspace(-10, 10, 250)
    courbe = poly_gauss(x, [-1, 1], [1, 1], [1, 1])
    courbe.add_curve(0, 2, 2)
    print(courbe)
    courbe.plot()


if __name__ == '__main__':
    main()
