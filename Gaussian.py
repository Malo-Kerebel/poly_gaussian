import numpy as np


def doppler(lambdax, lambda0, FWHM, B):
    coeff = 1 * np.sqrt(np.log(2.)/np.pi)/FWHM
    premiere_courbe = coeff * np.exp(-4*np.log(2.) * np.power((lambdax - lambda0-0.21*B), 2.) / (np.power(FWHM, 2.)))
    deuxieme_courbe = coeff * np.exp(-4*np.log(2.) * np.power((lambdax - lambda0+0.21*B), 2.) / (np.power(FWHM, 2.)))
    return premiere_courbe + deuxieme_courbe


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def Gaussian(percent_D, B=2.0, percent_temp1=1, Temp1=23208, Temp2=174060, n=1000, noise=False):

    mass_h = 1
    mass_d = 2
    lam0_D = 6561.0  # angströms
    lam0_H = 6562.8  # angströms
    # B = 2.0

    # FWHM=7.16E-7*lambda0*np.sqrt(Temp/mass)

    Temp1 = 23208.0  # 2.0 eV
    Temp2 = 174060.  # 15 eV
    FWHM_h1 = 7.16E-7*lam0_H*np.sqrt(Temp1/mass_h)
    FWHM_h2 = 7.16E-7*lam0_H*np.sqrt(Temp2/mass_h)

    FWHM_d1 = 7.16E-7*lam0_D*np.sqrt(Temp1/mass_d)
    FWHM_d2 = 7.16E-7*lam0_D*np.sqrt(Temp2/mass_d)

    # x = np.linspace(6562.4, 6564., 1000)
    x2 = np.linspace(6558, 6565., n)
    y0 = percent_temp1*(percent_D*doppler(x2, lam0_D, FWHM_d1, B)
               + (1-percent_D)*doppler(x2, lam0_H, FWHM_h1, B)) + \
        (1-percent_temp1)*(percent_D*doppler(x2, lam0_D, FWHM_d2, B)
              + (1-percent_D)*doppler(x2, lam0_H, FWHM_h2, B))

    ret = [lam0_D-0.21*B, lam0_D+0.21*B, lam0_H-0.21*B, lam0_H+0.21*B,
           FWHM_d1, FWHM_h1, 1 * np.sqrt(np.log(2.)/np.pi)/FWHM_h1*percent_D, 1 * np.sqrt(np.log(2.)/np.pi)/FWHM_h1*(1-percent_D)]

    if noise:
        y0 = y0 + 0.0025 * np.random.normal(size=y0.size) # adding some noise

    return y0, ret
