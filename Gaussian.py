import numpy as np


def doppler(lambdax, lambda0, FWHM, B):
    return (1*np.sqrt(np.log(2.)/np.pi)/FWHM)*np.exp(-4*np.log(2.)*np.power((lambdax - lambda0-0.21*B), 2.) / (np.power(FWHM, 2.))) + \
           (1*np.sqrt(np.log(2.)/np.pi)
            / FWHM)*np.exp(-4*np.log(2.)
                           * np.power((lambdax - lambda0+0.21*B), 2.)
                           / (np.power(FWHM, 2.)))


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def Gaussian(percent_D):

    mass_h = 1
    mass_d = 2
    lam0 = 6561.0  # angströms
    B = 2.0

    # FWHM=7.16E-7*lambda0*np.sqrt(Temp/mass)

    Temp1 = 23208.0  # 2.0 eV
    Temp2 = 174060.  # 15 eV
    FWHM_h1 = 7.16E-7*lam0*np.sqrt(Temp1/mass_h)
    FWHM_h2 = 7.16E-7*lam0*np.sqrt(Temp2/mass_h)

    FWHM_d1 = 7.16E-7*lam0*np.sqrt(Temp1/mass_d)
    FWHM_d2 = 7.16E-7*lam0*np.sqrt(Temp2/mass_d)

    # x = np.linspace(6562.4, 6564., 1000)
    x2 = np.linspace(6558, 6565., 1000)
    y0 = 0.55*(percent_D*doppler(x2, 6561.0, FWHM_d1, B)
               + (1-percent_D)*doppler(x2, 6562.8, FWHM_h1, B)) + \
        0.45*(percent_D*doppler(x2, 6561.0, FWHM_d2, B)
              + (1-percent_D)*doppler(x2, 6562.8, FWHM_h2, B))

    return y0
