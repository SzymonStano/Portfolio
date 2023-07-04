import numpy as np
from numpy import matlib
import matplotlib.pyplot as plt
import scipy.stats as ss
from statsmodels.distributions.empirical_distribution import ECDF

mi = 2
sigma = 2
n1 = 10 ** 3
n2 = 10**4
interval = (-5.0, 5.0)
theta = np.linspace(interval[0], interval[1], n2)

random_sample = np.random.normal(mi, sigma, n1)


def char_f(t):
    return np.exp(mi*1j*t - (sigma**2 * t**2)/2)


results = char_f(theta)


def emp_char_f_matrix(t):
    return np.mean(np.exp(1j * random_sample * t), axis=1)


theta_vertical = theta.reshape(-1, 1)
resultsv2 = emp_char_f_matrix(theta_vertical)

plt.step(theta, np.real(resultsv2), label="empiryczna f.ch.")
plt.plot(theta, np.real(results), label="teoretyczna f.ch.")
plt.title("Rozkład normalny")
plt.ylabel("$Re(\u03C6(\u03B8))$")
plt.xlabel("\u03B8")
plt.legend()
plt.show()

plt.step(np.real(resultsv2), np.imag(resultsv2), label="empiryczna f.ch.")
plt.plot(np.real(results), np.imag(results), label="teoretyczna f.ch.")
plt.title("Rozkład normalny")
plt.ylabel("$Im(\u03C6(\u03B8))$")
plt.xlabel("$Re(\u03C6(\u03B8))$")
plt.legend()
plt.show()

plt.step(theta, np.imag(resultsv2), label="empiryczna f.ch.")
plt.plot(theta, np.imag(results), label="teoretyczna f.ch.")
plt.title("Rozkład normalny")
plt.ylabel("$Im(\u03C6(\u03B8))$")
plt.xlabel("$\u03B8$")
plt.legend()
plt.show()

ax = plt.figure().add_subplot(projection='3d')
ax.step(theta, np.real(resultsv2), np.imag(resultsv2), label="empiryczna f.ch.")
ax.plot(theta, np.real(results), np.imag(results), label="teoretyczna f.ch.")
ax.set_xlabel("\u03B8")
ax.set_ylabel("$Re(\u03C6(\u03B8))$")
ax.set_zlabel("$Im(\u03C6(\u03B8))$")
plt.title("Rozkład normalny")
plt.legend()
plt.show()



n1 = 10 ** 3
interval = (-5.0, 5.0)
theta = np.linspace(interval[0], interval[1], n1)
lmbd = 1

random_sample = np.random.exponential(lmbd, n1)


def char_f(t):
    return 1/(1-1j*theta/lmbd)


results = char_f(theta)

x_trans = np.matrix.transpose(matlib.repmat(random_sample, n1, 1))
theta_vertical = matlib.repmat(theta, n1, 1)


def emp_char_f_matrix(t):
    return np.sum(np.exp(1j * x_trans * theta_vertical), axis=0) / n1


resultsv2 = emp_char_f_matrix(theta)

plt.step(theta, np.real(resultsv2), label="empiryczna f.ch.")
plt.plot(theta, np.real(results), label="teoretyczna f.ch.")
plt.title("Real values in theta")
plt.ylabel("$Re(\u03C6(\u03B8))$")
plt.xlabel("\u03B8")
plt.legend()
plt.title("Rozkład wykładniczy")
plt.show()

plt.step(np.real(resultsv2), np.imag(resultsv2), label="empiryczna f.ch.")
plt.plot(np.real(results), np.imag(results), label="teoretyczna f.ch.")
plt.title("Real versus Imaginary values")
plt.ylabel("$Im(\u03C6(\u03B8))$")
plt.xlabel("$Re(\u03C6(\u03B8))$")
plt.legend()
plt.title("Rozkład wykładniczy")
plt.show()

plt.step(theta, np.imag(resultsv2), label="empiryczna f.ch.")
plt.plot(theta, np.imag(results), label="teoretyczna f.ch.")
plt.title("Real versus Imaginary values")
plt.ylabel("$Im(\u03C6(\u03B8))$")
plt.xlabel("$\u03B8$")
plt.legend()
plt.title("Rozkład wykładniczy")
plt.show()

ax = plt.figure().add_subplot(projection='3d')
ax.step(theta, np.real(resultsv2), np.imag(resultsv2), label="empiryczna f.ch.")
ax.plot(theta, np.real(results), np.imag(results), label="teoretyczna f.ch.")
ax.set_xlabel("\u03B8")
ax.set_ylabel("$Re(\u03C6(\u03B8))$")
ax.set_zlabel("$Im(\u03C6(\u03B8))$")
plt.title("Rozkład wykładniczy")
plt.legend()
plt.show()



