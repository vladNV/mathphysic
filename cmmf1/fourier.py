import numpy as np
import matplotlib.pyplot as plt
import math
import Evaluate

function_title = 'u'' = -exp(x)'
f_str = 'np.exp(var)'


def analytic_f(l, h):
    x = np.arange(0, l + h, h)
    y = np.zeros(len(x))
    for i in range(0, len(x)):
        y[i] = (-math.exp(x[i]) + (math.e - 1) * x[i] + 1)
    return x, y


def f(x):
    expression = f_str.replace('var', str(x))
    return eval(expression)


def get_net_function(l, h):
    x = np.arange(0, l + h, h)
    fun = []
    for xi in x:
        fun.append(f(xi))
    return x, fun


def get_basis_functions(x, N):
    k = np.array(range(1, N)).reshape(N - 1, 1)
    xi = x[1:N]
    xi = xi.reshape(1, N - 1)
    kx = np.dot(k, xi)
    mu = np.sqrt(2 / l) * np.sin((np.pi / l) * kx)
    return mu


def fourier_decompose_f(mu, f, h, N):
    return h * np.dot(mu, f.reshape(N - 1, 1))


def eigenvalues(N, l, h):
    k = np.array(range(1, N))
    arg = np.pi * h / (2 * l) * k
    return np.sin(arg) ** 2 * 4 / (h ** 2)


def scalar(y, v, h):
    return h * np.dot(y, v.T)


def fourier_compose(C, mu):
    return np.dot(mu, C)


l = 1
N = 1000
h = l / N

x, y = get_net_function(l, h)

mu = get_basis_functions(x, N)

F = fourier_decompose_f(mu, np.array(y[1:-1]), h, N)

lam = eigenvalues(N, l, h)

C = F / lam.reshape(N - 1, 1)

res = fourier_compose(C, mu)
x_a, y_a = analytic_f(l, h)
print('ERROR: ' + str(Evaluate.Evaluate.get_error(y_a[:-2], res)))
plt.plot(x[:-2], y_a[:-2], color="red", linewidth=1.0, linestyle='-', label='analytic')
plt.plot(x[:-2], res, color="green", linewidth=1.0, linestyle='-', label=function_title)
plt.legend(loc='upper center')
plt.show()
