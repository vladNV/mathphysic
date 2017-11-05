import matplotlib.pyplot as plt
import numpy as np

N = 1000
x = []
y = [0 for i in range(0, N+1)]

# 0 < x < l
a = 0
l = 1
h = l / N
# { k1*u(0) + k2*u'(0) = g1
# { l1*u(l) + l2*u'(l) = g2

k1 = 0  # beta
k2 = 1  # -k
l1 = 2
l2 = -1

g1 = 3
g2 = 2.71828182846

A = []
B = []
C = []
D = []


# q(x) = -2x
# r(x) = 0
# f(x) = 2/x^3 - 2
# ku''(x) - r(x)*u'(x) - q(x)u(x) = -f(x)

def function_q(x):
    return -1


def function_r(x):
    return -2


def function_f(x):
    return -2 * np.exp(x)


def coeff():
    for i in range(N+1):
        A.append(1 - (function_q(x[i]) * h / 2))
        B.append(1 + (function_q(x[i]) * h / 2))
        C.append(2 - (function_r(x[i]) * h * h))
        D.append(function_f(x[i]) * h * h)

for i in range(0, N+1):
    x.append(a + h * i)

alpha = []
beta = []

c0 = k2
z0 = h * (k1 - k2 / h)
alpha.append(-c0 / z0)

cl = g1
zl = k1 - k2 / h
beta.append(cl / zl)

t1 = t2 = 0
coeff()
for i in range(0, N):
    t1 = B[i] / (C[i] - alpha[i] * A[i])
    t2 = (A[i] * beta[i] - D[i]) / (C[i] - alpha[i] * A[i])
    alpha.append(t1)
    beta.append(t2)

y[N] = (l2 * beta[N] + g2 * h) / (l2 + h * l1 - l2 * alpha[N])

for i in range(N, 0, -1):
    y[i - 1] = alpha[i] * y[i] + beta[i]

plt.plot(x, y)
plt.show()
