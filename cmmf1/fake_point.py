# from CMFFunc import *
# import matplotlib.pyplot as plt
# import math
# import Evaluate
#
# k = 1
# f_str = 'np.cos(var)'
# q_str = '1'
# beta = 1
# gamma1 = 1
# gamma2 = 0
#
# l = math.pi
# N = 1000
# h = l / N
#
#
# def analytic_f(l, h):
#     x = np.arange(0, l + h, h)
#     y = np.zeros(len(x))
#     for i in range(0, len(x)):
#         y[i] = (math.exp(-math.pi) / 2 - math.exp(-2 * math.pi) / 4) * math.exp(x[i]) + \
#                1 / 4 * math.exp(-x[i]) + math.cos(x[i]) / 2
#     return x, y
#
#
# def f(x):
#     expression = f_str.replace('var', str(x))
#     return eval(expression)
#
#
# def q(x):
#     expression = q_str.replace('var', str(x))
#     return eval(expression)
#
#
# def get_net_function(l, h):
#     x = np.arange(0, l + h, h)
#     fun = []
#     for xi in x:
#         fun.append(f(xi))
#     return x, fun
#
#
# def get_net_function_q(l, h):
#     x = np.arange(0, l, h)
#     fun = []
#     for xi in x:
#         fun.append(q(xi))
#     return x, fun
#
#
# def solve():
#     x, y = get_net_function(l, h)
#     q_x, q_y = get_net_function_q(l, h)
#
#     A = np.zeros((N, 1))
#     B = np.zeros((N, 1))
#     C = np.zeros((N, 1))
#
#     C[0, 0] = k / h + beta + h / 2 * q_x[0]
#     B[0, 0] = -k / h
#
#     for i in range(1, N):
#         A[i, 0] = -k / h ** 2
#         C[i, 0] = 2 * k / h ** 2 + q_y[i]
#         B[i, 0] = -k / h ** 2
#     C[N - 1, 0] = 1
#     a = np.zeros((N + 1, 1))
#     b = np.zeros((N + 1, 1))
#
#     a[0, 0] = -B[0, 0] / C[0, 0]
#     b[0, 0] = (gamma1 + h / 2 * y[0]) / C[0, 0]
#
#     for i in range(1, N):
#         a[i, 0] = -B[i - 1, 0] / (A[i - 1, 0] * a[i - 1, 0] + C[i - 1, 0])
#         b[i, 0] = (y[i] - A[i - 1, 0] * b[i - 1, 0]) / (A[i - 1, 0] * a[i - 1, 0] + C[i - 1, 0])
#
#     y_hat = np.zeros((N, 1))
#     y_hat[N - 1, 0] = gamma2
#     for i in range(N - 2, -1, -1):
#         y_hat[i] = a[i + 1, 0] * y_hat[i + 1, 0] + b[i + 1, 0]
#
#     x_a, y_a = analytic_f(l, h)
#     print(Evaluate.Evaluate.get_error(y_a[0:-1], y_hat))
#     plt.plot(x[0:-1], y_hat, color='red', linewidth=1.0, linestyle='-', label='u'' - u = -cos(x)')
#     plt.plot(x_a[0:-1], y_a[0:-1], color='blue', linewidth=1.0, linestyle='-', label='analytic')
#     plt.legend(loc='upper center')
#     plt.show()
#
#
# solve()


