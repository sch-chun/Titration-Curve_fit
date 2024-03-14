import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.optimize import curve_fit
from matplotlib import use
Kw = 10 ** (-14)
V0 = 50  # mL
# ca = 1  # mol/L
cb = 0.125  # mol/L
# Ka = 1.75 * (10 ** (-5))
x_data = np.array([0, 4, 8, 20, 36, 39.2, 39.92, 40, 40.08, 40.8, 41.6], dtype="float64")
y_data = np.array([2.4, 2.86, 3.21, 3.81, 4.76, 5.5, 6.51, 8.25, 10, 11, 11.24], dtype="float64")


def sb_wa(v, ca, ka):
    global cb, V0, Kw
    b = cb * v / (v + V0)
    c = ca * V0 / (v + V0)

    def eq(h, eb):
        denominator = ka + h
        a = c * ka / denominator
        oh = Kw / h
        return h + eb - a - oh
    left = 0
    right = 14
    eps = 0.0001
    while right - left > eps:
        middle = (left + right) / 2
        if eq(math.pow(10, -middle), b) == 0:
            break
        if eq(math.pow(10, -left), b) * eq(math.pow(10, -middle), b) < 0:
            right = middle
        else:
            left = middle
    return (left + right) / 2


def f(n, ca, ka):
    out = []
    for i in n:
        out.append(sb_wa(i, ca, ka))
    return out


line = curve_fit(f, x_data, y_data, p0=[0.1, 0.00016], bounds=[(0, 0), (1, 1)])
print(line)
ca_fit, ka_fit = line[0]
pcov = line[1]
print(ca_fit)
print(ka_fit)
print(np.sqrt(np.diag(pcov)))
use("TkAgg")
plt.figure()
x_fit = np.linspace(0, 41.6, 416)
y_fit = f(x_fit, ca_fit, ka_fit)
plt.plot(x_fit, y_fit)
plt.scatter(x_data, y_data, s=10, marker="s")
plt.xlabel("V/mL")
plt.ylabel("pH")


def nf(x):
    global ca_fit, ka_fit
    return f(x, ca_fit, ka_fit)


plt.figure()
dy_fit = np.gradient(nf(x_fit))
dx_fit = np.gradient(x_fit)
dydx_fit = dy_fit / dx_fit
plt.plot(x_fit, dydx_fit)
dy = np.diff(y_data)
dx = np.diff(x_data)
xa = (x_data[:-1] + x_data[1:]) / 2
plt.scatter(xa, dy / dx, s=10, marker="s")
plt.xlabel("V/mL")
plt.ylabel("dpH/dV")

plt.figure()
d2ydx_fit = np.gradient(dydx_fit)
d2ydx2_fit = d2ydx_fit / dx_fit
d2y = np.diff(dy)
dxa = np.diff(xa)
plt.scatter(x_data[1:-1], d2y / dxa ** 2, s=10, marker="s")
plt.plot(x_fit, d2ydx2_fit)
plt.xlabel("V/mL")
plt.ylabel(r"$ d^2pH/dV^2 $")
plt.show()
