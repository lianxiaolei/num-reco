import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd


# def func(x, a, b, c, d, e):
# def func(x,
#          a0, a1, a2, a3, a4, a5, a6, a7, a8, a9,
#          b0, b1, b2, b3, b4, b5, b6, b7, b8, b9,
#          c0, c1, c2, c3, c4, c5, c6, c7, c8, c9,
#          e0, e1, e2, e3, e4, e5, e6, e7, e8, e9
#          ):

def func(x,
         a0, a1, a2, a3, a4, a5,
         b0, b1, b2, b3, b4, b5,
         c0, c1, c2, c3, c4, c5,
         e0, e1, e2, e3, e4, e5,
         ):
    # print(-b * x)
    # return a * np.exp(-b * x) + c
    res = 0
    for i in range(6):
        # if i < 3:
        res += \
               locals()['b%s' % i] * np.sin(locals()['c%s' % i] * x) + locals()['e%s' % i]
        res += locals()['a%s' % i] * np.exp(locals()['b%s' % i])
        # else:
        #     res += locals()['a%s' % i] * np.exp(locals()['b%s' % i]) + locals()['c%s' % i]
    return res
    # return a * np.exp(-b * x) + c * x + d + 1e-4


# define the data to be fit with some noise
# xdata = np.linspace(0, 4, 50)
data = pd.read_csv('f:/dd.csv', delimiter='\t').values
xdata = data[:, 0]
# y = func(xdata, 2.5, 1.3, 0.5)
ydata = data[:, 1]
ydata = ydata / 60000.
# y_noise = 0.2 * np.random.normal(size=xdata.size)
# ydata = y + y_noise
plt.plot(xdata, ydata, 'b-', label='data')
# plt.show()

# Fit for the parameters a, b, c of the function `func`
popt, pcov = curve_fit(func, xdata, ydata)
plt.plot(xdata, func(xdata, *popt), 'r-', label='fit')

# Constrain the optimization to the region of ``0 < a < 3``, ``0 < b < 2``
# and ``0 < c < 1``:
# popt, pcov = curve_fit(func, xdata, ydata, bounds=(-10, -10))
# plt.plot(xdata, func(xdata, *popt), 'g--', label='fit-with-bounds')

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
