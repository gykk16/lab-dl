import numpy as np
import matplotlib.pyplot as plt

from ch04.e05_미분 import numerical_gradient


def fn(x):
    ''' x = [x0, x1] '''

    if x.ndim == 1:
        return np.sum(x ** 2)
    else:
        return np.sum(x ** 2, axis = 1)


x0 = np.arange(-2, 3)
print('x0 =', x0)
x1 = np.arange(-1, 2)
print('x1 =', x1)

X, Y = np.meshgrid(x0, x1)
print('X =', X)
print('Y =', Y)

X = X.flatten()
Y = Y.flatten()
print('X =', X)
print('Y =', Y)

XY = np.array([X, Y])
print('XY =', XY)

gradients = numerical_gradient(fn, XY)
print(gradients)

x0 = np.arange(-2, 2.5, 0.25)
x1 = np.arange(-2, 2.5, 0.25)
X, Y = np.meshgrid(x0, x1)
X = X.flatten()
Y = Y.flatten()
XY = np.array([X, Y])
gradients = numerical_gradient(fn, XY)
print('gradients =', gradients)

plt.quiver(X, Y, -gradients[0], -gradients[1], angles = 'xy')
plt.xlim([-2, 2])
plt.ylim([-2, 2])
plt.xlabel('x0')
plt.ylabel('x1')
plt.show()




