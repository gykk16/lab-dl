'''
f(x, y, z) = (x + y) * z
x = -2, y = 5, z = -4에서의 df/dx, df/dy, df/dz의 값을
ex01에서 구현한 MultiplyLayer와 AddLayer 클래스를 이용해서 구하세요.
numerical_gradient 함수에서 계산된 결과와 비교

'''
import numpy as np

from ch04.e05_미분 import numerical_gradient
from ch05.e01_basic_layer import AddLayer, MultiplyLayer




x = -2
y = 5
z = -4

t = x + y
add_gate = AddLayer()
q = add_gate.forward(x, y)

mul_gate = MultiplyLayer()
f = mul_gate.forward(q, z)

print('f =', f)

d_q, d_z = mul_gate.backward(1)
d_x, d_y = add_gate.backward(d_q)

print('df/dx =', d_x)
print('df/dy =', d_y)
print('df/dz =', d_z)


def f(x, y, z):
    return (x + y) * z


h = 1e-12
dx = (f(-2 + h, 5, -4) - f(-2 - h, 5, -4)) / (2 * h)
print('df/dx =', dx)
dy = (f(-2, 5 + h, -4) - f(-2, 5 - h, -4)) / (2 * h)
print('df/dy =', dy)
dz = (f(-2, 5, -4 + h) - f(-2, 5, -4 - h)) / (2 * h)
print('df/dz =', dz)




