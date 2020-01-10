'''
행렬의 내적(dot product)
A, B, ... : 2차원 이상
x, y, ... : 1차원 ndarray

'''
import numpy as np

x = np.array([1, 2])
W = np.array([[3, 4],
              [5, 6]])

print(x.dot(W))

A = np.arange(1, 7).reshape((2, 3))
print(A)
B = np.arange(1, 7).reshape((3, 2))
print(B)
print(A.dot(B))  # 2 x 2
print(B.dot(A))  # 3 x 3
# 행렬의 내적(dot product)은 교환 법칙(AB = BA)가 성립하지 않는다

# ndarray.shape -> (x, ), (x, y), (x, y, z), ...
x = np.array([1, 2, 3])
print(x)
print(x.shape)  # (3,) 원소의 갯수

x = x.reshape((3, 1))
print(x)
print(x.shape)

x = x.reshape((1, 3))
print(x)
print(x.shape)
