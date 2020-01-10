import numpy as np

a = np.array([[1, 2, 3],
              [2, 3, 4]])
b = np.array([[1, 2, 3],
              [2, 3, 4]])

print(a)
print(b)

print()

r = np.array([])
for i, j in zip(a, b):
    r = np.array_equal(i, j)
print(r)