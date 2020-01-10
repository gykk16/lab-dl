import numpy as np

np.random.seed(103)
X = np.random.randint(10, size = (2, 3))
print('x =', X)  # X : (2, 3)
W = np.random.randint(10, size = (3, 5))
print('W =', W)  # W : (3, 5)

# forward propagation
Z = X.dot(W)  # Z : (2, 5)
print('Z =', Z)

# back propagation
delta = np.random.randn(2, 5)
print('delta =', delta)

# X 방향으로의 오차 역전파
dX = delta.dot(W.T)  # (2, 5) @ (3, 5)T = (2, 5) @ (5, 3) = (2, 3) , X 의 모양
print('dX =', dX)

# W 방향으로의 오차 역전파
dW = X.T.dot(delta)  # (2, 3)T @ (2, 5) = (3, 2) @ (2, 5) = (3, 5) , W 의 모양
print('dW =', dW)
    