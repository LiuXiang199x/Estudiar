import numpy as np

# print(np.random.rand(2, 5))
# print(np.random.randint(0, 10, (3,5)))
# print(np.random.randn(2, 2))

a = np.zeros((2,2))
b = np.ones((2,2))

a[:, 0] = b[:, 0]
print(a)