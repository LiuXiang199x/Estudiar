import numpy as np

a = np.zeros((2,2))
b = []
for i in a:
	b.extend(i)
	print(i)
print(b)
