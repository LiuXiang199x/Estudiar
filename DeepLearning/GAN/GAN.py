import numpy as np
# TEST


a = np.random.uniform(1, 2, size=batch_size)[:, np.newaxis]
base = np.linspace(-1, 1, data_dim)[np.newaxis, :].repeat(batch_size, axis=0)
y = a * np.power(base, 2) + (a-1)