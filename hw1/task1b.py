import numpy as np

d = 1000
n = 10000
X = np.array([np.random.uniform(size=n) for i in range(d)])
Y = np.array([np.random.uniform(size=n) for j in range(d)])
Z = (X - Y) ** 2
Z0 = Z[0]
R = np.sum(Z, axis=0)
print(R.shape)
print(np.max(R))
print(f"E(Z) = {np.mean(Z0)}")
print(f"Var(Z) = {np.var(Z0)}")
print(f"E(R) = {np.mean(R)}")
print(f"SD(R) = {np.std(R)}")
print(f"Var(R) = {np.var(R)}")
print(f"E(R)/E(Z) = {np.mean(R) / np.mean(Z0)}")
print(f"Var(R)/Var(Z) = {np.var(R)/np.var(Z0)}")