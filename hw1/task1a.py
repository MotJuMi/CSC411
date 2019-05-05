import numpy as np

n = 100000
X = np.random.uniform(size=n)
Y = np.random.uniform(size=n)
Z = (X - Y) ** 2

print(f"E(Z) = {np.mean(X ** 2) + np.mean(Y ** 2) - 2 * np.mean(X * Y):.3f}")
print(f"E(XY) = {np.mean(X*Y):.3f}")
print(f"E(X)E(Y) = {np.mean(X) * np.mean(Y):.3f}")
print(f"E(Z) = {np.mean(Z):.3f}")
print(f"Var(Z) = {np.var(Z):.3f}")