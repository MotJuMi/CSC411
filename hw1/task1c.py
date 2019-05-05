import numpy as np

def distances(n=100000, d=10):
    X = np.array([np.random.uniform(size=n) for i in range(d)])
    Y = np.array([np.random.uniform(size=n) for j in range(d)])
    Z = (X - Y) ** 2
    R = np.sqrt(np.sum(Z, axis=0))
    R_sd = np.std(R)
    R_mean = np.mean(R)
    print(f"d = {d}, sqrt(d) = {np.sqrt(d):.3f}, [{R_mean - R_sd:.3f}, {R_mean + R_sd:.3f}], R_max: {np.max(R):.3f}")

def analytic_distances(d=10):
    E = 1/6 * d
    Var = 7/180 * d
    SD = np.sqrt(Var)
    print(f"d = {d} [{E - SD:.3f}, {E + SD:.3f}]")

def main():
    low, high = 1, 6
    dims = []
    for i in range(low - 1, high):
        for k in range(5):
            dims.append((2*k+1) * 10 ** i)
    for d in dims:
        distances(d=d)

if __name__ == "__main__":
    main()