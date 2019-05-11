import numpy as np
import matplotlib.pyplot as plt

def huber_loss(y, t=0, delta=0.5):
    a = y - t
    if np.abs(a) <= delta:
        loss = 0.5 * a ** 2
    else:
        loss = delta * (np.abs(a) - 0.5 * delta)
    return loss

def mse_loss(y, t=0):
    loss = (y - t) ** 2
    return loss

def main():
    ys = np.linspace(-100, 100, 100)
    ms = list(map(mse_loss, ys))
    for delta in [0.5, 1, 10, 100]:
        hs = list(map(lambda y: huber_loss(y, delta=delta), ys))
        plt.plot(ys, hs, label=f"huber, delta={delta}")
    plt.plot(ys, ms, label="mse")
    plt.legend(loc="best")
    plt.show()

if __name__ == "__main__":
    main()