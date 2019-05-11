import numpy as np
from scipy.optimize import check_grad

def huber_loss(y, t, delta=0.5):
    a = y - t
    loss = np.zeros_like(a)
    left = np.where(np.abs(a) <= delta)
    right = np.where(np.abs(a) > delta)
    loss[left] = 0.5 * np.power(a[left], 2)
    loss[right] = delta * (np.abs(a[right]) - 0.5 * delta)
    return np.mean(loss)

def huber_loss_full(X, y, w, delta=0.5):
    pred = X @ w
    return huber_loss(pred, y, delta)

def huber_loss_grad(X, y, w, delta=0.5):
    pred = X @ w
    a = pred - y
    #inlier_mask = np.where(np.abs(a) <= delta)
    #outlier_mask = np.where(np.abs(a) > delta)
    outlier_mask = np.abs(a) > delta
    gradw_inlier = X[~outlier_mask].T @ a[~outlier_mask]
    gradw_outlier = delta * X[outlier_mask].T @ np.sign(a[outlier_mask])
    gradw = gradw_inlier + gradw_outlier
    return gradw / X.shape[0]

def gradient_descent(X, y, lr, num_iter, delta):
    X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    w = np.random.uniform(size=(X.shape[1]))
    current_iter = 0
    while current_iter < num_iter:
        current_iter += 1
        gradw = huber_loss_grad(X, y, w, delta)
        if current_iter % 100 == 1:
            print(f"Iter [{current_iter}], Loss [{huber_loss_full(X, y, w, delta)}]")
            #print(f"Weight {w}")
        w = w - lr * gradw
    return w[:-1], w[-1]

def main():
    n, m = 100, 10
    X = np.random.randn(n, m)
    true_w = np.random.randn(m,)
    true_b = np.random.randn(1)
    y = X @ true_w + true_b
    delta = 0.5
    lr = 1
    num_iter  = 10000
    w, b = gradient_descent(X, y, lr, num_iter, delta)
    print(f"Weight difference: {true_w - w} \nIntercept difference: {true_b - b}")
    # print(check_grad(
    #     lambda w: huber_loss_full(X, y, w, delta), 
    #     lambda w: huber_loss_grad(X, y, w, delta), np.random.uniform(size=(X.shape[1]))))

if __name__ == "__main__":
    main()