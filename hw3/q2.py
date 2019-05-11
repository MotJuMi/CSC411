# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 20:39:09 2017

"""
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
np.random.seed(0)

# load boston housing prices dataset
boston = load_boston()
x = boston['data']
N = x.shape[0]
x = np.concatenate((np.ones((506,1)),x),axis=1) #add constant one feature - no bias needed
d = x.shape[1]
y = boston['target']

idx = np.random.permutation(range(N))

#helper function
def l2(A,B):
    '''
    Input: A is a Nxd matrix
           B is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    '''
    A_norm = (A**2).sum(axis=1).reshape(A.shape[0],1)
    B_norm = (B**2).sum(axis=1).reshape(1,B.shape[0])
    dist = A_norm+B_norm-2*A.dot(B.transpose())
    return dist

def weighting(X, X_test, tau):
    dists = np.linalg.norm(X - X_test, axis=1)
    weights = -np.power(dists, 2) / (2 * tau ** 2)
    weights -= np.max(weights)
    weights_exp = np.exp(weights)
    weights_exp_sum = np.sum(weights_exp)
    weights_final = weights_exp / weights_exp_sum
    
    return weights_final

def weighting_l2(X, X_test, tau):
    dists = l2(X, X_test)
    weights = -np.power(dists, 2) / (2 * tau ** 2)
    weights -= np.max(weights)
    weights_exp = np.exp(weights)
    weights_exp_sum = np.sum(weights_exp)
    weights_final = weights_exp / weights_exp_sum
    
    return weights_final
 
def loss(X_train, y_true, X_test, w, tau, lam=1e-5):
    a = weighting(X_train, X_test, tau)
    X_test = np.concatenate([X_test, np.ones((len(X_test), 1))], axis=1)
    l = 0.5 * np.sum(a * np.power((y_true - X_test @ w), 2) + 0.5 * lam * np.power(np.linalg.norm(w), 2))
    return l / len(X_test)

def loss(y_true, y_pred):
    l = 0.5 * np.sum((y_true - y_pred) ** 2)
    return l / len(y_true)

#to implement
def LRLS(test_datum,x_train,y_train, tau,lam=1e-5):
    '''
    Input: test_datum is a dx1 test vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter
           lam is the regularization parameter
    output is y_hat the prediction on test_datum
    '''
    ## TODO
    n = x_train.shape[0]
    X_train = np.concatenate([x_train, np.ones((n, 1))], axis=1)
    m = X_train.shape[1]
    X_test = np.concatenate([test_datum.T, np.ones((1, 1))], axis=1)
    a = weighting(X_train, X_test, tau)
    A = np.diag(a)
    w = np.linalg.solve((X_train.T @ A @ X_train + lam * np.eye(m)), (X_train.T @ A @ y_train))
    y_test = X_test @ w
    return y_test

def run_validation(x,y,taus,val_frac):
    '''
    Input: x is the N x d design matrix
           y is the N x 1 targets vector    
           taus is a vector of tau values to evaluate
           val_frac is the fraction of examples to use as validation data
    output is
           a vector of training losses, one for each tau value
           a vector of validation losses, one for each tau value
    '''
    ## TODO
    n = x.shape[0]
    indices = np.random.permutation(n)
    train_idx, test_idx = indices[:int(n * (1 - val_frac))], indices[int(n * (1 - val_frac)):]
    X_train, X_test = x[train_idx, :], x[test_idx, :]
    y_train, y_test = y[train_idx], y[test_idx]
    train_losses, test_losses = [], []
    for tau in taus:
        current_loss = 0
        for i, test_datum in enumerate(X_test[:]):
            test_datum = test_datum.reshape(1, -1)
            y_pred = LRLS(test_datum.T, X_train, y_train, tau)
            current_loss += loss([y_test[i]], y_pred)
        test_loss = current_loss / len(test_idx)
        test_losses.append(test_loss)
        current_loss = 0
        for i, test_datum in enumerate(X_train[:]):
            test_datum = test_datum.reshape(1, -1)
            y_pred = LRLS(test_datum.T, X_train, y_train, tau)
            current_loss += loss([y_train[i]], y_pred)
        train_loss = current_loss / len(train_idx)
        train_losses.append(train_loss)
        print(f"tau = {tau}, Test loss {test_loss}, Train loss {train_loss}") 
    ## TODO
    return train_losses, test_losses


if __name__ == "__main__":
    # In this excersice we fixed lambda (hard coded to 1e-5) and only set tau value. Feel free to play with lambda as well if you wish
    taus = [1e-10, 1e-5, 1e-1, 1e2, 1e3, 1e10]
    n, m = 1000, 10
    x = np.random.randn(n, m)
    w_true = np.random.randn(x.shape[1] + 1,)
    y = np.concatenate([x, np.ones((n,1))], axis=1) @ w_true
    train_losses, test_losses = run_validation(x,y,taus,val_frac=0.3)
    plt.semilogx(taus, train_losses, label="train")
    plt.semilogx(taus, test_losses, label="test")
    plt.legend()
    plt.show()
