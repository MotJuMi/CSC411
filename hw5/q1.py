'''
Question 1 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt
from scipy.special import logsumexp

def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    '''
    n_classes, dim = len(np.unique(train_labels)), train_data.shape[1]
    means = np.zeros((n_classes, dim))
    # Compute means
    for k in range(n_classes):
        class_mask = train_labels == k
        means[k] = np.sum(train_data[class_mask], axis=0) / \
                   np.sum(class_mask, axis=0)
    return means

def compute_sigma_mles(train_data, train_labels):
    '''
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class 
    '''
    n_classes, dim = len(np.unique(train_labels)), train_data.shape[1]
    covariances = np.zeros((n_classes, dim, dim))
    # Compute covariances
    means = compute_mean_mles(train_data, train_labels)
    eps = 0.01
    for k in range(n_classes):
        class_mask = train_labels == k
        A = train_data[class_mask] - means[k]
        A = A.T @ A
        covariances[k] = (A + eps * np.eye(A.shape[0])) / class_mask.sum()
    return covariances

def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array 
    '''
    n_classes, dim = means.shape
    n_samples = len(digits)
    log_p = np.zeros((n_classes, n_samples))
    for k in range(n_classes):
        cov_det = np.linalg.det(covariances[k])
        x_mu = digits - means[k]
        power = -0.5 * ((x_mu @ np.linalg.inv(covariances[k])) * x_mu).sum(axis=1)
        log_p[k] = -0.5 * dim * np.log(2 * np.pi) - 0.5 * np.log(cov_det) + power
    return log_p.T

def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    n_classes = len(covariances)
    logp_y = -np.log(n_classes)
    loglikelihood = generative_likelihood(digits, means, covariances)
    numerator = loglikelihood + logp_y
    denominator = logsumexp(numerator, axis=1).reshape(-1, 1)

    return numerator - denominator

def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    avg_cond_likelihood = np.mean(np.choose(
                                  labels.astype(np.int), cond_likelihood.T))

    # Compute as described above and return
    return avg_cond_likelihood

def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    # Compute and return the most likely class
    predictions = np.argmax(cond_likelihood, axis=1)
    return predictions

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)

    # Evaluation
    avg_ll_train = avg_conditional_likelihood(train_data, train_labels, 
                                              means, covariances)
    avg_ll_test = avg_conditional_likelihood(test_data, test_labels, 
                                             means, covariances)
    print(f"Average train log-ll: {avg_ll_train}")
    print(f"Average train log-ll: {avg_ll_test}")

    train_predictions = classify_data(train_data, means, covariances)
    test_predictions = classify_data(test_data, means, covariances)
    train_acc = (train_predictions == train_labels.astype(np.int)).sum() / \
                len(train_labels)
    test_acc = (test_predictions == test_labels.astype(np.int)).sum() / \
               len(test_labels)
    print(f"Train acc: {train_acc}")
    print(f"Test acc: {test_acc}")

if __name__ == '__main__':
    main()
