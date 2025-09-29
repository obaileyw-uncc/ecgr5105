import random as rd
import numpy as np
import matplotlib.pyplot as plt

def compute_loss(X, y, theta):
    H = X.dot(theta)
    sq_err = np.square(np.subtract(H, y))
    J = (1 / (2 * len(X))) * np.sum(sq_err)
    return J

def grad_desc(X, y, X_val, y_val, theta, alpha, N):
    m = len(y)
    train_loss_history = np.zeros(N)
    val_loss_history = np.zeros(N)

    for i in range(N):
        H = X.dot(theta)
        err = np.subtract(H, y)
        inc = (alpha / m) * X.transpose().dot(err)
        theta = np.subtract(theta, inc)
        train_loss_history[i] = compute_loss(X, y, theta)
        val_loss_history[i] = compute_loss(X_val, y_val, theta)

    return theta, train_loss_history, val_loss_history

def compute_loss_regularized(X, y, theta, lamb):
  H = X.dot(theta)
  sq_err = np.square(np.subtract(H, y))
  penalty = lamb * np.sum(np.square(theta[1:]))
  J = (1 / (2 * len(X))) * (np.sum(sq_err) + penalty)
  return J

def grad_desc_regularized(X, y, X_val, y_val, theta, alpha, lamb, N):
    m = len(y)
    train_loss_history = np.zeros(N)
    val_loss_history = np.zeros(N)

    for i in range(N):
        H = X.dot(theta)
        err = np.subtract(H, y)
        reg = theta * (1 - alpha * (lamb/m))
        inc = (alpha / m) * X.transpose().dot(err)
        theta = np.subtract(reg, inc)
        train_loss_history[i] = compute_loss_regularized(X, y, theta, lamb)
        val_loss_history[i] = compute_loss(X_val, y_val, theta)

    return theta, train_loss_history, val_loss_history

# split training and validations sets 80-20
def split_sets_80_20(X: np.ndarray, Y: np.ndarray):
    Y_val = []
    X_val = []

    if (len(Y) != len(X)):
        raise ValueError("Input and output vectors must be the same length")

    val_indices = rd.sample(range(len(X)), int(len(X)/5))
    for i in val_indices:
        Y_val = np.append(Y_val, Y[i])
        if len(X_val) == 0:
            X_val = X[i,:]
        else:
            X_val = np.vstack((X_val, X[i,:]))
    Y_train = np.delete(Y, val_indices)
    X_train = np.delete(X, val_indices, 0)

    return X_train, X_val, Y_train, Y_val

def plot_convergence(theta, train_loss_history, val_loss_history):
    print("Final value of parameters: {}".format(theta))
    print("Training loss history: {}".format(train_loss_history))
    print("Validation loss history: {}".format(val_loss_history))

    # plot the loss over time
    plt.plot(range(1, len(train_loss_history) + 1), train_loss_history, color='blue')
    plt.plot(range(1, len(train_loss_history) + 1), val_loss_history, color='orange')
    plt.grid(True)
    plt.legend(['Training', 'Validation'])
    plt.xlabel('Number of iterations')
    plt.ylabel('Loss (J)')
    plt.title('Convergence of gradient descent (X)')
    plt.show()

'''
Convert an input vector to a minimum-maximum normalized input vector with
values between 0 and 1
'''
def input_min_max_norm(X: np.ndarray) -> np.ndarray:
    (_, col) = X.shape
    for i in range(1, col):
        max = X[:,i].max()
        min = X[:,i].min()
        X[:,i] = np.divide(np.subtract(X[:,i], min), (max - min))
    return X

'''
Convert an input vector to a standarized input vector with values representing
standard deviations above or below the mean
'''
def input_standardize(X: np.ndarray) -> np.ndarray:
    (_, col) = X.shape
    for i in range(1, col):
        mu = np.sum(X[:,i]) / len(X[:,i])
        sigma = np.sqrt(np.divide(np.sum(np.square(np.subtract(X[:,i], mu))), len(X[:,i])))
        X[:,i] = np.divide(np.subtract(X[:,i], mu), sigma)
    return X
