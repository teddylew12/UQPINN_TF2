"""
Created on Wed Nov 2018

@author:Theodore Lewitt
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from models import ODE_UQPINN
import tensorflow as tf
import scipy.io
from scipy import stats

np.random.seed(1234)
tf.random.set_seed(1234)

if __name__ == "__main__":

    # Number of collocation points
    N_f = 100

    # Number of testing points
    N_ff = 200

    # Number of the training data (in this example on the boundary)
    N_u = 20

    # Define the input, output, latent variable dimension
    X_dim = 1
    Y_dim = 1
    Z_dim = 1

    # Right handside of the ODE
    def f(X):
        return np.sin(np.pi * X)


    # Position of the collocation points
    X_f = np.linspace(-1., 1., N_f)[:, None]

    # Position of the boundary of the problem
    X_ut = np.linspace(-1., 1., 2)[:, None]
    X_u = X_ut
    for i in range(N_u - 1):
        X_u = np.vstack((X_u, X_ut))

    # Generate stochastic boundary condition
    Y_ut = f(X_ut)
    Y_u = Y_ut + 0.05 * np.random.randn(2, Y_dim)
    for i in range(N_u - 1):
        Y_ut = 0.05 * np.random.randn(2, Y_dim)
        Y_u = np.vstack((Y_u, Y_ut))

    # Model creation
    model = ODE_UQPINN(lam=1.5, beta=1.)

    # Train the model
    model.train(30000, 2 * N_u,N_f, X_f, X_u, Y_u)
    # Reference solution on the testing points
    X_ff = np.linspace(-1., 1., N_ff)[:, None]
    Y_reff = f(X_ff)

    # Load the reference solution of the stochastic ODE generated by Monte Carlo
    data = scipy.io.loadmat('data/ODE2000.mat')
    Exact = np.real(data['U']).T

    # Prediction
    plt.figure(1)
    N_samples = 2000
    samples_mean = np.zeros((X_ff.shape[0], N_samples))
    for i in range(0, N_samples):
        samples_mean[:, i:i + 1] = model.generate_sample(X_ff)
        plt.plot(X_ff, samples_mean[:, i:i + 1], 'k.', alpha=0.005)
    plt.plot(X_ff, Y_reff, 'r*', alpha=0.2)

    # Compute the mean and the variance of the prediction 
    mu_pred = np.mean(samples_mean, axis=1)
    Sigma_pred = np.var(samples_mean, axis=1)

    # Plot the prediction with the uncertainty versus the reference solution
    ax = plt.figure(2, figsize=(7, 5))
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.plot(X_u, Y_u, 'kx', markersize=4, label="Boundary points")
    lower = mu_pred - 2.0 * np.sqrt(Sigma_pred)
    upper = mu_pred + 2.0 * np.sqrt(Sigma_pred)
    plt.fill_between(X_ff.flatten(), lower.flatten(), upper.flatten(),
                     facecolor='orange', alpha=0.5, label="Two std band")
    plt.plot(X_ff, Y_reff, 'b-', label="Exact", linewidth=2)
    plt.plot(X_ff, mu_pred, 'r--', label="Prediction", linewidth=2)
    plt.xlabel('$x$', fontsize=13)
    plt.ylabel('$u(x)$', fontsize=13)
    plt.legend(loc='upper left', frameon=False, prop={'size': 10})
    plt.savefig('./ODEnew1.png', dpi=600)

    # Compute the prediction relative error
    mu_pred = mu_pred[:, None]
    error_u = np.linalg.norm(Y_reff - mu_pred, 2) / np.linalg.norm(Y_reff, 2)
    print('Error u: %e' % (error_u))
    np.save('L2_error.npy', error_u)

    ######### Compare the uncertainty at x = -0.5 and x = 0.5 ########
    E1 = Exact[50, :][:, None]
    E3 = Exact[150, :][:, None]
    M1 = samples_mean[50, :][:, None]
    M3 = samples_mean[150, :][:, None]

    ######## Probability density kernel estimation ########
    xmin, xmax = E1.min(), E1.max()
    X_marginal_1 = np.linspace(xmin, xmax, 100)[:, None]
    positions_marginal_1 = X_marginal_1.flatten()
    values_marginal_1 = E1.flatten()
    gkde = stats.gaussian_kde(values_marginal_1)
    KDE_marginal_1 = gkde.evaluate(positions_marginal_1)

    xmin, xmax = E3.min(), E3.max()
    X_marginal_3 = np.linspace(xmin, xmax, 100)[:, None]
    positions_marginal_3 = X_marginal_3.flatten()
    values_marginal_3 = E3.flatten()
    gkde = stats.gaussian_kde(values_marginal_3)
    KDE_marginal_3 = gkde.evaluate(positions_marginal_3)

    xmin, xmax = M1.min(), M1.max()
    X_marginal_4 = np.linspace(xmin, xmax, 100)[:, None]
    positions_marginal_4 = X_marginal_4.flatten()
    values_marginal_4 = M1.flatten()
    gkde = stats.gaussian_kde(values_marginal_4)
    KDE_marginal_4 = gkde.evaluate(positions_marginal_4)

    xmin, xmax = M3.min(), M3.max()
    X_marginal_6 = np.linspace(xmin, xmax, 100)[:, None]
    positions_marginal_6 = X_marginal_6.flatten()
    values_marginal_6 = M3.flatten()
    gkde = stats.gaussian_kde(values_marginal_6)
    KDE_marginal_6 = gkde.evaluate(positions_marginal_6)

    ax = plt.figure(3, figsize=(6, 4.7))
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.hist(Exact[50, :], bins=50, density=True, alpha=0.6, color='blue')
    plt.hist(samples_mean[50, :], bins=50, density=True, alpha=0.6, color='red')
    plt.plot(X_marginal_1, KDE_marginal_1, 'b-', label='Exact')
    plt.plot(X_marginal_4, KDE_marginal_4, 'r-', label='Prediction')
    plt.xlabel('$u(x = - 0.5)$', fontsize=13)
    plt.ylabel('$p(u)$', fontsize=13)
    plt.legend(loc='upper left', frameon=False, prop={'size': 13})
    plt.savefig('./ODE_x50.png', dpi=600)

    ax = plt.figure(5, figsize=(6, 4.7))
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.hist(Exact[150, :], bins=50, density=True, alpha=0.6, color='blue')
    plt.hist(samples_mean[150, :], bins=50, density=True, alpha=0.6,
             color='red')
    plt.plot(X_marginal_3, KDE_marginal_3, 'b-', label='Exact')
    plt.plot(X_marginal_6, KDE_marginal_6, 'r-', label='Prediction')
    plt.xlabel('$u(x = 0.5)$', fontsize=13)
    plt.ylabel('$p(u)$', fontsize=13)
    plt.legend(loc='upper left', frameon=False, prop={'size': 13})
    plt.savefig('./ODE_x150.png', dpi=600)
