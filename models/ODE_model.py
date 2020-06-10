"""
Created on Wed Nov 2018

@author: Ted Lewitt
"""

import tensorflow as tf
import numpy as np

from pigan import NeuralNetwork
from collections import defaultdict
from tqdm import tqdm


class ODE_UQPINN:
    # Initialize the class
    def __init__(self, lam=1.0, beta=1.0):
        # Used for logging losses
        self.logger = defaultdict(list)
        # Hyperparameter for Encoder Weight
        self.lam = lam
        # Hyperparameter for PDE and BC Weight
        self.beta = beta
        # Discriminator iterations for each global step
        self.disc_iters = 1
        # Generator iterations for each global step
        self.gen_iters = 5
        # Number of samples used to check PDE and BC weights
        self.bc_samples = 100
        # Define optimizers
        self.gen_opt = tf.keras.optimizers.Adam(1e-4)
        self.disc_opt = tf.keras.optimizers.Adam(1e-4)
        # Define the input dimension
        X = 1
        # Output Dimension
        Y = 1
        # Noise Dimension
        Z = 1
        # Initialize all 4 neural networks
        self.P_network = NeuralNetwork(name="P",
                                       input_shape=X + Z,
                                       layers=[50, 50, 50, 50],
                                       activation="tanh",
                                       dimensionality=Y)
        self.Q_network = NeuralNetwork(name="Q",
                                       input_shape=X + Y,
                                       layers=[50, 50, 50, 50],
                                       activation="tanh",
                                       dimensionality=Z)
        self.T_network = NeuralNetwork(name="T",
                                       input_shape=X + Y,
                                       layers=[50, 50, 50, 50],
                                       activation="tanh",
                                       dimensionality=1)
        #Normalization constants
        self.jacobian = 0
        self.Xstd = 0
        self.Xmean = 0
        # Initialize network weights and biases

    def train(self, nIter, N_u, N_f, X_f, X_u, Y_u):
        #Make all data have the type tf.float32
        X_f = tf.cast(X_f, dtype=tf.float32)
        X_u = tf.cast(X_u, dtype=tf.float32)
        Y_u = tf.cast(Y_u, dtype=tf.float32)
        # Normalize data
        self.Xmean = tf.math.reduce_mean(X_f, axis=0)
        self.Xstd = tf.math.reduce_std(X_f, axis=0)
        self.Jacobian = 1 / self.Xstd
        X_u = (X_u - self.Xmean) / self.Xstd

        #Main training loop
        for _ in tqdm(range(nIter)):

            # Sampling from the latent space for data and collocation points
            Z_u = tf.random.normal((N_u, 1))
            Z_f = tf.random.normal((N_f, 1))

            for i in range(self.disc_iters):
                disc_loss_dict = self.discriminator_step(X_u, Y_u, Z_u)
                self.log_step(disc_loss_dict)
            for j in range(self.gen_iters):
                gen_loss_dict = self.generator_step(X_u, X_f, Z_u, Z_f)
                self.log_step(gen_loss_dict)

    @tf.function
    def generator_step(self, X_u, X_f, Z_u, Z_f):
        with tf.GradientTape(persistent=True) as gen_tape:
            gen_tape.watch(X_u)
            gen_output = self.net_P(X_u, Z_u)
            pde_output = self.get_r(X_f, Z_f, gen_tape)
            encoder_output = self.net_Q(X_u, gen_output)
            disc_output = self.net_T(X_u, gen_output)
            G_loss = self.compute_generator_loss(pde_output, encoder_output,
                                                 disc_output, Z_u)

        grads = gen_tape.gradient(G_loss["total_gen_loss"],
                                  [self.P_network._network.trainable_variables,
                                   self.Q_network._network.trainable_variables])
        self.gen_opt.apply_gradients(
            zip(grads[0], self.P_network._network.trainable_variables))
        self.gen_opt.apply_gradients(
            zip(grads[1], self.Q_network._network.trainable_variables))
        del gen_tape
        return G_loss

    def compute_generator_loss(self, pde_out, encode_out, disc_out, noise_in):

        # KL-divergence between the data distribution and the model distribution
        KL = tf.reduce_mean(disc_out)

        # Entropic regularization
        log_q = - tf.reduce_mean(tf.square(noise_in - encode_out))

        # Physics-informed loss
        loss_f = tf.reduce_mean(tf.square(pde_out))

        # Generator loss
        total_loss = KL + (1.0 - self.lam) * log_q + self.beta * loss_f
        losses = {}
        losses["KL"] = KL
        losses["Log_Q"] = log_q
        losses["loss_f"] = loss_f
        losses["total_gen_loss"] = total_loss
        return losses

    @tf.function
    def discriminator_step(self, locations, true_output, noise):

        with tf.GradientTape() as disc_tape:
            disc_tape.watch(locations)
            gen_output = self.net_P(locations, noise)
            T_loss = self.compute_discriminator_loss(locations, gen_output,
                                                     true_output)

        grads = disc_tape.gradient(T_loss,
                                   self.T_network._network.trainable_variables)
        self.disc_opt.apply_gradients(
            zip(grads, self.T_network._network.trainable_variables))
        return T_loss

    # Compute the discriminator loss
    def compute_discriminator_loss(self, locations, gen_samples, true_samples):
        # Discriminator loss
        T_real = self.net_T(locations, true_samples)
        T_fake = self.net_T(locations, gen_samples)

        T_real = tf.sigmoid(T_real) - +1e-8
        T_fake = tf.sigmoid(T_fake) + 1e-8

        T_loss = -tf.reduce_mean(tf.math.log(1.0 - T_real) + tf.math.log(
            T_fake))
        losses = {}
        losses["Real_Loss"] = T_real
        losses["Gen_Loss"] = T_fake
        losses["total_gen_loss"] = T_loss
        return losses

    # Predict y given x
    # Forcing term (right hand of the ODE)
    def f(self, X_normalized):  #
        X = self.Xstd * X_normalized + self.Xmean
        return - np.pi ** 2 * tf.sin(np.pi * X) - np.pi * tf.cos(
            np.pi * X) * tf.sin(np.pi * X) ** 2

    #Generator: p(y|x,z)
    def net_P(self, X, Z):
        Y = self.P_network.forward_pass(tf.concat([X, Z], 1))
        return Y

    # Encoder: q(z|x,y)
    def net_Q(self, X, Y):
        Z = self.Q_network.forward_pass(tf.concat([X, Y], 1))
        return Z

    # Discriminator
    def net_T(self, X, Y):
        T = self.T_network.forward_pass(tf.concat([X, Y], 1))
        return T

    # Physics-Informed residual on the collocation points
    def get_r(self, X, Z, tape):
        z_prior = Z
        tape.watch(X)
        u = self.net_P(X, z_prior)
        u_x = tape.gradient(u, X)
        u_xx = tape.gradient(u_x, X)
        f = self.f(X)
        r = (self.Jacobian ** 2) * u_xx - (self.Jacobian) * (u ** 2) * u_x - f
        return r

    def generate_sample(self, X_star):
        X_star = (X_star - self.Xmean) / self.Xstd
        Z = np.random.randn(X_star.shape[0], 1)
        Y_pred = self.net_P(X_star, Z)
        return Y_pred

    # Get the posterior of z over the latent space
    def get_z(self, X, Z):
        Y_pred = self.net_P(X, Z)
        z = self.net_Q(X, Y_pred)
        return z

    def log_step(self, loss):
        for key, value in loss.items():
            self.logger[key].append(value.numpy())
