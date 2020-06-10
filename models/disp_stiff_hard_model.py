"""
@author: Ted Lewitt
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
import h5py

from neural_network import NeuralNetwork

class DISP_STIFF_UQPINN:
    # Initialize the class
    def __init__(self, run_name, b1, lam=1.0, beta=1.0):
        # Used for saving results
        self.run_name = run_name
        # Used for saving checkpoints
        self.ckpt_name = run_name / "ckpts/"
        self.ckpt_name.mkdir(parents=True, exist_ok=True)
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
        self.gen_opt = tf.keras.optimizers.Adam(beta_1=b1, lr=1e-4)
        self.disc_opt = tf.keras.optimizers.Adam(beta_1=b1, lr=1e-4)
        # Define the input dimension
        X = 1
        # Output Dimension
        Y = 1
        # Noise Dimension
        Z = 1
        # Initialize all 4 neural networks
        self.E_generator = NeuralNetwork(name="E",
                                         input_shape=X + Z,
                                         layers=[50, 50, 50, 50],
                                         activation="tanh",
                                         dimensionality=Y)
        self.U_generator = NeuralNetwork(name="U",
                                         input_shape=X + Z,
                                         layers=[50, 50, 50, 50],
                                         activation="tanh",
                                         dimensionality=Y)
        self.encoder = NeuralNetwork(name="Q",
                                     input_shape=X + Y,
                                     layers=[50, 50, 50, 50],
                                     activation="tanh",
                                     dimensionality=Z)
        self.critic = NeuralNetwork(name="T",
                                    input_shape=X + Y,
                                    layers=[50, 50, 50, 50],
                                    activation="tanh",
                                    dimensionality=1)

    def train(self, nIter, N_u, N_f, N_s, X_f, X_u, Y_u, X_t, save_freq):
        # Make all data have tf.float32 type and tile num_samples
        X_f = tf.cast(tf.tile(X_f, [N_s, 1]), dtype=tf.float32)
        X_u = tf.cast(tf.tile(X_u, [N_s, 1]), dtype=tf.float32)
        Y_u = tf.cast(Y_u, dtype=tf.float32)

        # Main training loop
        for step in tqdm(range(nIter)):

            # Sampling randomness for data and collocation points
            Z_u = tf.random.normal((N_u * N_s, 1))
            Z_f = tf.random.normal((N_f * N_s, 1))
            for i in range(self.disc_iters):
                disc_loss_dict = self.discriminator_step(X_u, Y_u, Z_u)
                # Save losses
                self.log_step(disc_loss_dict)
            for j in range(self.gen_iters):
                gen_loss_dict = self.generator_step(X_u, X_f, Z_u, Z_f)
                self.log_step(gen_loss_dict)
            # Create checkpoints to track training progress
            if (step + 1) % save_freq == 0 and (step + 1) != nIter:
                self.save_pigan_samples(X_t, step + 1, self.ckpt_name)

    @tf.function
    def generator_step(self, X_u, X_f, Z_u, Z_f):
        # Create gradient tape to get derivatives
        with tf.GradientTape(persistent=True) as gen_tape:
            gen_tape.watch(X_u)
            # Forward pass through U
            u_gen_output = self.gen_U(X_u, Z_u)
            # Takes U output for forward pass through Encoder
            encoder_output = self.net_Q(X_u, u_gen_output)
            # Takes U output for forward pass through Critic
            disc_output = self.net_T(X_u, u_gen_output)
            # Calculate Loss scalar
            G_loss = self.compute_generator_loss(X_f, Z_f, encoder_output,
                                                 disc_output, Z_u, gen_tape)
        # Autodiff the loss to get gradient update
        grads = gen_tape.gradient(G_loss["total_gen_loss"],
                                  [
                                      self.U_generator._network.trainable_variables,
                                      self.E_generator._network.trainable_variables,
                                      self.encoder._network.trainable_variables])
        # Apply the gradients to the weights using the optimizer
        self.gen_opt.apply_gradients(
            zip(grads[0], self.U_generator._network.trainable_variables))
        self.gen_opt.apply_gradients(
            zip(grads[1], self.E_generator._network.trainable_variables))
        self.gen_opt.apply_gradients(
            zip(grads[2], self.encoder._network.trainable_variables))
        # Delete the persistent tape
        del gen_tape
        # Return loss dictionary
        return G_loss

    def compute_generator_loss(self, X_f, Z_f, encode_out, disc_out, noise_in,
                               tape):
        # Get random noise for BC evaluations
        Z_boundary = tf.random.normal((self.bc_samples, 1))
        # Left boundary points and expected values at those points
        left_boundary = tf.zeros((self.bc_samples, 1))
        expected_lb = tf.zeros_like(left_boundary)
        # Right boundary points and expected values at those points
        right_boundary = tf.ones((self.bc_samples, 1))
        expected_rb = tf.fill(right_boundary.shape, 1.5)
        # Tell the gradient tap what to watch
        tape.watch(left_boundary)
        tape.watch(right_boundary)
        tape.watch(X_f)

        # Evaluate left Boundary and get MSE Loss
        lb_out = self.get_left_boundary(left_boundary, Z_boundary)
        loss_lb = tf.reduce_mean(tf.square(lb_out - expected_lb))
        # Evaluate right Boundary and get MSE Loss
        rb_out = self.get_right_boundary(right_boundary, Z_boundary, tape)
        loss_rb = tf.reduce_mean(tf.square(rb_out - expected_rb))
        # Evaluate PDE Loss
        loss_f = self.evaluate_pde(X_f, Z_f, tape)
        # Sum all Physics Losses
        physics_loss = loss_f + loss_lb + loss_rb
        # Get scalar loss from Critic Output
        KL = tf.reduce_mean(disc_out)

        # Get scalar loss from Encoder Output
        log_q = - tf.reduce_mean(tf.square(noise_in - encode_out))

        # Sum all losses for total generator loss
        total_loss = KL + (1.0 - self.lam) * log_q + self.beta * physics_loss
        # Create loss dictionary
        losses = {}
        losses["KL"] = KL
        losses["Log_Q"] = log_q
        losses["loss_f"] = loss_f
        losses["loss_lb"] = loss_lb
        losses["loss_rb"] = loss_rb
        losses["physics_loss"] = physics_loss
        losses["total_gen_loss"] = total_loss
        return losses

    @tf.function
    def discriminator_step(self, locations, true_output, noise):
        # Create gradient tape to get derivatives
        with tf.GradientTape() as disc_tape:
            # Tell gradient tape what to watch
            disc_tape.watch(locations)
            # Forward pass through U
            gen_output = self.gen_U(locations, noise)
            # Get loss scalar
            T_loss = self.compute_discriminator_loss(locations, gen_output,
                                                     true_output)
        # Autodiff the loss to get gradient update
        grads = disc_tape.gradient(T_loss,
                                   self.critic._network.trainable_variables)
        # Apply the gradients to the weights using the optimizer
        self.disc_opt.apply_gradients(
            zip(grads, self.critic._network.trainable_variables))
        return T_loss

    # Compute the discriminator loss
    def compute_discriminator_loss(self, locations, gen_samples, true_samples):
        # Make sure true samples and gen samples have same shape
        true_samples = tf.reshape(true_samples, (-1, 1))
        # Forward pass through Critic
        T_real = self.net_T(locations, true_samples)
        T_fake = self.net_T(locations, gen_samples)

        # Get scalar of losses (with 1e-8 for numerical stability)
        T_real_scalar = tf.reduce_mean(tf.sigmoid(T_real)) - 1e-8
        T_fake_scalar = tf.reduce_mean(tf.sigmoid(T_fake)) + 1e-8
        # Sum all losses for total loss
        T_loss = -tf.math.log(1.0 - T_real_scalar) + tf.math.log(T_fake_scalar)
        # Create dictionary of all losses
        losses = {}
        losses["Real_Loss"] = T_real_scalar
        losses["Gen_Loss"] = T_fake_scalar
        losses["total_gen_loss"] = T_loss
        return losses

    def initialize_nns(self, X, Y, Z):
        '''
        Use Pigan nueral network class
        :param X: Input dimension of generator
        :param Y: Ouput dimesnion of generator
        :param Z: Noise dimension
        '''

    def gen_U(self, X, Z):
        '''
        Forward pass of U
        :param X: locations
        :param Z: noise
        '''
        U = self.U_generator.forward_pass(tf.concat([X, Z], 1))
        return U

    def gen_E(self, X, Z):
        '''
        Forward pass of E
        :param X: locations
        :param Z: noise
        '''
        E = self.E_generator.forward_pass(tf.concat([X, Z], 1))
        return E

    def net_Q(self, X, Y):
        '''
        Forward pass of U
        :param X: locations
        :param Z: generator output
        '''
        Z = self.encoder.forward_pass(tf.concat([X, Y], 1))
        return Z

    def net_T(self, X, Y):
        '''
        Forward pass of U
        :param X: locations
        :param Z: generator output
        '''
        T = self.critic.forward_pass(tf.concat([X, Y], 1))
        return T

    def evaluate_pde(self, X, Z, tape):
        '''

        :param X: locations
        :param Z: noise
        :param tape:
        :return:
        '''
        tape.watch(X)
        # Forward pass through U and E
        u = self.gen_U(X, Z)
        e = self.gen_E(X, Z)

        # Get derivatives
        u_x = tape.gradient(u, X)
        u_xx = tape.gradient(u_x, X)
        e_x = tape.gradient(e, X)
        # Evalualte PDE
        residual = tf.multiply(u_xx, e) + tf.multiply(u_x, e_x)
        # MSE with 0 to get scalar loss
        residual_scalar = tf.reduce_mean(tf.square(residual))
        return residual_scalar

    def get_left_boundary(self, X, Z):
        # Evaulate dirichlet BC on left boundary
        return self.gen_U(X, Z)

    def get_right_boundary(self, locs, noise, tape):
        # Evaulate neumann BC on left boundary
        tape.watch(locs)
        # Forward pass on U and E
        e = self.gen_E(locs, noise)
        u = self.gen_U(locs, noise)
        # Get derivative
        du_dx = tape.gradient(u, locs)
        # Evaluate neumann equation
        return du_dx * e

    def generate_samples(self, X, N_ts):
        # Cast and tile locations
        X = tf.cast(tf.tile(X, [N_ts, 1]), dtype=tf.float32)
        # Sample random noise
        Z = np.random.randn(X.shape[0], 1)
        # Forward pass for U,E and Encoder
        U_pred = self.gen_U(X, Z)
        E_pred = self.gen_E(X, Z)
        Z_pred = self.net_Q(X, U_pred)
        # Reshape inputs
        U_pred = tf.reshape(U_pred, [N_ts, -1])
        E_pred = tf.reshape(E_pred, [N_ts, -1])
        Z_pred = tf.reshape(Z_pred, [N_ts, -1])
        return U_pred, E_pred, Z_pred

    def save_pigan_samples(self, test_locs, step, save_dir):
        U, E, Z = self.generate_samples(test_locs, 500)
        fname = "generated_" + str(step) + ".hdf5"
        path_name = save_dir / fname

        with h5py.File(path_name, 'w') as fileObj:
            fileObj.create_dataset('E', data=E.numpy())
            fileObj.create_dataset('U', data=U.numpy())
            fileObj.create_dataset('Z', data=Z.numpy())

    def log_step(self, loss):
        for key, value in loss.items():
            self.logger[key].append(value.numpy())

    def save_log(self, save_dir):
        save_path = save_dir / "logs/"
        save_path.mkdir(parents=True, exist_ok=True)
        file_path = save_path / "training_log.hdf5"
        with h5py.File(file_path, "w") as fileObj:
            losses = fileObj.create_group("losses")
            for key, value in self.logger.items():
                losses.create_dataset(key, data=np.array(value))
        fig = plt.figure()
        for idx, kv in enumerate(self.logger.items()):
            key, val = kv
            fig.add_subplot(3, 3, idx + 1)
            plt.plot(val)
            plt.title(key)
        fig_path = save_path / "training_log.png"
        plt.savefig(fig_path)

    def save(self, save_dir):
        self.save_log(save_dir)
        save_path = save_dir.joinpath("models/")
        save_path.mkdir(parents=True, exist_ok=True)
        settings_name = save_path.joinpath("settings.hdf5")
        with h5py.File(settings_name, 'w') as fileObj:
            u_gen = fileObj.create_group("U_gen")
            e_gen = fileObj.create_group("E_gen")
            disc = fileObj.create_group("Critic")
            enc = fileObj.create_group("Encoder")

            self.E_generator.save(e_gen, save_path)
            self.U_generator.save(u_gen, save_path)
            self.encoder.save(enc, save_path)
            self.critic.save(disc, save_path)

    def load(self, file_path):
        self.save_log(file_path)
        model_path = file_path / "models"
        settings = model_path / "settings.hdf5"
        with h5py.File(settings, 'r') as fileObj:
            self.E_generator = NeuralNetwork()
            self.E_generator.load(fileObj["E_gen"], model_path)
            self.U_generator = NeuralNetwork()
            self.U_generator.load(fileObj["U_gen"], model_path)
            self.encoder = NeuralNetwork()
            self.encoder.load(fileObj["Encoder"], model_path)
            self.critic = NeuralNetwork()
            self.critic.load(fileObj["Critic"], model_path)
