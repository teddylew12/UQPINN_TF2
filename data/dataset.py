import numpy as np
import tensorflow as tf

class DatasetLoader():
    def __init__(self, L, sigma):
        self.LENGTH = L
        self.SIGMA = sigma
    def generate_test_data(self, num_test_samples=1000, num_test_pts=100):
        #Generate randomness
        theta = np.random.beta(1, 1, size=(num_test_samples, 1))
        theta = tf.cast(theta, dtype=tf.float32)
        X_t = self.generate_input(num_test_pts)
        u_test = self._gen_U_samples(X_t,num_test_samples,theta)
        e_test = self._gen_E_samples(X_t,num_test_samples,theta)
        return X_t, u_test, e_test

    def generate_train_data(self, N_u,N_f, N_s):
        #Generate Collocation Points
        X_f = np.linspace(0.,self.LENGTH, N_f,dtype=np.float32)[:, None]
        #Generate U locations
        X_u  = np.linspace(0., self.LENGTH, N_u,dtype=np.float32)[:, None]
        #Generate Dataset
        theta = np.random.beta(1, 1, size=(N_s, 1))
        theta = tf.cast(theta, dtype=tf.float32)
        snapshots = self._gen_U_samples(X_u,N_s,theta)

        return X_u,X_f, snapshots

    def generate_input(self, num_sensors):
        '''
        Creates an (num_sensors,1) tensor linearly spaced between 0 and Length
        '''
        X = np.linspace(start=0.0, stop=self.LENGTH, num=num_sensors)
        X = tf.reshape(X, [num_sensors, -1])
        X = tf.cast(X, dtype=tf.float32)
        return X

    def _gen_U_samples(self,locations, num_samples,theta):
        '''
        Create snapshots given locations and some randomness through
        analytical solution
        '''
        locations = tf.tile(locations, [num_samples, 1])
        locations = tf.reshape(locations,[num_samples, -1])

        a = 3 * np.pi / self.LENGTH
        b = np.pi / 2 * (theta - 0.5)
        c = 0.5
        U = self.SIGMA * (
                1.2 * locations - c * tf.cos(a * locations + b) / a + c *
                tf.cos(b) / a)
        return tf.cast(U, dtype=tf.float32)

    def _gen_E_samples(self,locations,num_samples,theta):
        '''
        Create snapshots given locations and some randomness through
        analytical solution
        '''
        locations = tf.tile(locations, [num_samples, 1])
        locations = tf.reshape(locations, [num_samples, -1])
        a = 3 * np.pi / self.LENGTH
        b = np.pi / 2 * (theta - 0.5)
        c = 0.5
        E = 1 / (1.2 + c * tf.sin(a * locations + b))

        return tf.cast(E, dtype=tf.float32)

