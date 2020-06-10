import tensorflow as tf
from tensorflow.keras import layers


class NeuralNetwork:
    def __init__(self, input_shape=1, layers=[], activation="",
                 dimensionality=1,
                 name=""):
        '''
        Wrapper class around a Keras model that validates the model creation
        :param config_dict: Dictionary for initialization of the model
        Has the following keys:
        -Input Shape: Integer
        -Layers: Sub-Dictionary (planning on adding more sub-keys if needed)
        -Layers/Units: Array with units for dense layers, number of dense layers
        corresponds to length of array
        -Activation: Keras activation used for each layer
        -Kernel Initializer: Sets initial random weight of layers
        -Bias Initializer: Sets initial random bias weights of layers
        -Dimensionality: Desired output dimensionality
        :param name: String for differentiation of one NN from another
        '''
        self.name = name
        self._input_shape = input_shape
        self._layers = layers
        self._activation = activation
        self._dimensionality = dimensionality
        self._validate_inputs()
        self._network = self._create_model()

    def _validate_inputs(self):
        for unit in self._layers:
            if type(unit) is not int:
                raise ValueError("Layers must be integer valued")
        if type(self._dimensionality) is not int:
            raise ValueError("Output layer units must be integer")
        if type(self._input_shape) is not int:
            raise ValueError("Input shape must be integer")

    def _create_model(self):
        model = tf.keras.Sequential()
        model.add(layers.Input(shape=self._input_shape))
        model.add(layers.Flatten())
        for units in self._layers:
            model.add(layers.Dense(units, activation=self._activation))
        model.add(layers.Dense(self._dimensionality))
        return model

    def get_dimensionality(self):
        return self._dimensionality

    def forward_pass(self, input_tensor, train=True):
        '''
        Wraps the keras models' forward pass to make prediction
        '''
        return self._network(input_tensor, training=train)

    def save(self, h5pyObj, save_path):
        h5pyObj.attrs["input_shape"] = self._input_shape
        h5pyObj.attrs["layers"] = self._layers
        h5pyObj.attrs["activation"] = self._activation
        h5pyObj.attrs["dimensionality"] = self._dimensionality
        save_name = self.name + "_network.h5"
        self._network.save(save_path.joinpath(save_name))

    def load(self, h5pyObj, file_path, name):
        self._input_shape = h5pyObj.attrs["input_shape"]
        self._layers = h5pyObj.attrs["layers"]
        self._activation = h5pyObj.attrs["activation"]
        self._dimensionality = h5pyObj.attrs["dimensionality"]
        model_path = file_path.joinpath(name + "_network.h5")
        self._network = tf.keras.models.load_model(model_path, compile=False)
        self.name = name
