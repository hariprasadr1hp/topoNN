"""
neural_net.py
"""

import numpy as np


class NeuralNet:
    """
    The Network architecture

    :type layerSizes: list
    :param layerSizes: contains the number of neurons in each layer
            For instance, for a network with 2 hidden layers with
            2 neurons and the input and output parameters of 3 and
            4 respectively, layerSize = [3,2,2,4]

    :type batchSize: int
    :param batchSize: the volume of data per batch
    """

    def __init__(self, layer_sizes, batch_size=100):
        self.layer_sizes = layer_sizes
        self.batch_size = batch_size
        self.num_layers = len(self.layer_sizes)
        self.generate_variables()

    def generate_variables(self):
        self.y = np.array(
            [np.zeros([self.layer_sizes[i], 1]) for i in range(self.num_layers)],
            dtype=object,
        )

        self.w = np.array(
            [
                np.random.uniform(
                    low=1, high=1, size=[self.layer_sizes[i], self.layer_sizes[i + 1]]
                )
                for i in range(self.num_layers - 1)
            ],
            dtype=object,
        )

        self.b = np.array(
            [
                np.random.uniform(low=1, high=1, size=[self.layer_sizes[i + 1], 1])
                for i in range(self.num_layers - 1)
            ],
            dtype=object,
        )

        self.df = np.array(
            [np.zeros(self.layer_sizes[i + 1]) for i in range(self.num_layers - 1)],
            dtype=object,
        )

        self.dw = np.array(
            [
                np.zeros([self.layer_sizes[i], self.layer_sizes[i + 1]])
                for i in range(self.num_layers - 1)
            ],
            dtype=object,
        )

        self.db = np.array(
            [
                np.zeros([self.layer_sizes[i + 1], 1])
                for i in range(self.num_layers - 1)
            ],
            dtype=object,
        )

        self.df = np.array(
            [
                np.zeros([self.layer_sizes[i + 1], 1])
                for i in range(self.num_layers - 1)
            ],
            dtype=object,
        )

        self.y[0] = np.ones([self.layer_sizes[0], 1])

    def architecture(self) -> None:
        """
        prints the schema of the neural architecture
        """
        print("-------------------------------------------")
        print("Printing Network Architecture...")
        print("-------------------------------------------")
        print(f"The Network has {self.num_layers} layers.")
        print(f"Hidden Layers: {self.num_layers - 2}")
        print(f"Input Layer: {self.layer_sizes[0]} neuron(s)")
        print(f"Output Layer: {self.layer_sizes[-1]} neuron(s)")
        print(f"Batch Size: {self.batch_size}")
        print("-------------------------------------------")
        print("Displaying Dimensions....")
        print("-------------------------")
        print("Layer {1}:")
        print("y_input : {np.shape(self.y[0])}")
        for i in range(self.num_layers - 1):
            print("-------------------------------------------")
            print(f"Layer {i + 2}:")
            print(f"Weight: {np.shape(self.w[i])}")
            print(f"Bias  : {np.shape(self.b[i])}")
            print(f"f(x)  : {np.shape(self.y[i + 1])}")
            print(f"df(x) : {np.shape(self.df[i])}")
            print(f"dw    : {np.shape(self.dw[i])}")
            print(f"db    : {np.shape(self.db[i])}")
            print()
            print(f"y({1}) = [w({1})*y({i})] + b({i+1})")
            print(
                f"{np.shape(self.y[i + 1])} = ({np.shape(self.y[i])}*{np.shape(self.w[i])}) + transpose({np.shape(self.b[i])})"
            )
        print("-------------------------------------------")

    @staticmethod
    def net_f_df(z, act="relu"):
        """
        Computes the activation function and its derivative
        :type z: float
        :param z: linear evaluation

        :type act: str
        :param act: type of the activation function
        """
        if act == "sigmoid":
            f = 1 / (1 + np.exp(-z))
            df = (f**2) * np.exp(-z)
        elif act == "relu":
            f = z * (z > 0)
            df = z > 0
        return f, df

    def forward_step(self, y_in, w, b):
        """
        Computes the linear and non-linear value for the given Z
        """
        z = np.dot(y_in, w) + np.transpose(b)
        return self.net_f_df(z, act="relu")

    @staticmethod
    def backward_step(delta, w, df):
        val = np.dot(delta, np.transpose(w)) * df
        return val

    def back_propogate(self, y_target):
        batch_size = np.shape(self.y[0])[0]
        delta = (self.y[-1] - y_target) * self.df[-1]
        self.dw[-1] = np.dot(np.transpose(self.y[-2]), delta) / batch_size
        self.db[-1] = np.dot(np.transpose(delta), np.ones((batch_size, 1))) / batch_size

        for i in range(self.num_layers - 2):
            delta = self.backward_step(delta, self.w[-1 - i], self.df[-2 - i])
            self.dw[-2 - i] = np.dot(np.transpose(self.y[-3 - i]), delta) / batch_size
            self.db[-2 - i] = (
                np.dot(np.transpose(delta), np.ones((batch_size, 1))) / batch_size
            )

    def gradient_step(self, lr: float):
        for i in range(self.num_layers - 1):
            self.w[i] -= lr * self.dw[i]
            self.b[i] -= lr * self.db[i]

    def apply_net(self, y_in):
        """
        Initializes the architecture
        """
        self.y[0] = y_in
        for i in range(self.num_layers - 1):
            # print("Computing f and df for layer {}".format(i+1))
            self.y[i + 1], self.df[i] = self.forward_step(
                self.y[i], self.w[i], self.b[i]
            )
        return self.y[-1]

    def train_net(self, y_in, y_target, lr):
        batch_size = np.shape(self.y[0])[0]
        y_out = self.apply_net(y_in)
        self.back_propogate(y_target)
        self.gradient_step(lr)
        # cost = ((y_target-self.y[-1])**2).sum()/batchSize
        cost = ((y_target - y_out) ** 2).sum() / batch_size
        return cost

    def simulate_model(self, steps, y_in, y_target, lr):
        cc = []
        for i in range(steps):
            cost = self.train_net(y_in, y_target, lr)
            cc.append(cost)
        return cc

    @staticmethod
    def my_func(x):
        res = np.sin(x)
        return res
