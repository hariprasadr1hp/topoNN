"""
neuralNet.py
"""

import numpy as np


class neuralNet:
    def __init__(self, layerSizes, batchSize):
        self.layerSizes = layerSizes
        self.batchSize = batchSize
        self.numLayers = len(self.layerSizes)
        self.variableGen()
    # _______________________________________________________________

    def variableGen(self):
        self.y = np.array([np.zeros([self.layerSizes[i], 1])
                           for i in range(self.numLayers)], dtype=object)

        self.w = np.array([np.random.uniform(low=1, high=1, size=[
                          self.layerSizes[i], self.layerSizes[i+1]]) for i in range(self.numLayers-1)], dtype=object)

        self.b = np.array([np.random.uniform(low=1, high=1, size=[
                          self.layerSizes[i+1], 1]) for i in range(self.numLayers-1)], dtype=object)

        self.df = np.array([np.zeros(self.layerSizes[i+1])
                            for i in range(self.numLayers-1)], dtype=object)

        self.dw = np.array([np.zeros([self.layerSizes[i], self.layerSizes[i+1]])
                            for i in range(self.numLayers-1)], dtype=object)

        self.db = np.array([np.zeros([self.layerSizes[i+1], 1])
                            for i in range(self.numLayers-1)], dtype=object)

        self.df = np.array([np.zeros([self.layerSizes[i+1], 1])
                            for i in range(self.numLayers-1)], dtype=object)

        self.y[0] = np.ones([self.layerSizes[0], 1])

    # _______________________________________________________________

    def architecture(self):
        """
        prints the schema of the neural architecture
        """
        print("-------------------------------------------")
        print("Printing Network Architecture...")
        print("-------------------------------------------")
        print("The Network has {} layers.".format(self.numLayers))
        print("Hidden Layers: {}".format(self.numLayers-2))
        print("Input Layer: {} neuron(s)".format(self.layerSizes[0]))
        print("Output Layer: {} neuron(s)".format(self.layerSizes[-1]))
        print("Batch Size: {}".format(self.batchSize))
        print("-------------------------------------------")
        print("Displaying Dimensions....")
        print("-------------------------")
        print("Layer {}:".format(1))
        print("y_input : {}".format(np.shape(self.y[0])))
        for i in range(self.numLayers-1):
            print("-------------------------------------------")
            print("Layer {}:".format(i+2))
            print("Weight: {}".format(np.shape(self.w[i])))
            print("Bias  : {}".format(np.shape(self.b[i])))
            print("f(x)  : {}".format(np.shape(self.y[i+1])))
            print("df(x) : {}".format(np.shape(self.df[i])))
            print("dw    : {}".format(np.shape(self.dw[i])))
            print("db    : {}".format(np.shape(self.db[i])))
            print()
            print("y({1}) = [w({1})*y({0})] + b({1})".format(i, i+1))
            print("{} = ({}*{}) + transpose({})"
                  .format(
                      np.shape(self.y[i+1]),
                      np.shape(self.y[i]),
                      np.shape(self.w[i]),
                      np.shape(self.b[i])))
        print("-------------------------------------------")
    # _______________________________________________________________

    @staticmethod
    def net_f_df(z, act='relu'):
        """
        Computes the activation function and its derivative
        :type z: float
        :param z: linear evaluation

        :type act: str
        :param act: type of the activation function
        """
        if act == 'sigmoid':
            f = 1/(1+np.exp(-z))
            df = (f**2) * np.exp(-z)
        elif act == 'relu':
            f = z*(z > 0)
            df = z > 0
        return f, df
    # _______________________________________________________________

    def forwardStep(self, y_in, w, b):
        """
        Computes the linear and non-linear value for the given Z
        """
        z = np.dot(y_in, w) + np.transpose(b)
        return self.net_f_df(z, act='relu')
    # _______________________________________________________________

    @staticmethod
    def backwardStep(delta, w, df):
        val = np.dot(delta, np.transpose(w)) * df
        return val
    # _______________________________________________________________

    def backProp(self, y_target):
        batchSize = np.shape(self.y[0])[0]
        delta = (self.y[-1] - y_target) * self.df[-1]
        self.dw[-1] = np.dot(np.transpose(self.y[-2]),
                             delta)/batchSize
        self.db[-1] = np.dot(np.transpose(delta),
                             np.ones((batchSize, 1)))/batchSize

        for i in range(self.numLayers-2):
            delta = self.backwardStep(delta,
                                      self.w[-1-i],
                                      self.df[-2-i])
            self.dw[-2-i] = np.dot(
                np.transpose(self.y[-3-i]),
                delta)/batchSize
            self.db[-2-i] = np.dot(np.transpose(delta),
                                   np.ones((batchSize, 1)))/batchSize
    # _______________________________________________________________

    def gradientStep(self, lr: float):
        for i in range(self.numLayers-1):
            self.w[i] -= lr * self.dw[i]
            self.b[i] -= lr * self.db[i]
    # _______________________________________________________________

    def applyNet(self, y_in):
        """
        Initializes the architecture
        """
        self.y[0] = y_in
        for i in range(self.numLayers-1):
            #print("Computing f and df for layer {}".format(i+1))
            self.y[i+1], self.df[i] = self.forwardStep(self.y[i],
                                                       self.w[i],
                                                       self.b[i])
        return self.y[-1]
    # _______________________________________________________________

    def trainNet(self, y_in, y_target, lr):
        batchSize = np.shape(self.y[0])[0]
        y_out = self.applyNet(y_in)
        self.backProp(y_target)
        self.gradientStep(lr)
        cost = ((y_target-self.y[-1])**2).sum()/batchSize
        return cost
    # _______________________________________________________________

    @staticmethod
    def myfunc(x):
        res = np.sin(x)
        return res
    # _______________________________________________________________
