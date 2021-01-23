"""
neuralNet1.py
"""

import numpy as np

class neuralNet:
    def __init__(self, layerSizes, batchSize):
        self.layerSizes = layerSizes
        self.batchSize = batchSize
        self.variableGen()
    # _______________________________________________________________

    def variableGen(self):
        # global y,w,b,df,dw,db
        numLayers = len(self.layerSizes)
        self.y  = ['self.' + 'y'+str(i) for i in range(numLayers)]
        self.w  = ['self.' + 'w'+str(i) for i in range(1, numLayers)]
        self.b  = ['self.' + 'b'+str(i) for i in range(1, numLayers)]
        self.df = ['self.' + 'df'+str(i) for i in range(1, numLayers)]
        self.dw = ['self.' + 'dw'+str(i) for i in range(1, numLayers)]
        self.db = ['self.' + 'db'+str(i) for i in range(1, numLayers)]

        for i in range(numLayers-1):
            globals()[self.w[i]] = np.random.uniform(low=-1, high=1,
                        size=[self.layerSizes[i], self.layerSizes[i+1]])
            globals()[self.b[i]] = np.random.uniform(low=-1, high=1,
                        size=[self.layerSizes[i+1], 1])

            globals()[self.dw[i]] = np.zeros(
                (self.layerSizes[i], self.layerSizes[i+1]))
            globals()[self.db[i]] = np.zeros((self.layerSizes[i+1], 1))
            globals()[self.df[i]] = np.zeros((self.layerSizes[i+1], 1))

        globals()[self.y[0]] = np.random.uniform(low=-1, high=1,
                                                 size=[self.batchSize, self.layerSizes[0]])

        for i in range(1, numLayers):
            globals()[self.y[i]] = np.zeros(self.layerSizes[i])
    # _______________________________________________________________

    def architecture(self):
        """
        prints the schema of the neural architecture
        """
        # global y,w,b,dw,db
        numLayers = len(self.layerSizes)
        print("-------------------------------------------")
        print("Printing Network Architecture...")
        print("-------------------------------------------")
        print("The Network has {} layers.".format(numLayers))
        print("Hidden Layers: {}".format(numLayers-2))
        print("Input Layer: {} neuron(s)".format(self.layerSizes[0]))
        print("Output Layer: {} neuron(s)".format(self.layerSizes[-1]))
        print("Batch Size: {}".format(self.batchSize))
        print("-------------------------------------------")
        print("Displaying Dimensions....")
        print("-------------------------")
        print("Layer {}:".format(1))
        print("y_input : {}".format(np.shape(globals()[self.y[0]])))
        for i in range(numLayers-1):
            print("-------------------------------------------")
            print("Layer {}:".format(i+2))
            print("Weight: {}".format(np.shape(globals()[self.w[i]])))
            print("Bias  : {}".format(np.shape(globals()[self.b[i]])))
            print("f(x)  : {}".format(np.shape(globals()[self.y[i+1]])))
            print("df(x) : {}".format(np.shape(globals()[self.df[i]])))
            print("dw    : {}".format(np.shape(globals()[self.dw[i]])))
            print("db    : {}".format(np.shape(globals()[self.db[i]])))
            print()
            print("y({1}) = [w({1})*y({0})] + b({1})".format(i, i+1))
            print("{} = ({}*{}) + transpose({})"
                  .format(
                      np.shape(globals()[self.y[i+1]]),
                      np.shape(globals()[self.y[i]]),
                      np.shape(globals()[self.w[i]]),
                      np.shape(globals()[self.b[i]])))
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
        # global y,w,b,df,dw,db
        # global layerSizes
        batchSize = np.shape(globals()[self.y[0]])[0]
        numLayers = len(self.layerSizes)
        delta = (globals()[self.y[-1]] - y_target) * globals()[self.df[-1]]
        globals()[self.dw[-1]] = np.dot(
            np.transpose(globals()[self.y[-2]]),
            delta)/batchSize
        globals()[self.db[-1]] = np.dot(np.transpose(delta),
                                        np.ones((batchSize, 1)))/batchSize

        for i in range(numLayers-2):
            delta = self.backwardStep(delta,
                                      globals()[self.w[-1-i]],
                                      globals()[self.df[-2-i]])
            globals()[self.dw[-2-i]] = np.dot(
                np.transpose(globals()[self.y[-3-i]]),
                delta)/batchSize
            globals()[self.db[-2-i]] = np.dot(np.transpose(delta),
                                              np.ones((batchSize, 1)))/batchSize
    # _______________________________________________________________

    def gradientStep(self, lr: float):
        # global w,b,dw,db,layerSizes
        numLayers = len(self.layerSizes)
        for i in range(numLayers-1):
            globals()[self.w[i]] -= lr * globals()[self.dw[i]]
            globals()[self.b[i]] -= lr * globals()[self.db[i]]
    # _______________________________________________________________

    def applyNet(self, y_in):
        # global layerSizes,y,w,b,df
        numLayers = len(self.layerSizes)
        globals()[self.y[0]] = y_in
        for i in range(numLayers-1):
            #print("Computing f and df for layer {}".format(i+1))
            globals()[self.y[i+1]], globals()[self.df[i]] = self.forwardStep(
                globals()[self.y[i]],
                globals()[self.w[i]],
                globals()[self.b[i]])
        return globals()[self.y[-1]]
    # _______________________________________________________________

    def trainNet(self, y_in, y_target, lr):
        # global y,w
        # global layerSizes
        batchSize = np.shape(globals()[self.y[0]])[0]
        y_out = self.applyNet(globals()[self.y[0]])
        self.backProp(y_target)
        self.gradientStep(lr)
        cost = ((y_target-globals()[self.y[-1]])**2).sum()/batchSize
        return cost
    # _______________________________________________________________

    def myfunc(self, x):
        res = np.sin(x)
        return res
    # _______________________________________________________________
