"""
This module generates training data
"""
import numpy as np
import matplotlib.pyplot as plt
from pythonlib.meshGen import Plate2D
from pythonlib import util
from pythonlib.util import WriteSvg
from pythonlib import elementRoutine as ER

class DataGen:
    """
    Generates Data for structural optimization

    :type nelx: int
    :param nelx: no.of elements in the x-direction

    :type nely: int
    :param nely: no.of elements in the y-direction

    :type amount: int
    :param amount: volume of the dataset
    """

    def __init__(self, nelx: int, nely: int, amount: int) -> None:
        self.nelx = nelx
        self.nely = nely
        self.amount = amount
        self.generateData()

    def generateData(self) -> None:
        """
        Generating dataset
        """
        for i in range(amount):
            np.random.rand(self.nelx, self.nely)





