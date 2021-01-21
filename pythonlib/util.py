"""
util: An utility module
"""
import numpy as np
import h5py
from pyevtk.hl import imageToVTK



def formCond2D(NodeIDs, values: tuple):
    """
    Formulate a condition matrix given the values to the node IDs
    """
    condMat = np.zeros((np.size(NodeIDs),3))
    condMat[:,0] = NodeIDs
    condMat[:,1] = values[0]
    condMat[:,2] = values[1]    
    return condMat



class WriteVTK:
    def __init__(self, elemNode, nodeCoord, fname):
        self.elemNode = elemNode
        self.nodeCoord = nodeCoord
        self.fname = fname
        self.initialize()

    def initialize(self):
        self.totnodes = np.shape(self.totnodes)[0]
        self.totelems = np.shape(self.totelems)[0]


    def writeImage(self):
        pass

# Dimensions
# totelems = nx * ny * nz
# totnodes = (nx + 1) * (ny + 1) * (nz + 1)
# # Variables
# pressure = np.random.rand(totelems).reshape((nx, ny, nz), order='C')
# temp = np.random.rand(totnodes).reshape((nx + 1, ny + 1, nz + 1))
# imageToVTK("./image", cellData={"pressure": pressure}, pointData={"temp": temp})
