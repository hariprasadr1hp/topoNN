"""
util: An utility module
"""
import time
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import h5py
from pyevtk.hl import imageToVTK

class WriteHdf5:
    """
    Storing numpy arrays in hdf5 format along with its metadata(attributes) 
    
    :type fname: str
    :param nelx: The file name for the hdf5 file

    :type amount: int
    :param amount: volume of the dataset
    """
    def __init__(self, fname):
        self.fname = fname

    def storeArray(self, array, name):
        pass

    def saveFile(self):
        pass

def timeit(method):
    def timed(*args, **kwargs):
        tstart = time.time()
        result = method(*args, **kwargs)
        tend = time.time()
        print("Time taken: {} seconds".format(tend-tstart))
        return result
    return timed


def saveContour(array, title, fname, xlabel=None, ylabel=None):
    """
    To save an array as a contour image.
    """
    plt.cla()
    plt.clf()
    hm = plt.imshow(array, cmap='coolwarm', vmin=-1,
                    vmax=1, interpolation="nearest")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar(hm)
    plt.savefig(fname)

def func(actual, pred, x, fname, title=None):
    plt.cla()
    plt.clf()
    fig, ax = plt.subplots()
    ax.plot(x, actual, '-', label='actual')
    ax.plot(x, pred, '-', label='predicted')
    plt.title(title)
    plt.legend(loc='upper right')
    plt.savefig(fname)

def formCond2D(NodeIDs, values: tuple):
    """
    Formulate a condition matrix given the values to the node IDs
    """
    condMat = np.zeros((np.size(NodeIDs), 3))
    condMat[:, 0] = NodeIDs
    condMat[:, 1] = values[0]
    condMat[:, 2] = values[1]
    return condMat


def formCond3D(NodeIDs, values: tuple):
    """
    Formulate a condition matrix given the values to the node IDs
    """
    condMat = np.zeros((np.size(NodeIDs), 4))
    condMat[:, 0] = NodeIDs
    condMat[:, 1] = values[0]
    condMat[:, 2] = values[1]
    condMat[:, 3] = values[2]
    return condMat


def eucl_U2(ux, uy):
    """Euclidean distance of u_global"""
    return np.sqrt(ux**2 + uy**2)


def eucl_U3(ux, uy, uz):
    """Euclidean distance of u_global"""
    return np.sqrt(ux**2 + uy**2 + uz**2)


def formMagnitude(nodeVal, xnodes, ynodes):
    """
    forms the magnitude matrix
    """
    totnodes = int(np.shape(nodeVal)[0]/2)
    ux = nodeVal[0:totnodes:2]
    uy = nodeVal[1:totnodes:2]
    func = np.vectorize(ux, uy)
    return func(ux, uy).reshape(xnodes, ynodes)


def splitXY(vec):
    ux = vec[0: np.size(vec): 2]
    uy = vec[1: np.size(vec): 2]
    return ux, uy


class WriteSvg:
    """
    Writing svg files
    """

    mini = 0
    maxi = 100

    def __init__(self, fname, array, scheme="grayscale") -> None:
        self.fname = fname
        self.xdim, self.ydim = np.shape(array)
        self.array = self.normalize_mat(array)
        self.array = self.array.astype(int)
        self.scheme = scheme
        self.initialize()

    @staticmethod
    def initialize():
        """
        Initializing class
        """

    @classmethod
    def set_range(cls, values: tuple) -> None:
        """
        Set the range of the scale
        """
        cls.mini = values[0]
        cls.maxi = values[1]

    @classmethod
    def normalize_mat(cls, mat):
        """
        normalize array values to a range(-x,+x)
        """
        span = cls.maxi - cls.mini
        temp1 = mat - cls.mini
        return np.round((250 / span) * temp1) - 125

    # @staticmethod
    # def normalize_mat(mat):
    #     """
    #     normalize array values to a range(-x,+x)
    #     """
    #     span = mat.max() - mat.min()
    #     temp1 = mat - mat.min()
    #     return np.round((250 / span) * temp1) - 125

    @staticmethod
    def color_grayscale(value: int) -> str:
        """
        Returns a graycale colorcode
        """
        if value > 0:
            return "#{0:x}{0:x}{0:x}".format(2 * value)
        return "#{0:x}{0:x}{0:x}".format(-value)

    @staticmethod
    def color_temperature(value: int) -> str:
        """
        Returns a temperature gradient colorcode
        """
        if value > 0:
            return "#{0:x}0000".format(value)
        return "#0000{0:x}".format(-value)

    def square_fill(self, itag):
        """
        Fills the squares based on its state
        """
        for i in range(self.xdim):
            for j in range(self.ydim):
                ET.SubElement(
                    itag,
                    "rect",
                    width="10",
                    height="10",
                    x="{}".format(10 * i),
                    y="{}".format(10 * j),
                    style="fill:{0};stroke:#000000;stroke-width:1".format(
                        self.color_grayscale(self.array[i, j])
                    ),
                )

    def write_doc(self):
        """
        Writing vector graphics to an SVG file
        """
        svg_doc = ET.Element(
            "svg",
            width="{}mm".format(10 * self.xdim),
            height="{}mm".format(10 * self.ydim),
            viewBox="0 0 {} {}".format(10 * self.xdim, 10 * self.ydim),
        )
        g = ET.SubElement(svg_doc, "g")
        self.square_fill(g)

        tree = ET.ElementTree(svg_doc)
        tree.write(self.fname, encoding="UTF-8", xml_declaration=True)


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
