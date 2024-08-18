"""
util: An utility module
"""

import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# from pyevtk.hl import imageToVTK


def timeit(method):
    def timed(*args, **kwargs):
        tstart = time.time()
        result = method(*args, **kwargs)
        tend = time.time()
        print(f"Time taken: {tend - tstart} seconds")
        return result

    return timed


def create_dir_if_not_exists(fpath: Path) -> None:
    if not os.path.exists(fpath):
        os.makedirs(fpath)


def save_contour(array, title, fname, xlabel=None, ylabel=None):
    """
    To save an array as a contour image.
    """
    plt.cla()
    plt.clf()
    hm = plt.imshow(array, cmap="coolwarm", vmin=-1, vmax=1, interpolation="nearest")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar(hm)
    plt.savefig(fname)


def plot_fit(actual, pred, x, fname, title=None):
    plt.cla()
    plt.clf()
    fig, ax = plt.subplots()
    ax.plot(x, actual, "-", label="actual")
    ax.plot(x, pred, "-", label="predicted")
    plt.title(title)
    plt.legend(loc="upper right")
    plt.savefig(fname)


def formulate_2d_condns(node_ids, values: tuple):
    """
    Formulate a condition matrix given the values to the node IDs
    """
    condn_matx = np.zeros((np.size(node_ids), 3))
    condn_matx[:, 0] = node_ids
    condn_matx[:, 1] = values[0]
    condn_matx[:, 2] = values[1]
    return condn_matx


def get_eucl_u_global_2d(ux, uy):
    """Euclidean distance of u_global"""
    return np.sqrt(ux**2 + uy**2)


def formulate_magnitude(node_values, nodes_x, nodes_y):
    """
    forms the magnitude matrix
    """
    totnodes = int(np.shape(node_values)[0] / 2)
    ux = node_values[0:totnodes:2]
    uy = node_values[1:totnodes:2]
    func = np.vectorize(ux, uy)
    return func(ux, uy).reshape(nodes_x, nodes_y)


def split_xy(vec):
    ux = vec[0 : np.size(vec) : 2]
    uy = vec[1 : np.size(vec) : 2]
    return ux, uy


class WriteSvg:
    """
    Writing svg files
    """

    minimum = 0
    maximum = 100

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
        cls.minimum = values[0]
        cls.maximum = values[1]

    @classmethod
    def normalize_mat(cls, mat):
        """
        normalize array values to a range(-x,+x)
        """
        span = cls.maximum - cls.minimum
        temp1 = mat - cls.minimum
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
            return f"#{2 * value:x}{2 * value:x}{2 * value:x}"
        return f"#{-value:x}{-value:x}{-value:x}"

    @staticmethod
    def color_temperature(value: int) -> str:
        """
        Returns a temperature gradient colorcode
        """
        if value > 0:
            return f"#{value:x}0000"
        return f"#0000{-value:x}"

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
                    x=f"{10 * i}",
                    y=f"{10 * j}",
                    style=f"fill:{self.color_grayscale(self.array[i, j])};stroke:#000000;stroke-width:1",
                )

    def write_doc(self):
        """
        Writing vector graphics to an SVG file
        """
        svg_doc = ET.Element(
            "svg",
            width=f"{10 * self.xdim}mm",
            height=f"{10 * self.ydim}mm",
            viewBox=f"0 0 {10 * self.xdim} {10 * self.ydim}",
        )
        g = ET.SubElement(svg_doc, "g")
        self.square_fill(g)

        tree = ET.ElementTree(svg_doc)
        tree.write(self.fname, encoding="UTF-8", xml_declaration=True)


class WriteVTK:
    def __init__(self, elem_node, node_coord, fname):
        self.elem_node = elem_node
        self.node_coord = node_coord
        self.fname = fname
        self.initialize()

    def write_image(self):
        pass


# Dimensions
# totelems = nx * ny * nz
# totnodes = (nx + 1) * (ny + 1) * (nz + 1)
# # Variables
# pressure = np.random.rand(totelems).reshape((nx, ny, nz), order='C')
# temp = np.random.rand(totnodes).reshape((nx + 1, ny + 1, nz + 1))
# imageToVTK("./image", cellData={"pressure": pressure}, pointData={"temp": temp})
