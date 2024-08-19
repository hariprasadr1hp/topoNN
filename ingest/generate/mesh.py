"""
mesh_generate.py
"""

from typing import Any

import numpy as np

from process.errors import TopoNNError


class Plate:
    """
    Defines the geometry of a two-dimensional plate (non-perforated, rectangular)
    """

    def __init__(self, nelx: int, nely: int) -> None:
        self.nelx = nelx
        self.nely = nely
        self.initialize()

    def initialize(self) -> None:
        self.totelems = self.nelx * self.nely
        self.totnodes = (self.nelx + 1) * (self.nely + 1)
        self.elem_nodes = self.get_elem_nodes_connectivity_matx()
        self.node_coords = self.get_node_coords_matx()

    def set_elem_xy(self, elem: tuple[int, int]) -> None:
        """
        Sets the x,y,z coordinates of the element
        """
        self.nelx = elem[0]
        self.nely = elem[1]

    def get_elem_xy(self) -> tuple[int, int]:
        """
        Returns the x,y coordinates of the element
        """
        return (self.nelx, self.nely)

    def coord_to_node(self, coord: tuple[float, float]) -> int:
        """
        Returns the Global node ID for the given origin coordinates

        :type coord: tuple
        :param coord: the x,y coordinates of the node
        """
        if self.check_coords(coord):
            node = (self.nelx + 1) * coord[1] + coord[0] + 1
            return int(node)
        raise TopoNNError("Invalid Coordinates")

    def node_to_coord(self, node: int) -> tuple[float, float]:
        """
        Returns the x,y coordinates of the element

        :type node: int
        :param node: the node ID
        """
        if self.check_node_id(node):
            node = node - 1
            coord_y = node // (self.nelx + 1)
            coord_x = node % (self.nelx + 1)
            return (coord_x, coord_y)
        raise TopoNNError("Invalid Node")

    def get_elem_origin(self, elem: int) -> tuple[float, float]:
        """
        Returns the coordinates (x,y) of the element's origin

        :type elem: int
        :param elem: the element ID
        """
        temp = elem - 1
        coord_y = temp // (self.nelx)
        coord_x = temp % (self.nelx)
        return (coord_x, coord_y)

    def get_elem_id(self, origin: tuple[float, float]) -> int:
        """
        Returns the element ID for the given origin

        :type origin: tuple
        :param origin: the coordinates of the element's origin
        """
        if self.check_origin(origin):
            return int((origin[1] * self.nelx) + origin[0] + 1)
        return -1

    def get_quad_ids(self, origin: tuple[float, float]):
        """
        Writes the global IDs of the nodes of an element, given the
        coordinates of an element at its origin

        :type origin: tuple
        :param origin: the coordinates of the element's origin
        """
        if self.check_origin(origin):
            quad_id = np.zeros((4), dtype=int)
            quad_id[0] = self.coord_to_node(origin)
            quad_id[1] = quad_id[0] + 1
            quad_id[2] = quad_id[0] + (self.nelx + 1)
            quad_id[3] = quad_id[2] + 1
            return quad_id
        raise TopoNNError("Invalid coordinates")

    def check_elem_id(self, elem: int) -> bool:
        """
        checks whether an element with the following ID exist

        :type elem: int
        :param elem: the element ID
        """
        return bool(0 < elem <= self.totelems)

    def check_node_id(self, node: int) -> bool:
        """
        checks whether a node with the following ID exist

        :type node: int
        :param node: the node ID
        """
        return bool(0 < node <= self.totnodes)

    def check_coords(self, coords: tuple[float, float]) -> bool:
        """
        checks whether the node coordinates exist

        :type coords: tuple
        :param coords: (x,y) coordinates
        """
        on_x = bool(0 <= coords[0] <= self.nelx)
        on_y = bool(0 <= coords[1] <= self.nely)
        return bool(on_x and on_y)

    def check_origin(self, origin: tuple[float, float]) -> bool:
        """
        checks whether the origin coordinates exist

        :type origin: tuple
        :param origin: the coordinates of the element's origin
        """
        on_x = bool(0 <= origin[0] < self.nelx)
        on_y = bool(0 <= origin[1] < self.nely)
        return bool(on_x and on_y)

    def get_left(self) -> np.ndarray[Any, Any]:
        """
        Returns nodes to the left edge
        """
        return np.arange(1, self.totnodes + 1, (self.nelx + 1))

    def get_right(self):
        """
        Returns nodes to the right edge
        """
        return np.arange((self.nelx + 1), self.totnodes + 1, (self.nelx + 1))

    def get_up(self) -> np.ndarray[Any, Any]:
        """
        Returns nodes on the top edge
        """
        return np.arange(self.totnodes - (self.nelx + 1) + 1, self.totnodes + 1)

    def get_down(self) -> np.ndarray[Any, Any]:
        """
        Returns nodes at the bottom edge
        """
        return np.arange(1, (self.nelx + 1) + 1)

    def get_all(self) -> np.ndarray[Any, Any]:
        """
        Returns all the nodes of the model
        """
        return np.arange(1, self.totnodes + 1)

    def get_elem_nodes_connectivity_matx(self) -> np.ndarray[Any, Any]:
        """
        Generates the element-node connectivity table

        each row contains corresponding element nodes, elements sorted (ascending)
        """
        elem_nodes = np.zeros((self.totelems, 4), dtype=int)
        for elem in range(self.totelems):
            elem_nodes[elem, :] = self.get_quad_ids(self.get_elem_origin(elem + 1))
        return elem_nodes

    def get_node_coords_matx(self) -> np.ndarray[Any, Any]:
        """
        Generates the node-coordinates table
        """
        node_coords = np.zeros((self.totnodes, 2), dtype=float)
        for node in range(self.totnodes):
            node_coords[node, :] = self.node_to_coord(node + 1)
        return node_coords
