"""
meshGen
"""

import numpy as np

from pythonlib.errors import TopoNNError


class Plate3D:
    """
    Defines the Geometry of a three-dimensional Plate
    """

    def __init__(self, nelx: int, nely: int, nelz: int) -> None:
        """
        initializing class
        """
        self.nelx = nelx
        self.nely = nely
        self.nelz = nelz
        self.initialize()
        self.generate_elem_nodes()
        self.generate_node_coords()

    def initialize(self) -> None:
        self.totelems = self.nelx * self.nely * self.nelz
        self.totnodes = (self.nelx + 1) * (self.nely + 1) * (self.nelz + 1)

    def set_elem_xyz(self, elem: tuple[int, int, int]) -> None:
        """
        Sets the x,y,z coordinates of the element
        """
        self.nelx = elem[0]
        self.nely = elem[1]
        self.nelz = elem[2]

    def get_elem_xyz(self) -> tuple[int, int, int]:
        """
        Returns the x,y,z coordinates of the element
        """
        return (self.nelx, self.nely, self.nelz)

    def coord_to_node(self, coord: tuple[float, float, float]) -> int:
        """
        Returns the Global node ID for the given origin coordinates

        :type coord: tuple
        :param coord: the x,y,z coordinates of the node
        """
        if self.check_coords(coord):
            node = (
                (coord[2] * (self.nelx + 1) * (self.nely + 1))
                + (coord[1] * (self.nelx + 1))
                + coord[0]
                + 1
            )
            return int(node)
        raise TopoNNError("Invalid Coordinates")

    def node_to_coord(self, node: int) -> tuple[float, float, float]:
        """
        Returns the x,y,z coordinates of the element

        :type node: int
        :param node: the node ID
        """
        if self.check_node_id(node):
            node = node - 1
            coord_z = node // ((self.nelx + 1) * (self.nely + 1))
            temp = node % ((self.nelx + 1) * (self.nely + 1))
            coord_y = temp // (self.nelx + 1)
            coord_x = temp % (self.nelx + 1)
            return (coord_x, coord_y, coord_z)
        raise TopoNNError("Invalid Node")

    def get_elem_origin(self, elem: int) -> tuple[float, float, float]:
        """
        Returns the coordinates (x,y,z) of the element's origin

        :type elem: int
        :param elem: the element ID
        """
        temp1 = elem - 1
        coord_z = temp1 // (self.nelx * self.nely)
        temp2 = temp1 % (self.nelx * self.nely)
        coord_y = temp2 // self.nelx
        coord_x = temp2 % self.nelx
        return (coord_x, coord_y, coord_z)

    def get_elem_id(self, origin: tuple[float, float, float]) -> int:
        """
        Returns the element ID for the given origin

        :type origin: tuple
        :param origin: the coordinates of the element's origin
        """
        if self.check_origin(origin):
            return int(
                (origin[2] * self.nelx * self.nely)
                + (origin[1] * self.nelx)
                + origin[0]
                + 1
            )
        return -1

    def get_hex8_ids(self, origin: tuple[float, float, float]) -> np.ndarray:
        """
        Writes the global IDs of the nodes of an element, given the
        coordinates of an element at its origin

        :type origin: tuple
        :param origin: the coordinates of the element's origin
        """
        if self.check_origin(origin):
            hex_id = np.zeros((8), dtype=int)
            hex_id[0] = self.coord_to_node(origin)
            hex_id[1] = hex_id[0] + 1
            hex_id[2] = hex_id[0] + (self.nelx + 1)
            hex_id[3] = hex_id[2] + 1
            hex_id[4] = hex_id[0] + ((self.nelx + 1) * (self.nely + 1))
            hex_id[5] = hex_id[4] + 1
            hex_id[6] = hex_id[4] + (self.nelx + 1)
            hex_id[7] = hex_id[6] + 1
            return hex_id
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

    def check_coords(self, coords: tuple[float, float, float]) -> bool:
        """
        checks whether the node coordinates exist

        :type coords: tuple
        :param coords: (x,y,z) coordinates
        """
        on_x = bool(0 <= coords[0] <= self.nelx)
        on_y = bool(0 <= coords[1] <= self.nely)
        on_z = bool(0 <= coords[2] <= self.nelz)
        return bool(on_x and on_y and on_z)

    def check_origin(self, origin: tuple[float, float, float]) -> bool:
        """
        checks whether the origin coordinates exist

        :type origin: tuple
        :param origin: the coordinates of the element's origin
        """
        on_x = bool(0 <= origin[0] < self.nelx)
        on_y = bool(0 <= origin[1] < self.nely)
        on_z = bool(0 <= origin[2] < self.nelz)
        return bool(on_x and on_y and on_z)

    def get_left(self):
        """
        Returns the nodes to the left
        """
        return np.arange(1, self.totnodes + 1, (self.nelx + 1))

    def get_right(self):
        """
        Returns the nodes to the right
        """
        return np.arange((self.nelx + 1), self.totnodes + 1, (self.nelx + 1))

    def get_up(self):
        """
        Returns the nodes on top
        """
        return np.arange(1, ((self.nelx + 1) * (self.nely + 1)) + 1)

    def get_down(self):
        """
        Returns the nodes on the bottom
        """
        return np.arange(
            self.totnodes - ((self.nelx + 1) * (self.nely + 1)) + 1, self.totnodes + 1
        )

    def get_all(self):
        """
        Returns all the nodes of the model
        """
        return np.arange(1, self.totnodes + 1)

    def generate_elem_nodes(self):
        """
        Generates the Element-Node connectivity table
        """
        self.elem_nodes = np.zeros((self.totelems, 8), dtype=int)
        for elem in range(self.totelems):
            self.elem_nodes[elem, :] = self.get_hex8_ids(self.get_elem_origin(elem + 1))

    def generate_node_coords(self):
        """
        Generates the Node-coordinates table
        """
        self.node_coords = np.zeros((self.totnodes, 3), dtype=float)
        for node in range(self.totnodes):
            self.node_coords[node, :] = self.node_to_coord(node + 1)


class Plate2D:
    """
    Defines the Geometry of a two-dimensional Plate
    """

    def __init__(self, nelx: int, nely: int) -> None:
        """
        initializing class
        """
        self.nelx = nelx
        self.nely = nely
        self.initialize()
        self.generate_elem_nodes()
        self.generate_node_coords()

    def initialize(self) -> None:
        self.totelems = self.nelx * self.nely
        self.totnodes = (self.nelx + 1) * (self.nely + 1)

    def set_elem_xy(self, elem: tuple[int, int]):
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

    def get_left(self):
        """
        Returns the nodes to the left
        """
        return np.arange(1, self.totnodes + 1, (self.nelx + 1))

    def get_right(self):
        """
        Returns the nodes to the right
        """
        return np.arange((self.nelx + 1), self.totnodes + 1, (self.nelx + 1))

    def get_up(self):
        """
        Returns the nodes on top
        """
        return np.arange(self.totnodes - (self.nelx + 1) + 1, self.totnodes + 1)

    def get_down(self):
        """
        Returns the nodes on the bottom
        """
        return np.arange(1, (self.nelx + 1) + 1)

    def get_all(self):
        """
        Returns all the nodes of the model
        """
        return np.arange(1, self.totnodes + 1)

    def generate_elem_nodes(self):
        """
        Generates the Element-Node connectivity table
        """
        self.elem_nodes = np.zeros((self.totelems, 4), dtype=int)
        for elem in range(self.totelems):
            self.elem_nodes[elem, :] = self.get_quad_ids(self.get_elem_origin(elem + 1))

    def generate_node_coords(self):
        """
        Generates the Node-coordinates table
        """
        self.node_coords = np.zeros((self.totnodes, 2), dtype=float)
        for node in range(self.totnodes):
            self.node_coords[node, :] = self.node_to_coord(node + 1)
