"""
meshGen
"""
import numpy as np


class Plate:
    def __init__(self, nelx: int, nely: int, nelz: int):
        """
        initializing class
        """
        self.nelx = nelx
        self.nely = nely
        self.nelz = nelz
        self.initialize()
        self.genElemNodes()
        self.genNodeCoords()

    def initialize(self):
        self.totelems = self.nelx * self.nely * self.nelz
        self.totnodes = (self.nelx+1) * (self.nely+1) * (self.nelz+1)

    def setElemXYZ(self, elem: tuple):
        """
        Sets the x,y,z coordinates of the element
        """
        self.nelx = elem[0]
        self.nely = elem[1]
        self.nelz = elem[2]

    def getElemXYZ(self) -> tuple:
        """
        Returns the x,y,z coordinates of the element
        """
        return (self.nelx, self.nely, self.nelz)

    def coordToNode(self, coord: tuple) -> int:
        """
        Returns the Global node ID for the given origin coordinates

        :type coord: tuple
        :param coord: the x,y,z coordinates of the node
        """
        if self.checkCoords(coord):
            node = (coord[2] * (self.nelx+1) * (self.nely+1)) + \
                (coord[1] * (self.nelx+1)) + coord[0] + 1
            return node
        raise Exception("Invalid Coordinates")

    def nodeToCoord(self, node: int) -> tuple:
        """
        Returns the x,y,z coordinates of the element

        :type node: int
        :param node: the node ID
        """
        if self.checkNodeID(node):
            node = node - 1
            coordZ = node // ((self.nelx+1) * (self.nely+1))
            temp = node % ((self.nelx+1) * (self.nely+1))
            coordY = temp // (self.nelx+1)
            coordX = temp % (self.nelx+1)
            return (coordX, coordY, coordZ)
        raise Exception("Invalid Node")

    def getElemOrigin(self, elem: int) -> tuple:
        """
        Returns the coordinates (x,y,z) of the element's origin

        :type elem: int
        :param elem: the element ID
        """
        temp1 = elem - 1
        coordZ = temp1 // (self.nelx * self.nely)
        temp2 = temp1 % (self.nelx * self.nely)
        coordY = temp2 // self.nelx
        coordX = temp2 % self.nelx
        return (coordX, coordY, coordZ)

    def getElemID(self, origin: tuple) -> int:
        """
        Returns the element ID for the given origin

        :type origin: tuple
        :param origin: the coordinates of the element's origin
        """
        if self.checkOrigin(origin):
            return int((origin[2] * self.nelx * self.nely) +
                       (origin[1] * self.nelx) +
                       origin[0] + 1)
        return -1

    def getHex8IDs(self, origin: tuple):
        """
        Writes the global IDs of the nodes of an element, given the
        coordinates of an element at its origin

        :type origin: tuple
        :param origin: the coordinates of the element's origin
        """
        if self.checkOrigin(origin):
            hexID = np.zeros((8), dtype=int)
            hexID[0] = self.coordToNode(origin)
            hexID[1] = hexID[0] + 1
            hexID[2] = hexID[0] + (self.nelx + 1)
            hexID[3] = hexID[2] + 1
            hexID[4] = hexID[0] + ((self.nelx+1)*(self.nely+1))
            hexID[5] = hexID[4] + 1
            hexID[6] = hexID[4] + (self.nelx + 1)
            hexID[7] = hexID[6] + 1
            return hexID
        raise Exception("Invalid coordinates")

    def checkElemID(self, elem: int) -> bool:
        """
        checks whether an element with the following ID exist

        :type elem: int
        :param elem: the element ID
        """
        return bool(0 < elem <= self.totelems)

    def checkNodeID(self, node: int) -> bool:
        """
        checks whether a node with the following ID exist

        :type node: int
        :param node: the node ID
        """
        return bool(0 < node <= self.totnodes)

    def checkCoords(self, coords: tuple) -> bool:
        """
        checks whether the node coordinates exist

        :type coords: tuple
        :param coords: (x,y,z) coordinates
        """
        onX = bool(0 <= coords[0] <= self.nelx)
        onY = bool(0 <= coords[1] <= self.nely)
        onZ = bool(0 <= coords[2] <= self.nelz)
        return bool(onX and onY and onZ)

    def checkOrigin(self, origin: tuple) -> bool:
        """
        checks whether the origin coordinates exist

        :type origin: tuple
        :param origin: the coordinates of the element's origin
        """
        onX = bool(0 <= origin[0] < self.nelx)
        onY = bool(0 <= origin[1] < self.nely)
        onZ = bool(0 <= origin[2] < self.nelz)
        return bool(onX and onY and onZ)

    def getLeft(self):
        """
        Returns the nodes to the left
        """
        return np.arange(1, self.totnodes, (self.nelx+1))

    def getRight(self):
        """
        Returns the nodes to the right
        """
        return np.arange((self.nelx+1), self.totnodes, (self.nelx+1))

    def getUp(self):
        """
        Returns the nodes on top
        """
        return np.arange(1, ((self.nelx+1)*(self.nely+1))+1)

    def getDown(self):
        """
        Returns the nodes on the bottom
        """
        return np.arange(self.totnodes-((self.nelx+1)*(self.nely+1))+1, self.totnodes+1)

    def getAll(self):
        """
        Returns all the nodes of the model
        """
        return np.arange(1,self.totnodes+1)

    def genElemNodes(self):
        """
        Generates the Element-Node connectivity table
        """
        self.elemNodes = np.zeros((self.totelems, 8), dtype=int)
        for elem in range(self.totelems):
            self.elemNodes[elem, :] = self.getHex8IDs(
                self.getElemOrigin(elem+1))

    def genNodeCoords(self):
        """
        Generates the Node-coordinates table
        """
        self.nodeCoords = np.zeros((self.totnodes, 3), dtype=float)
        for node in range(self.totnodes):
            self.nodeCoords[node, :] = self.nodeToCoord(node+1)
