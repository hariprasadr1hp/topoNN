"""
This module will solve the FE problem
"""
import numpy as np
import matplotlib.pyplot as plt
from pythonlib.meshGen import Plate3D
from pythonlib import util
from pythonlib import elementRoutine as ER


class solveFE3D:
    """
    Solves the 3D FE problem

    :type mesh3D: class
    :param nodeCoords: the mesh-grid structure

    :type BC: ndarray(N X 4)
    :param BC: Boundary conditions, nodes along with constraints in
                x,y,z direction

    :type FC: ndarray(N X 4)
    :param FC: Boundary conditions, nodes along with values in
                x,y,z direction

    :type matlParams: tuple(E, Nu)
    :param matlParams: youngs modulus and poissons ratio
    """
    def __init__(self, mesh3D, BC, FC, matlParams=(120, 0.3, 5)):
        self.mesh3D = mesh3D
        self.nodeCoords = mesh3D.nodeCoords
        self.elemNodes = mesh3D.elemNodes
        self.matlParams = matlParams
        self.nelx = mesh3D.nelx
        self.nely = mesh3D.nely
        self.nelz = mesh3D.nelz

        self.BC = BC
        self.FC = FC

        self.totnodes = np.shape(self.nodeCoords)[0]
        self.totelems = np.shape(self.elemNodes)[0]
        self.totDOFs = self.totnodes * 3
        self.elemDOF = 24

        # Initialize Matrices
        self.Fint_Global = np.zeros((self.totDOFs))
        self.Fext_Global = np.zeros((self.totDOFs))
        self.K_Global = np.zeros((self.totDOFs, self.totDOFs), dtype=float)
        self.u_Global = np.zeros((self.totDOFs))
        self.du_Global = np.zeros((self.totDOFs))

        # boundary conditions
        self.u_freeDOF = np.array([False for i in range(self.totDOFs)])
        self.u_fixedDOF = np.array([False for i in range(self.totDOFs)])
        self.K_mask = np.array(
            [[True for j in range(self.totDOFs)] for j in range(self.totDOFs)])
        self.boundarySet(self.BC)
        self.forceSet(self.FC)

        # attributes
        self.strainDensity = np.zeros((self.nelx, self.nely, self.nelz))
        self.u_x = np.zeros((self.nelx+1, self.nely+1, self.nelz+1))
        self.u_y = np.zeros((self.nelx+1, self.nely+1, self.nelz+1))
        self.u_xy = np.zeros((self.nelx+1, self.nely+1, self.nelz+1))
    # ____________________________________________________________________

    # @staticmethod
    def get_dofIDs(self, NodeIDs):
        """
        Returns the [x,y,z] IDs for the given Node IDs

        :type NodeIDs: ndarray
        :param NodeIDs: the ID of the node
        """
        def xyz(x):
            return [3*x-3, 3*x-2, 3*x-1]
        NodeDOFs = np.concatenate([xyz(i) for i in NodeIDs])
        return NodeDOFs
    # ____________________________________________________________________

    def forceSet(self, fc) -> None:
        """
        Sets the force conditions

        :type fc: nd.array(N X 4)
        :param fc: the node IDs along with their force constraints
                    in the x,y direction
        """
        for each in fc:
            self.Fext_Global[int(each[0])*3-3] = each[1]
            self.Fext_Global[int(each[0])*3-2] = each[2]
            self.Fext_Global[int(each[0])*3-1] = each[3]
    # ____________________________________________________________________

    # def boundarySet(self,U_ID,vals):
    def boundarySet(self, bc) -> None:
        """
        Sets the boundary conditions

        :type bc: nd.array(N X 4)
        :param bc: the node IDs along with their displacements
                        constraints in the x,y direction
        """
        for each in bc:
            if each[1] == -1:
                self.u_fixedDOF[int(each[0])*3-3] = True
            if each[2] == -1:
                self.u_fixedDOF[int(each[0])*3-2] = True
            if each[3] == -1:
                self.u_fixedDOF[int(each[0])*3-1] = True
            # else:
            #     pass
        self.u_freeDOF = ~self.u_fixedDOF

    # ___________________________________________________________________

    @staticmethod
    def calcStrainEnergy(u_el, K_el) -> float:
        """
        Calculates the starin energy density of an element

        :type u_el: ndarray (24,1)
        :param u_el: the element displacement

        :type K_el: ndarray (24,24)
        :param K_el: the element stiffness matrix
        """
        return u_el.T @ K_el @ u_el

    # ___________________________________________________________________

    def fillMatrix(self, elemID: int, K_el, Fint_el) -> None:
        """
        Fills the sparse Global stiffness matrix with the local
        stifnesses

        :type K_el: ndarray (24,24)
        :param K_el: the local stiffness matrix of an element
        
        :type Fint_el: ndarray (24,1)
        :param Fint_el: the internal force matrix of an element
        """

        def xyz(x):
            return [3*x-3, 3*x-2, 3*x-1]
        dofs = np.concatenate([xyz(i) for i in self.elemNodes[elemID-1]])
        for i, j in enumerate(dofs):
            self.K_Global[j, j] += K_el[i, i]
            self.Fint_Global[j] += Fint_el[i]

    # ____________________________________________________________________

    def assemblyGlobal(self):
        """
        Assembly of the global Force and Stiffness matrices
        """
        hex8_el = np.zeros((8, 3), dtype=float)
        u_el = np.zeros((24), dtype=float)

        for i, elemID in enumerate(self.elemNodes):
            for j, nodeID in enumerate(elemID):
                hex8_el[j] = self.nodeCoords[nodeID-1]
            u_el = self.get_dofIDs(self.elemNodes[i])
            element = ER.Brick(GP=1, hex8_rc=hex8_el)
            K_el, Fint_el = element.hex8(self.matlParams, u_el)
            
            self.calcStrainEnergy(u_el, K_el)
            self.strainDensity[self.mesh3D.getElemOrigin(
                i)] = self.calcStrainEnergy(u_el, K_el)
            self.fillMatrix(i+1, K_el, Fint_el)
    # ___________________________________________________________________

    def matlSet(self, matlParams):
        """
        Sets the material Parameters

        :type matlParams: tuple(float) -> (E,Nu,sigY)
        :param matlParams: the material parameters of the element
        """
        self.matlParams = matlParams
    # ____________________________________________________________________

    def update_duGlobal(self, du_Global_red):
        """
        Updates the global displacement vector after each iteration

        :type du_Global_red: numpy.array(int)
        :param du_Global_red: the nodal displacements of IDs with
                                non-zero displacement
        """
        count = 0
        while count < np.size(du_Global_red):
            for each, _ in enumerate(self.u_freeDOF):
                if self.u_freeDOF[each]:
                    self.du_Global[each] = du_Global_red[count]
                    count += 1
    # ____________________________________________________________________

    def solveProblem(self):
        """
        Solves the formulated FE Problem
        """
        # #iterative-solve
        # for step in range(1):
        #     self.assemblyGlobal()
        #     K_Global_red = self.K_Global[self.u_freeDOF]
        #     K_Global_red = (K_Global_red.T[self.u_freeDOF]).T
        #     Fint_Global_red = self.Fint_Global[self.u_freeDOF]
        #     Fext_Global_red = self.Fext_Global[self.u_freeDOF]
        #     G_red = Fext_Global_red - Fint_Global_red

        #     du_Global_red = np.linalg.solve(K_Global_red, G_red)
        #     self.update_duGlobal(du_Global_red)
        #     self.u_Global += self.du_Global

        #direct-solve
        self.assemblyGlobal()
        K_Global_red = self.K_Global[self.u_freeDOF]
        K_Global_red = (K_Global_red.T[self.u_freeDOF]).T
        Fint_Global_red = self.Fint_Global[self.u_freeDOF]
        Fext_Global_red = self.Fext_Global[self.u_freeDOF]
        G_red = Fext_Global_red - Fint_Global_red
        du_Global_red = np.linalg.solve(K_Global_red, Fext_Global_red)
        self.update_duGlobal(du_Global_red)
        self.u_Global += self.du_Global

        # print(np.max(np.abs(self.Fext_Global-self.Fint_Global)))

        # fname_ux = "data/ux.jpg"
        # fname_uy = "data/uy.jpg"
        # fname_uxy = "data/uxy.jpg"
        # ux, uy = util.splitXY(self.u_Global)
        # uxy = ux**2 + uy**2
        # ux = np.flip(ux.reshape(self.nelx+1, self.nely+1), 0)
        # uy = np.flip(uy.reshape(self.nelx+1, self.nely+1), 0)
        # uxy = np.flip(uxy.reshape(self.nelx+1, self.nely+1), 0)
        # util.saveContour(ux, fname_ux)
        # util.saveContour(uy, fname_uy)
        # util.saveContour(uxy, fname_uxy)

        # print(uy)
        # fname_k = "data/k_matrix.jpg"
        # util.saveContour(self.K_Global, fname_k)
        return self.u_Global

    # ____________________________________________________________________

    def eucl_U(self):
        """Euclidean distance of u_global"""
        ux = self.u_Global[0:self.totelems:3]
        uy = self.u_Global[1:self.totelems:3]
        uz = self.u_Global[2:self.totelems:3]

        return np.sqrt(ux**2 + uy**2 + uz**2)
    # ____________________________________________________________________
    #####################################################################
