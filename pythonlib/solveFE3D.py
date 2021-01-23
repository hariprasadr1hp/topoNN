"""
This module will solve the FE problem
"""
import numpy as np
from scipy.sparse import coo_matrix
from pythonlib import elementRoutine as ER


class solveFE3D:
    def __init__(self, nodeCoords, elemNodes, BC, FC, matlParams=(120,0.3,5)):
        self.nodeCoords = nodeCoords
        self.elemNodes  = elemNodes
        self.matlParams = matlParams
        self.BC = BC
        self.FC = FC

        self.totnodes   = np.shape(self.nodeCoords)[0]
        self.totelems   = np.shape(self.elemNodes)[0]
        self.totDOFs    = self.totnodes * 3
        self.elemDOF    = 24


        #Initialize Matrices
        self.Fint_Global = np.zeros((self.totDOFs))
        self.Fext_Global = np.zeros((self.totDOFs))
        self.K_Global = coo_matrix((self.totDOFs, self.totDOFs),dtype=float)
        self.u_Global = np.zeros((self.totDOFs))
        self.du_Global = np.zeros((self.totDOFs))

    #____________________________________________________________________

    def assemblyGlobal(self):
        """
        Assembly of the global Force and Stiffness matrices
        """
        hex8_el = np.zeros((8,3), dtype=float)
        u_el =  np.zeros((8), dtype=float)

        for i, elemID in enumerate(self.elemNodes):
            for j, nodeID in enumerate(elemID):
                hex8_el[j] = self.nodeCoords[nodeID]
                element = ER.Brick(1,hex8_el)
                K_el, Fint_el = element.hex8(self.matlParams, u_el)
    #___________________________________________________________________

    def fillMatrix(self, K_el):
        """
        Fills the sparse Global stiffness matrix with the local 
        stifnesses

        :type K_el: ndarray (24,24)
        :param K_el: the local stiffness matrix of an element
        """
 
        for i, u in enumerate(K_el):
            for j, v in enumerate(u):
                pass
    #___________________________________________________________________

    def matlSet(self,matlParams):
        """
        Sets the material Parameters

        :type matlParams: tuple(float) -> (E,Nu,sigY)
        :param matlParams: the material parameters of the element
        """
        self.matlParams = matlParams
    #____________________________________________________________________

    def forceSet(self,F_ID,vals):
        """
        Sets the force conditions

        :type F_ID: numpy.array(int)
        :param F_ID: the node IDs for which the force conditions
                        has to be set

        :type vals: tuple(float) -> (x,y,z)
        :param vals: the force values in each direction for the ID
        """
        # for i in F_ID:
        #     for j,val in enumerate(vals):
        #         self.Fext_Global[3*(i-1)-2+j] = val
    #____________________________________________________________________

    def boundarySet(self,U_ID,vals):
        """
        Sets the boundary conditions

        :type U_ID: numpy.array(int)
        :param U_ID: the node IDs for which the force conditions
                        has to be set

        :type vals: tuple(float) -> (x,y,z)
        :param vals: the displacement values in each direction for the ID
        """
        # for i in U_ID:
        #     self.maskID[i-1] = False
        #     for j,val in enumerate(vals):
        #         if val == -1:
        #             self.maskDOF[3*i+j-3] = False
        # self.u_fixedID = self.GIDS[~self.maskID]
        # self.u_freeID = self.GIDS[self.maskID]
        # self.u_fixedDOF = self.GDOFS[~self.maskDOF]
        # self.u_freeDOF = self.GDOFS[self.maskDOF]
    #____________________________________________________________________

    def update_duGlobal(self,du_Global_red):
        """
        Updates the global displacement vector after each iteration

        :type du_Global_red: numpy.array(int)
        :param du_Global_red: the nodal displacements of IDs with
                                non-zero displacement
        """
        # for i in self.u_freeDOF:
        #     for j in du_Global_red:
        #         self.du_Global[i-1] = j
    #____________________________________________________________________

    def solveProblem(self):
        """
        Solves the formulated FE Problem
        """
        # for _ in range(4):
        #     self.assemblyGlobal()
        #     # print(np.shape(self.K_Global))
        #     # print(np.linalg.det(self.K_Global))
        #     K_Global_red = np.delete(self.K_Global,
        #                              self.u_fixedDOF-1, axis=0)
        #     K_Global_red = np.delete(K_Global_red,
        #                              self.u_fixedDOF-1, axis=1)
        #     # print(np.shape(K_Global_red))
        #     # print(np.linalg.det(K_Global_red))
        #     Fint_Global_red = np.delete(self.Fint_Global, self.u_fixedDOF-1)
        #     Fext_Global_red = np.delete(self.Fext_Global, self.u_fixedDOF-1)
        #     G_red =  Fext_Global_red - Fint_Global_red

        #     du_Global_red = np.linalg.solve(K_Global_red,G_red)
        #     self.update_duGlobal(du_Global_red)

        #     self.u_Global += self.du_Global

        #     # print(self.u_Global)
        #     # print(G_red)
        #     # print(self.u_fixedDOF-1)
        #     print(np.max(np.abs(self.Fext_Global-self.Fint_Global)))
        #     # np.savetxt("data/csv/foo.csv", self.K_Global, delimiter=",")
        #     self.saveResults()
        # return self.u_Global

    #____________________________________________________________________

    def eucl_U(self):
        """Euclidean distance of u_global"""
        ux = self.u_Global[0:self.totelems:3]
        uy = self.u_Global[1:self.totelems:3]
        uz = self.u_Global[2:self.totelems:3]

        return np.sqrt(ux**2 + uy**2 + uz**2)
    #____________________________________________________________________
    #####################################################################
