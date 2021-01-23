"""
This module will solve the 2D-FE problem
"""
import numpy as np
from pythonlib.meshGen import Plate2D
from pythonlib.util import WriteSvg
from pythonlib import elementRoutine as ER


class solveFE2D:
    def __init__(self, nodeCoords, elemNodes, BC, FC, matlParams=(120, 0.3)):
        self.nodeCoords = nodeCoords
        self.elemNodes = elemNodes
        self.matlParams = matlParams
        self.BC = BC
        self.FC = FC

        self.totnodes = np.shape(self.nodeCoords)[0]
        self.totelems = np.shape(self.elemNodes)[0]
        self.totDOFs = self.totnodes * 2
        self.elemDOF = 8

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

    # ____________________________________________________________________

    # @staticmethod
    def get_dofIDs(self, NodeIDs):
        """
        Returns the [x,y] IDs for the given Node IDs

        :type NodeIDs: ndarray
        :param NodeIDs: the ID of the node
        """
        def xy(x):
            return [2*x-2, 2*x-1]
        NodeDOFs = np.concatenate([xy(i) for i in NodeIDs])
        return NodeDOFs

    # ____________________________________________________________________

    def forceSet(self, fc) -> None:
        """
        Sets the force conditions

        :type fc: nd.array(N X 3)
        :param fc: the node IDs along with their force constraints
                    in the x,y direction
        """
        for each in fc:
            self.Fext_Global[int(each[0])*2-2] = each[1]
            self.Fext_Global[int(each[0])*2-1] = each[2]
    # ____________________________________________________________________

    # def boundarySet(self,U_ID,vals):
    def boundarySet(self, bc) -> None:
        """
        Sets the boundary conditions

        :type bc: nd.array(N X 3)
        :param bc: the node IDs along with their displacements
                        constraints in the x,y direction
        """
        for each in bc:
            if each[1] == -1:
                self.u_fixedDOF[int(each[0])*2-2] = True
            if each[2] == -1:
                self.u_fixedDOF[int(each[0])*2-1] = True
            # else:
            #     pass
        self.u_freeDOF = ~self.u_fixedDOF

    # ___________________________________________________________________

    @staticmethod
    def strainEnergy(u_el, K_el) -> float:
        """
        Calculates the starin energy density of an element

        :type u_el: ndarray (8,2)
        :param u_el: the element displacement

        :type K_el: ndarray (8,8)
        :param K_el: the element stiffness matrix
        """
        return u_el.T @ K_el @ u_el

    # ___________________________________________________________________

    def fillMatrix(self, elemID: int, K_el, Fint_el) -> None:
        """
        Fills the sparse Global stiffness matrix with the local
        stifnesses

        :type K_el: ndarray (8,8)
        :param K_el: the stiffness matrix of an element

        :type Fint_el: ndarray (8,1)
        :param Fint_el: the internal force matrix of an element
        """
        def xy(x):
            return [2*x-2, 2*x-1]
        DOFs = np.concatenate([xy(i) for i in self.elemNodes[elemID-1]])
        for i, j in enumerate(DOFs):
            self.K_Global[j, j] += K_el[i, i]
            self.Fint_Global[j] += Fint_el[i]

    # ____________________________________________________________________

    def assemblyGlobal(self) -> None:
        """
        Assembly of the global Force and Stiffness matrices
        """
        quad_el = np.zeros((4, 2), dtype=float)
        u_el = np.zeros((8), dtype=float)
        strain_volume = np.zeros((self.totelems,1))

        for i, elemID in enumerate(self.elemNodes):
            for j, nodeID in enumerate(elemID):
                quad_el[j] = self.nodeCoords[nodeID-1]
            u_el = self.get_dofIDs(self.elemNodes[i])
            element = ER.Quad(GP=1, quad_rc=quad_el)
            K_el, Fint_el = element.quad_el(self.matlParams, u_el)
        
            strain_volume[i] = self.strainEnergy(u_el,K_el)
            self.fillMatrix(i+1, K_el, Fint_el)
        fname_uku = "data/uku_{}".format(i)
        # o = strain_volume.reshape(10,10)
        # svg = WriteSvg(fname_uku, o)
        # print(o)
        # svg.write_doc()

    # ________________________________________________

    def matlSet(self, matlParams) -> None:
        """
        Sets the material Parameters

        :type matlParams: tuple(float) -> (E,Nu,sigY)
        :param matlParams: the material parameters of the element
        """
        self.matlParams = matlParams
    # ____________________________________________________________________

    def update_duGlobal(self, du_Global_red) -> None:
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
        for step in range(4):
            self.assemblyGlobal()
            K_Global_red = self.K_Global[self.u_freeDOF]
            K_Global_red = (K_Global_red.T[self.u_freeDOF]).T
            Fint_Global_red = self.Fint_Global[self.u_freeDOF]
            Fext_Global_red = self.Fext_Global[self.u_freeDOF]
            G_red = Fext_Global_red - Fint_Global_red

            du_Global_red = np.linalg.solve(K_Global_red, G_red)
            self.update_duGlobal(du_Global_red)
            self.u_Global += self.du_Global
            # print(np.max(np.abs(self.Fext_Global-self.Fint_Global)))
            # print(np.linalg.det(K_Global_red))
            fname_k = "data/u_{}".format(step)
            svg1 = WriteSvg(fname_k, self.K_Global)
            svg1.write_doc()
        return self.u_Global

    # ____________________________________________________________________

    def eucl_U(self):
        """Euclidean distance of u_global"""
        ux = self.u_Global[0:self.totelems:3]
        uy = self.u_Global[1:self.totelems:3]
        uz = self.u_Global[2:self.totelems:3]

        return np.sqrt(ux**2 + uy**2 + uz**2)
    # ____________________________________________________________________
    
    def analytical(self):
        stiff = self.matlParams[0] * np.array([
            [+2, +0, -1, +0, -1, +0, +0, +0],
            [+0, +2, +0, -1, +0, -1, +0, +0],
            [-1, +0, +2, +0, +0, +0, -1, +0],
            [+0, -1, +0, +2, +0, +0, +0, -1],
            [-1, +0, +0, +0, +2, +0, -1, +0],
            [+0, -1, +0, +0, +0, +2, +0, -1],
            [+0, +0, -1, +0, -1, +0, +2, +0],
            [+0, +0, +0, -1, +0, -1, +0, +2],
        ])
        stiff_Global = np.zeros((self.totDOFs,self.totDOFs))
        u_anal = np.zeros((self.totDOFs))
        def xy(x):
            return [2*x-2, 2*x-1]
        for k, elemID in enumerate(self.elemNodes):
            DOFs = np.concatenate([xy(l) for l in self.elemNodes[k]])
            for i, j in enumerate(DOFs):
                stiff_Global[j, j] += stiff[i, i]
        stiff_red = stiff_Global[self.u_freeDOF]
        stiff_red = (stiff_red.T[self.u_freeDOF]).T
        Fext_Global_red = self.Fext_Global[self.u_freeDOF]
        # print(np.linalg.det(stiff))
        # print(np.linalg.det(stiff_red))
        u_anal_red = np.linalg.solve(stiff_red, Fext_Global_red)
        count = 0
        while count < np.size(u_anal_red):
            for each, _ in enumerate(self.u_freeDOF):
                if self.u_freeDOF[each]:
                    u_anal[each] = u_anal_red[count]
                    count += 1
        return u_anal,stiff_Global


    #####################################################################
