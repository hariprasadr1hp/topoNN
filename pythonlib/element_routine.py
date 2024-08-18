"""
elementRoutine: Defines the characteristics of the element
"""

import numpy as np
from pythonlib import material_routine as MR


class Brick:
    """
    Defines an Hex-8 element

    :type GP: int
    :param GP: the number of Gauss Points

    :type hex8_rc: the reference coordinates for the hex8 element
    :param hex8_rc: ndarray(8 X 3)

    """

    def __init__(self, GP: int, hex8_rc) -> None:
        self.GP = GP
        self.hex8_rc = hex8_rc
        self.grids = 8
        self.axes = 3
        self.components = 6
        self.Gtab = self.calcQuadrature()

    # ____________________________________________________________________
    # UNDER CONSTRUCTION

    def calcQuadrature(self):
        """
        Returns the Gauss Points table based on the no.of Gauss Points
        """
        self.Gtab = np.zeros(((self.GP**self.axes), 5))
        self.Gtab[0, :] = [1, 0.5773502692, 0.5773502692, 0.5773502692, 1]
        return self.Gtab

    # ____________________________________________________________________

    def natCoords(self):
        """
        hex8-Natural Coordinates[xi1, xi2, xi3] -> (8 X 3)
        """
        hex8_nc = np.zeros((self.grids, self.axes), dtype="int32")
        hex8_nc[0] = (-1, -1, -1)
        hex8_nc[1] = (+1, -1, -1)
        hex8_nc[2] = (-1, +1, -1)
        hex8_nc[3] = (+1, +1, -1)
        hex8_nc[4] = (-1, -1, +1)
        hex8_nc[5] = (+1, -1, +1)
        hex8_nc[6] = (-1, +1, +1)
        hex8_nc[7] = (+1, +1, +1)
        return hex8_nc

    # ____________________________________________________________________

    def shapeFunc(self, hex8_nc, GP):
        """
        Compute shape functions for an hex8 element -> 8 X 1
        shapefunction = (1/8)*(1 + aa(i))*(1 + bb(i))*(1 + cc(i))

        :type hex8_nc: numpy.array(float,float) -> (8 X 3)
        :param hex8_nc: the natural coordinates of each node of the element

        :type GP: int
        :param GP: Gauss point
        """
        N = np.zeros((self.grids))
        N = (
            (1 / 8)
            * (1 + (self.Gtab[GP, 1] * hex8_nc[:, 0]))
            * (1 + (self.Gtab[GP, 2] * hex8_nc[:, 1]))
            * (1 + (self.Gtab[GP, 3] * hex8_nc[:, 2]))
        )
        return N

    # ____________________________________________________________________

    def dN_nat(self, hex8_nc, GP):
        """
        Computes the derivatives of the shape function -> (8 X 3)
        w.r.t its natural coordinates, for the hex8 element

        :type hex8_nc: numpy.array(float,float) -> (8 X 3)
        :param hex8_nc: the natural coordinates of each node of the element

        :type GP: int
        :param GP: Gauss point
        """
        dN_nat = np.zeros((self.grids, self.axes))
        dN_nat[:, 0] = (
            (1 / 8)
            * (hex8_nc[:, 0])
            * (1 + (self.Gtab[GP, 2] * hex8_nc[:, 1]))
            * (1 + (self.Gtab[GP, 3] * hex8_nc[:, 2]))
        )
        dN_nat[:, 1] = (
            (1 / 8)
            * (hex8_nc[:, 1])
            * (1 + (self.Gtab[GP, 1] * hex8_nc[:, 0]))
            * (1 + (self.Gtab[GP, 3] * hex8_nc[:, 2]))
        )
        dN_nat[:, 2] = (
            (1 / 8)
            * (hex8_nc[:, 2])
            * (1 + (self.Gtab[GP, 1] * hex8_nc[:, 0]))
            * (1 + (self.Gtab[GP, 2] * hex8_nc[:, 1]))
        )
        return dN_nat

    # ____________________________________________________________________

    @staticmethod
    def JacobianMat(dN_nat, hex8_rc):
        """
        Computes the Jacobian Matrix  -> (3 X 3)
        J = dx/de = x_i * dN/de

        :type dN_nat: numpy.array(float,float) -> (8 X 3)
        :param dN_nat: the derivates of shape functions w.r.t to its
                        natural coordinates

        :type hex8_nc: numpy.array(float,float) -> (8 X 3)
        :param hex8_nc: the natural coordinates of each node of the element
        """
        Jmat = hex8_rc.T @ dN_nat
        return Jmat

    # ____________________________________________________________________

    @staticmethod
    def dN_ref(Jmat, dN_nat):
        """
        Computes the derivatives of the shape function -> (8 X 3)
        w.r.t its reference coordinates, for the hex8 element

        :type Jmat: numpy.array(float,float) -> (3 X 3)
        :param Jmat: the Jacobian matrix of the element

        :type dN_nat: numpy.array(float,float) -> (8 X 3)
        :param dN_nat: the derivates of shape functions w.r.t to its
                        natural coordinates
        """
        Jinv = np.linalg.inv(Jmat)
        dN_ref = dN_nat @ Jinv.T
        return dN_ref

    # ____________________________________________________________________

    def B_matrix(self, dN_ref):
        """
        Computes the strain-displacement matrix(B) -> (6 X 24)

        :type dN_ref: numpy.array(float,float) -> (8 X 3)
        :param dN_ref: the derivates of shape functions w.r.t to its
                        reference coordinates
        """
        Bmat = np.zeros((self.components, (self.grids * self.axes)))
        for i in range(self.grids):
            # 9 entries
            Bmat[0, (3 * i + 0)] = dN_ref[i, 0]
            Bmat[1, (3 * i + 1)] = dN_ref[i, 1]
            Bmat[2, (3 * i + 2)] = dN_ref[i, 2]
            Bmat[3, (3 * i + 0)] = dN_ref[i, 0]
            Bmat[3, (3 * i + 1)] = dN_ref[i, 1]
            Bmat[4, (3 * i + 1)] = dN_ref[i, 1]
            Bmat[4, (3 * i + 2)] = dN_ref[i, 2]
            Bmat[5, (3 * i + 0)] = dN_ref[i, 0]
            Bmat[5, (3 * i + 2)] = dN_ref[i, 2]
        return Bmat

    # ____________________________________________________________________

    def hex8(self, matlParams, u_el):
        """
        Compute element's local stiffness matrix 'K_el'
        and the internal force vector 'Fint_el'

        K_el     -> 24 X 24
        Fint_el  -> 24 X 1
        u_el     -> 24 X 1
        strain_el->  6 X 1
        stress_el->  6 X 1
        matstiff ->  6 X 6

        :type matlParams: tuple(float) -> (E,Nu,sigY)
        :param matlParams: the material parameters of the element

        :type u_el: numpy.array(float) -> (8 X 1)
        :param u_el: the nodal displacements of the element
        """
        Fint_el = np.zeros((self.grids * self.axes))
        K_el = np.zeros(((self.grids * self.axes), (self.grids * self.axes)))

        for GP in range(self.GP**self.axes):
            hex8_nc = self.natCoords()
            N = self.shapeFunc(hex8_nc, GP)
            dN_nat = self.dN_nat(hex8_nc, GP)
            Jmat = self.JacobianMat(dN_nat, self.hex8_rc)
            detJ = np.linalg.det(Jmat)
            dN_ref = self.dN_ref(Jmat, dN_nat)
            Bmat = self.B_matrix(dN_ref)

            # strain Matrix
            strain_el = Bmat @ u_el

            # Constitutive Matrix
            matl = MR.MatGen3D(matlParams, strain_el, del_eps=0)
            stress_el, Cmat = matl.LEIsotropic3D()

            # K_el Formulation
            BtCB = Bmat.T @ Cmat @ Bmat
            K_el_GP = (self.Gtab[GP, 4] * detJ) * BtCB
            Fint_el_GP = (self.Gtab[GP, 4] * detJ) * Bmat.T @ stress_el
            K_el += K_el_GP
            Fint_el += Fint_el_GP

        return K_el, Fint_el


########################################################################


class Quad:
    """
    Defines a first order quad element

    :type GP: int
    :param GP: the number of Gauss Points

    :type hex8_rc: the reference coordinates(x,y) for the quad
                    element
    :param hex8_rc: ndarray(4 X 2)

    """

    def __init__(self, GP, quad_rc):
        self.GP = GP
        self.quad_rc = quad_rc
        self.grids = 4
        self.axes = 2
        self.components = 3

    # ____________________________________________________________________

    def natCoords(self):
        """
        quad-Natural Coordinates[xi1, xi2] -> (4 X 2)
        """
        quad_nc = np.zeros((self.grids, self.axes), dtype="int32")
        quad_nc[0] = (-1, -1)
        quad_nc[1] = (+1, -1)
        quad_nc[2] = (-1, +1)
        quad_nc[3] = (+1, +1)
        return quad_nc

    # ____________________________________________________________________

    @staticmethod
    def shapeFunc(self, quad_nc):
        """
        Compute shape functions for an hex8 element -> 4 X 1
        shapefunction = (1/4)*(1 + aa(i))*(1 + bb(i))

        :type quad_nc: numpy.array(float,float) -> (4 X 1)
        :param quad_nc: the natural coordinates of each node of the element
        """
        N = 1 / 4 * (1 + quad_nc[:, 0]) * (1 + quad_nc[:, 1])
        return N

    # ____________________________________________________________________

    def dN_nat(self, quad_nc):
        """
        Computes the derivatives of the shape function -> (4 X 2)
        w.r.t its natural coordinates, for the hex8 element

        :type quad_nc: numpy.array(float,float) -> (4 X 2)
        :param quad_nc: the natural coordinates of each node of the element
        """
        dN_nat = np.zeros((self.grids, self.axes))
        dN_nat[:, 0] = 1 / 4 * quad_nc[:, 0]
        dN_nat[:, 1] = 1 / 4 * quad_nc[:, 1]
        # print(dN_nat)
        return dN_nat

    # ____________________________________________________________________

    @staticmethod
    def JacobianMat(dN_nat, quad_rc):
        """
        Computes the Jacobian Matrix  -> (2 X 2)
        J = dx/de = x_i * dN/de

        :type dN_nat: numpy.array(float,float) -> (4 X 2)
        :param dN_nat: the derivates of shape functions w.r.t to its
                        natural coordinates

        :type quad_nc: numpy.array(float,float) -> (4 X 2)
        :param quad_nc: the natural coordinates of each node of the element
        """
        Jmat = quad_rc.T @ dN_nat
        # print()
        # print(quad_rc.T)
        # print()
        # print(Jmat)
        return Jmat

    # ____________________________________________________________________

    @staticmethod
    def dN_ref(Jmat, dN_nat):
        """
        Computes the derivatives of the shape function -> (4 X 2)
        w.r.t its reference coordinates, for the hex8 element

        :type Jmat: numpy.array(float,float) -> (2 X 2)
        :param Jmat: the Jacobian matrix of the element

        :type dN_nat: numpy.array(float,float) -> (4 X 2)
        :param dN_nat: the derivates of shape functions w.r.t to its
                        natural coordinates
        """
        j_inv = np.linalg.inv(Jmat)
        dN_ref = dN_nat @ j_inv.T
        return dN_ref

    # ____________________________________________________________________

    def b_matrix(self, dN_ref):
        """
        Computes the strain-displacement matrix(B) -> (3 X 8)

        :type dN_ref: numpy.array(float,float) -> (4 X 2)
        :param dN_ref: the derivates of shape functions w.r.t to its
                        reference coordinates
        """
        Bmat = np.zeros((self.components, (self.grids * self.axes)))
        for i in range(self.grids):
            # 4 entries
            Bmat[0, (2 * i + 0)] = dN_ref[i, 0]
            Bmat[1, (2 * i + 1)] = dN_ref[i, 1]
            Bmat[2, (2 * i + 0)] = dN_ref[i, 1]
            Bmat[2, (2 * i + 1)] = dN_ref[i, 0]
        return Bmat

    # ____________________________________________________________________

    def quad_el(self, matlParams, u_el):
        """
        Compute element's local stiffness matrix 'K_el'
        and the internal force vector 'Fint_el'

        K_el     -> 8 X 8
        Fint_el -> 8 X 1
        u_el     -> 8 X 1
        strain_el->  3 X 1
        stress_el->  3 X 1
        matstiff ->  3 X 3

        :type matlParams: tuple(float) -> (E,Nu,sigY)
        :param matlParams: the material parameters of the element

        :type u_el: numpy.array(float) -> (8 X 1)
        :param u_el: the nodal displacements of the element
        """
        f_int_el = np.zeros((self.grids * self.axes))
        k_el = np.zeros(((self.grids * self.axes), (self.grids * self.axes)))

        quad_nc = self.natCoords()
        dN_nat = self.dN_nat(quad_nc)
        j_mat = self.JacobianMat(dN_nat, self.quad_rc)
        j_det = np.linalg.det(j_mat)
        dN_ref = self.dN_ref(j_mat, dN_nat)
        b_mat = self.b_matrix(dN_ref)

        # strain Matrix
        strain_el = b_mat @ u_el

        # Constitutive Matrix
        matl = MR.matGen2D(matlParams, strain_el, del_eps=0)
        stress_el, c_mat = matl.LEIsotropic2D()

        # K_el Formulation
        BtCB = b_mat.T @ c_mat @ b_mat
        k_el = j_det * BtCB
        f_int_el = j_det * b_mat.T @ stress_el
        return k_el, f_int_el
