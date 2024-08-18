"""
elementRoutine: Defines the characteristics of the element
"""

import numpy as np
from pythonlib.material_routine import MatGen2D


class Quad:
    """
    Defines a first order quad element

    :type gauss_points: int
    :param gauss_points: the number of Gauss Points

    :type quad_rc: the reference coordinates(x,y) for the quad
                    element
    :param quad_rc: ndarray(4 X 2)

    """

    def __init__(self, gauss_points, quad_rc):
        self.gauss_points = gauss_points
        self.quad_rc = quad_rc
        self.grids = 4
        self.axes = 2
        self.components = 3

    def get_natural_coords(self):
        """
        quad-natural Coordinates[xi1, xi2] -> (4 X 2)
        """
        quad_nc = np.zeros((self.grids, self.axes), dtype="int32")
        quad_nc[0] = (-1, -1)
        quad_nc[1] = (+1, -1)
        quad_nc[2] = (-1, +1)
        quad_nc[3] = (+1, +1)
        return quad_nc

    @staticmethod
    def get_shape_func(quad_nc):
        """
        Compute shape functions for an hex8 element -> 4 X 1
        shapefunction = (1/4)*(1 + aa(i))*(1 + bb(i))

        :type quad_nc: numpy.array(float,float) -> (4 X 1)
        :param quad_nc: the natural coordinates of each node of the element
        """
        return 1 / 4 * (1 + quad_nc[:, 0]) * (1 + quad_nc[:, 1])

    def get_dn_natural_matx(self, quad_nc):
        """
        Computes the derivatives of the shape function -> (4 X 2)
        w.r.t its natural coordinates, for the hex8 element

        :type quad_nc: numpy.array(float,float) -> (4 X 2)
        :param quad_nc: the natural coordinates of each node of the element
        """
        dn_natural_matx = np.zeros((self.grids, self.axes))
        dn_natural_matx[:, 0] = 1 / 4 * quad_nc[:, 0]
        dn_natural_matx[:, 1] = 1 / 4 * quad_nc[:, 1]
        # print(dN_nat)
        return dn_natural_matx

    @staticmethod
    def get_jacobian_matrix(dn_nat_matx, quad_rc):
        """
        Computes the Jacobian Matrix  -> (2 X 2)
        J = dx/de = x_i * dN/de

        :type dN_nat: numpy.array(float,float) -> (4 X 2)
        :param dN_nat: the derivates of shape functions w.r.t to its
                        natural coordinates

        :type quad_nc: numpy.array(float,float) -> (4 X 2)
        :param quad_nc: the natural coordinates of each node of the element
        """
        jacobian_matx = quad_rc.T @ dn_nat_matx
        # print()
        # print(quad_rc.T)
        # print()
        # print(Jmat)
        return jacobian_matx

    @staticmethod
    def get_dn_ref_matx(jacobian_matx, dn_nat_matx):
        """
        Computes the derivatives of the shape function -> (4 X 2)
        w.r.t its reference coordinates, for the hex8 element

        :type Jmat: numpy.array(float,float) -> (2 X 2)
        :param Jmat: the Jacobian matrix of the element

        :type dN_nat: numpy.array(float,float) -> (4 X 2)
        :param dN_nat: the derivates of shape functions w.r.t to its
                        natural coordinates
        """
        j_inv = np.linalg.inv(jacobian_matx)
        dn_ref_matx = dn_nat_matx @ j_inv.T
        return dn_ref_matx

    def b_matrix(self, dn_ref_matx):
        """
        Computes the strain-displacement matrix(B) -> (3 X 8)

        :type dN_ref: numpy.array(float,float) -> (4 X 2)
        :param dN_ref: the derivates of shape functions w.r.t to its
                        reference coordinates
        """
        b_matx = np.zeros((self.components, (self.grids * self.axes)))
        for i in range(self.grids):
            # 4 entries
            b_matx[0, (2 * i + 0)] = dn_ref_matx[i, 0]
            b_matx[1, (2 * i + 1)] = dn_ref_matx[i, 1]
            b_matx[2, (2 * i + 0)] = dn_ref_matx[i, 1]
            b_matx[2, (2 * i + 1)] = dn_ref_matx[i, 0]
        return b_matx

    def quad_el(self, matl_params, u_el):
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

        quad_nat_coords = self.get_natural_coords()
        dn_natural_matx = self.get_dn_natural_matx(quad_nat_coords)
        jacobian_matx = self.get_jacobian_matrix(dn_natural_matx, self.quad_rc)
        jacobian_det = np.linalg.det(jacobian_matx)
        dn_ref_matx = self.get_dn_ref_matx(jacobian_matx, dn_natural_matx)
        b_matx = self.b_matrix(dn_ref_matx)

        # strain Matrix
        strain_el = b_matx @ u_el

        # Constitutive Matrix
        matl = MatGen2D(matl_params, strain_el, del_eps=0)
        stress_el, c_mat = matl.linear_elactic_isotropic_2d()

        # K_el Formulation
        b_trans_c_b = b_matx.T @ c_mat @ b_matx
        k_el = jacobian_det * b_trans_c_b
        f_int_el = jacobian_det * b_matx.T @ stress_el
        return k_el, f_int_el
