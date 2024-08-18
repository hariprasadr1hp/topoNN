"""
libMat: Defines the material routine
"""

import numpy as np


class MatGen3D:
    """
    Defines the material routine

    """

    def __init__(self, matlParams: tuple, eps, del_eps=0) -> None:
        self.ymod = matlParams[0]
        self.Nu = matlParams[1]
        self.eps = eps
        self.del_eps = del_eps

    def LEIsotropic3D(self):
        """
        Returns Stress and Material Stiffness matrix
        for Linear-Elastic Isotopic Material
        """
        const = self.ymod / ((1 + self.Nu) * (1 - (2 * self.Nu)))
        a = const * self.Nu
        b = const * (1 - self.Nu)
        c = (a - b) / 2
        Cmat = np.array(
            [
                [b, a, a, 0, 0, 0],
                [a, b, a, 0, 0, 0],
                [a, a, b, 0, 0, 0],
                [0, 0, 0, c, 0, 0],
                [0, 0, 0, 0, c, 0],
                [0, 0, 0, 0, 0, c],
            ],
            dtype=float,
        )
        stress_el = Cmat @ self.eps
        return stress_el, Cmat


class matGen2D:
    def __init__(self, matlParams: tuple, eps, del_eps=0) -> None:
        self.ymod = matlParams[0]
        self.Nu = matlParams[1]
        self.eps = eps
        self.del_eps = del_eps

    def LEIsotropic2D(self):
        """
        Returns Stress and Material Stiffness matrix
        for Linear-Elastic Isotopic Material
        """
        const = self.ymod / ((1 + self.Nu) * (1 - (2 * self.Nu)))
        a = const * self.Nu
        b = const * (1 - self.Nu)
        c = const * 0.5 * (1 - 2 * self.Nu)
        Cmat = np.array(
            [
                [b, a, 0],
                [a, b, 0],
                [0, 0, c],
            ],
            dtype=float,
        )
        stress_el = Cmat @ self.eps
        return stress_el, Cmat
