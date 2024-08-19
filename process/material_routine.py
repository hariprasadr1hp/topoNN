"""
libMat: Defines the material routine
"""

import numpy as np


class MatGen2D:
    def __init__(self, youngs_mod, nu, eps, del_eps=0) -> None:
        self.youngs_mod = youngs_mod
        self.nu = nu
        self.eps = eps
        self.del_eps = del_eps

    def linear_elactic_isotropic_2d(self):
        """
        Returns Stress and Material Stiffness matrix
        for Linear-Elastic Isotopic Material
        """
        const = self.youngs_mod / ((1 + self.nu) * (1 - (2 * self.nu)))
        a = const * self.nu
        b = const * (1 - self.nu)
        c = const * 0.5 * (1 - 2 * self.nu)
        stiffness_matx = np.array(
            [
                [b, a, 0],
                [a, b, 0],
                [0, 0, c],
            ],
            dtype=float,
        )
        stress_el = stiffness_matx @ self.eps
        return stress_el, stiffness_matx
