import numpy as np
import pytest
from pythonlib import util
from pythonlib.meshGen import Plate3D
from pythonlib.solveFE3D import solveFE3D


def test_solve():
    mesh = Plate3D(10,10, 10)
    BC = util.formCond3D(mesh.getDown(), (-1, -1 -1))
    FC = util.formCond3D(mesh.getUp(), (10, 10, 10))
    solve = solveFE3D(mesh,
                        BC,
                        FC,
                        matlParams=(120, 0.3))
    u = solve.solveProblem()
    u_act = solve.analytical()
    np.testing.assert_equal(u, u_act) 