import numpy as np
import pytest
from pythonlib import util
from pythonlib.meshGen import Plate2D
from pythonlib import solveFE2D


def test_solve():
    mesh = Plate2D(10,10)
    BC = util.formCond2D(mesh.getDown(), (-1, -1))
    FC = util.formCond2D(mesh.getUp(), (0, 10))
    solve = solveFE2D.solveFE2D(mesh,
                                BC,
                                FC,
                                matlParams=(120, 0.3))
    u = solve.solveProblem()
    u_act = solve.analytical()
    np.testing.assert_equal(u, u_act) 