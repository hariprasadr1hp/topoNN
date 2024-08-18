import numpy as np
from pythonlib import util
from pythonlib.mesh_generate import Plate2D
from pythonlib import solvefe_2d


def test_solve():
    mesh = Plate2D(1, 1)
    boundary_condns = util.formulate_2d_condns(mesh.get_left(), (-1, -1))
    force_condns = util.formulate_2d_condns(mesh.get_right(), (5, 0))
    solve = solvefe_2d.SolveFE2D(
        mesh, boundary_condns, force_condns, matl_params=(120, 0.3)
    )
    u = solve.solve_problem()
    u_act = solve.get_analytical_soln()
    np.testing.assert_equal(u, u_act)
