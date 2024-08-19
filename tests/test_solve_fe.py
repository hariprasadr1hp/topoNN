import numpy as np

from ingest.generate.mesh import Plate
from process import solve_fe, util


def test_solve():
    mesh = Plate(1, 1)
    boundary_condns = util.formulate_condns(mesh.get_left(), (-1, -1))
    force_condns = util.formulate_condns(mesh.get_right(), (5, 0))
    solve = solve_fe.SolveFE(
        mesh, boundary_condns, force_condns, matl_params=(120, 0.3)
    )
    u = solve.solve_problem()
    u_act = solve.get_analytical_soln()
    np.testing.assert_equal(u, u_act)
