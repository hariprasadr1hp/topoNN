from pythonlib import util
from pythonlib.mesh_generate import Plate3D
from pythonlib.solvefe_3d import SolveFE3D


def test_solve():
    mesh = Plate3D(10, 10, 10)
    boundary_condns = util.formulate_3d_condns(mesh.get_down(), (-1, -1 - 1))
    force_condns = util.formulate_3d_condns(mesh.get_up(), (10, 10, 10))
    solve = SolveFE3D(mesh, boundary_condns, force_condns, matl_params=(120, 0.3))
    u = solve.solve_problem()
    # u_act = solve.get_analytical_soln()
    # np.testing.assert_equal(u, u_act)
