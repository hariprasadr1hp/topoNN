import numpy as np
from icecream import ic

from ingest.generate.mesh import Plate
from process.solve_fe import SolveFE
from process.util import formulate_condns

# from pythonlib.neural_net import NeuralNet


ELEMS_X = 2
ELEMS_Y = 1
mesh = Plate(ELEMS_X, ELEMS_Y)

elem_nodes = mesh.elem_nodes
node_coords = mesh.node_coords
boundary_condns = formulate_condns(mesh.get_left(), (-1, -1))
force_condns = formulate_condns(mesh.get_right(), (5, 0))
matl_params = (120, 0.3)

# ic(mesh.elem_nodes)
# ic(mesh.node_coords)

solve = SolveFE(
    mesh=mesh,
    boundary_condns=boundary_condns,
    force_condns=force_condns,
    matl_params=matl_params,
)
u = solve.solve_problem()
u_act = solve.get_analytical_soln()
ic(np.shape(u))
ic(np.shape(u_act))


################################################


# Net = neuralNet.neuralNet([2,7,7,1],10)
# # Net.architecture()

# func = lambda x,y: 3*x**2 + 5*y**3
# # func = lambda x,y: np.sin(x) + np.sin(y)


# func = lambda x,y: 3*x + 5*y
# func = np.vectorize(func)
# nNet = neuralNet([2,5,5,1])
# y_in = np.random.rand(50,2)
# nNet.applyNet(y_in)
# for i in range(100):
#     y_in = np.random.rand(50,2)
#     temp = func(y_in[:,0], y_in[:,1])
#     temp = temp[:,np.newaxis]
#     nNet.trainNet(y_in, temp, lr=0.001)
#     # print(np.shape(nNet.y[-1]))
#     # print(np.shape(temp))

# a = np.random.rand(50,2)
# y_target = func(a[:,0], a[:,1])
# y_target = y_target[:,np.newaxis]
# y_pred = nNet.applyNet(a)

# plt.cla()
# plt.clf()
# plt.scatter(np.arange(0,50),y_pred)
# plt.scatter(np.arange(0,50),y_target)
# plt.show()


################################################
# input : 1 parameter
# func = lambda x: 3*x**3 + 5*x**2 + 7*x + 8
# func = lambda x: 7*x + 8
# nNet = neuralNet([1,4,4,1])
# y_in = np.random.rand(50,1)
# nNet.applyNet(y_in)
# for i in range(100):
#     y_in = np.random.rand(50,1)
#     temp = func(y_in)
#     nNet.trainNet(y_in, temp, lr=0.001)

# a = np.random.rand(50,1)
# a = np.sort(a, axis=0)
# y_target = func(a)
# y_pred = nNet.applyNet(a)

# plt.cla()
# plt.clf()
# plt.scatter(y_pred,np.arange(0,50), color='#00ff00')
# plt.scatter(y_target,np.arange(0,50), color="#ff0000")
# plt.show()
