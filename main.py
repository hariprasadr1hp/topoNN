import numpy as np
from pythonlib import mesh_generate
from pythonlib import solvefe_2d
from pythonlib import util

# from pythonlib.neural_net import NeuralNet


NODES_X = 3
NODES_Y = 3
mesh2D = mesh_generate.Plate2D(NODES_X, NODES_Y)
elem_nodes = mesh2D.elem_nodes
node_coords = mesh2D.node_coords
boundary_condns = util.formulate_2d_condns(mesh2D.get_left(), (-1, -1))
force_condns = util.formulate_2d_condns(mesh2D.get_right(), (5, 0))

solve = solvefe_2d.SolveFE2D(
    mesh2D, boundary_condns, force_condns, matl_params=(120, 0.3)
)
u = solve.solve_problem()
u_act = solve.get_analytical_soln()
print(np.shape(u))
print(np.shape(u_act))


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
