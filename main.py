import numpy as np
import matplotlib.pyplot as plt
from pythonlib import meshGen
from pythonlib import solveFE2D
from pythonlib import util
from pythonlib import neuralNet

# nx = 1
# ny = 1
# nz = 1

# mesh3D = meshGen.Plate3D(nx, ny, nz)


################################################


nx = 10
ny = 10
mesh2D = meshGen.Plate2D(nx, ny)
elemNodes = mesh2D.elemNodes
nodeCoords = mesh2D.nodeCoords
BC = util.formCond2D(mesh2D.getDown(), (-1, -1))
FC = util.formCond2D(mesh2D.getUp(), (0, 10))

solve = solveFE2D.solveFE2D(mesh2D,
                            BC,
                            FC,
                            (nx, ny),
                            matlParams=(120, 0.3))
# print(solve.Fext_Global)
u = solve.solveProblem()
u_anal, st = solve.analytical()
# print(u)
# print()
# print(u_anal)
# print(u - u_anal)


# svg2 = WriteSvg(fname_u, util.formMagnitude(
#     solve.u_Global, nx, ny
# ))
# svg2.write_doc()


################################################

# Net = neuralNet.neuralNet([2,4,4,1],10)
# # Net.architecture()

# # func = lambda x,y: 3*x**2 + 5*y**3
# func = lambda x,y: x or y
# # func = lambda x,y: x and y
# func = np.vectorize(func)

# y_in = np.array(np.random.rand(12,2))

# cc = []
# for i in range(1000):
#     y_in = np.round(np.random.rand(12,2))
#     # y_target = np.array(np.random.rand(12,1))
#     y_target = func(y_in[:,0],y_in[:,1])
#     y_target = y_target[:,np.newaxis]
#     cost = Net.trainNet(y_in, y_target, 0.0001)
#     cc.append(cost)

# print(cost)
# print()
# # plt.plot([i for i in range(100)],cc)
# # plt.show()

# # inp = np.array(np.random.rand(12,2))
# # y_pred = Net.applyNet(inp)
# # y_target = func(y_in[:,0],y_in[:,1])
# # y_act = y_target[:,np.newaxis]
# # plt.scatter(y_pred,y_act)
# # plt.show()


# # a = np.array([
# #     [1,1],
# #     [2,1],
# #     [4,1],
# #     [6,1],
# #     [8,1],
# #     [10,1],
# #     [12,1],
# #     [14,1],
# #     [16,1],
# #     [18,1],
# #     [110,1],
# #     [121,1],
# # ])
# a = np.array([
#     [0,0],
#     [0,1],
#     [1,0],
#     [1,1],
#     [0,0],
#     [0,1],
#     [1,0],
#     [1,1],
#     [0,0],
#     [0,1],
#     [1,0],
#     [1,1],
# ])

# print(func(a[:,0],a[:,1]))
# print()
# print(Net.applyNet(a))


# ################################################
