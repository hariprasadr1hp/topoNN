import numpy as np
import matplotlib.pyplot as plt
from pythonlib import meshGen
from pythonlib import solveFE2D
from pythonlib import solveFE3D
from pythonlib import util
from pythonlib import neuralNet


# 3D-case
# nx = 1
# ny = 1
# nz = 1
# mesh3D = meshGen.Plate3D(nx, ny, nz)
# elemNodes = mesh3D.elemNodes
# nodeCoords = mesh3D.nodeCoords
# BC = util.formCond3D(mesh3D.getDown(), (-1, 0, -1))
# FC = util.formCond3D(mesh3D.getUp(), (10, 0, 10))

# solve = solveFE3D.solveFE3D(mesh3D,
#                             BC,
#                             FC,
#                             matlParams=(120, 0.3))
# u = solve.solveProblem()
# print(u)

################################################

# 2D-case
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
                            matlParams=(120, 0.3))
u = solve.solveProblem()




################################################

# Net = neuralNet.neuralNet([2,7,7,1],10)
# # Net.architecture()

# # func = lambda x,y: 3*x**2 + 5*y**3
# # func = lambda x,y: np.sin(x) + np.sin(y)
# func = lambda x,y: 3*x + 5*y
# # func = lambda x,y: x or y
# # func = lambda x,y: x and y
# func = np.vectorize(func)

# y_in = np.array(np.random.rand(12,2))

# cc = []
# for i in range(1000):
#     y_in = np.round(np.random.rand(12,2))
#     # y_target = np.array(np.random.rand(12,1))
#     y_target = func(y_in[:,0],y_in[:,1])
#     y_target = y_target[:,np.newaxis]
#     cost = Net.trainNet(y_in, y_target, 0.00001)
#     cc.append(cost)

# print(cost)
# print()
# plt.plot([i for i in range(1000)],cc)
# plt.show()

# inp = np.array(np.random.rand(12,2))
# y_pred = Net.applyNet(inp)
# y_target = func(inp[:,0],inp[:,1])
# y_act = y_target
# # y_act = y_target[:,np.newaxis]
# print(y_target)
# print(y_pred)
# print(Net.applyNet(np.array([1,1])))
# print(Net.applyNet(np.array([2,1])))
# print(Net.applyNet(np.array([3,1])))
# print(Net.applyNet(np.array([4,1])))
# print(Net.applyNet(np.array([5,1])))
# print(Net.applyNet(np.array([6,1])))
# # plt.scatter(y_pred,[i for i in range(12)], label='y_pred')
# plt.scatter(y_act,np.arange(0,12), label='y_act', color="#00ff00")
# plt.scatter(y_pred, np.arange(0,12), label='y_pred', color="#ff0000")
# plt.show()


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
