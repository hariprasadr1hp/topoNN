import numpy as np
import matplotlib.pyplot as plt
from pythonlib import meshGen
from pythonlib import solveFE2D
from pythonlib import solveFE3D
from pythonlib import util
from pythonlib.neuralNet import neuralNet


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


# input : 1 parameter
func = lambda x: np.cos(x)
func = lambda x: 3*x**3 + 5*x**2 + 7*x + 8
func = lambda x: 5*x**2 + 7*x + 8
func = lambda x: 7*x + 8
nNet = neuralNet([1,4,4,1])
y_in = np.random.rand(50,1)
nNet.applyNet(y_in)
for i in range(100):
    y_in = np.random.rand(50,1)
    temp = func(y_in)
    # temp = temp[:,np.newaxis]
    nNet.trainNet(y_in, temp, lr=0.001)
    # print(np.shape(nNet.y[-1]))
    # print(np.shape(temp))
    
a = np.random.rand(50,1)
a = np.sort(a, axis=0)
y_target = func(a)
# y_target = y_target[:,np.newaxis]
y_pred = nNet.applyNet(a)

plt.cla()
plt.clf()
plt.scatter(y_pred,np.arange(0,50))
plt.scatter(y_target,np.arange(0,50))
plt.show()



