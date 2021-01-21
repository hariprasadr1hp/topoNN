import numpy as np
from pythonlib import meshGen
from pythonlib import solveFE2D
from pythonlib import util

# nx = 4
# ny = 1
# nz = 2

# mesh3D = meshGen.Plate3D(nx, ny, nz)


nx = 10
ny = 2
mesh2D = meshGen.Plate2D(nx, ny)
BC = util.formCond2D(mesh2D.getLeft(), (-1,0))
FC = util.formCond2D(mesh2D.getRight(), (5,0))
elemNodes = mesh2D.elemNodes
nodeCoords = mesh2D.nodeCoords

solve = solveFE2D.solveFE2D(nodeCoords,elemNodes,BC,FC)
solve.solveProblem()
