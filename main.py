from pythonlib import meshGen

nx = 4
ny = 1
nz = 2

mesh = meshGen.Plate3D(nx, ny, nz)

a = mesh.elemNodes
b = mesh.nodeCoords


print(mesh.getLeft())
print(mesh.getRight())
print(mesh.getUp())
print(mesh.getDown())
