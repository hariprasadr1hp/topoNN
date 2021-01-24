"""
This module generates training data
"""
import numpy as np
import h5py
from pythonlib.meshGen import Plate2D
from pythonlib.solveFE2D import solveFE2D
from pythonlib import util


class DataGen:
    """
    Generates Data for structural optimization

    :type mesh: class
    :param nelx: Generates the plate Geometry for the given x,y elements

    :type amount: int
    :param amount: volume of the dataset
    """

    def __init__(self, mesh: int, amount: int) -> None:
        self.mesh = mesh
        self.nelx = mesh.nelx
        self.nely = mesh.nely
        self.amount = amount
        self.fname = "topoNN.h5"
        self.generateData()

    def generateForces(self, HDFfile):
        """
        Generating volume strain energy density data for a specific
        pattern of boundary conditions by only altering the magnitudes
        """
        def randVal100() -> int:
            """Generates a random value from 1 to 100"""
            return np.round(100*np.random.rand())
        #########################################################
        # CASE 1: constrained: BOTTOM, FORCE: TOP
        #########################################################
        with h5py.File(self.fname, 'w') as f:
            g1 = f.create_group('T1B2L0R0')
            BC = util.formCond2D(self.mesh.getDown(), (-1, -1))
            for i in range(1000):
                FC = util.formCond2D(self.mesh.getUp(),
                                     (randVal100(), randVal100()))
                strainDensity = solveFE2D(self.mesh, BC, FC)
                g1.create_dataset("config_{i}", data=strainDensity)
            #########################################################
            # CASE 2: constrained: TOP, FORCE: BOTTOM
            #########################################################
            g2 = f.create_group('T2B1L0R0')
            BC = util.formCond2D(self.mesh.getUp(), (-1, -1))
            for i in range(1000):
                FC = util.formCond2D(self.mesh.getDown(),
                                     (randVal100(), randVal100()))
                strainDensity = solveFE2D(self.mesh, BC, FC)
                g2.create_dataset("config_{i}", data=strainDensity)
            #########################################################
            # CASE 3: constrained: LEFT, FORCE: RIGHT
            #########################################################
            g3 = f.create_group('T0B0L1R2')
            BC = util.formCond2D(self.mesh.getLeft(), (-1, -1))
            for i in range(1000):
                FC = util.formCond2D(self.mesh.getRight(),
                                     (randVal100(), randVal100()))
                strainDensity = solveFE2D(self.mesh, BC, FC)
                g3.create_dataset("config_{i}", data=strainDensity)
            #########################################################
            # CASE 4: constrained: RIGHT, FORCE: LEFT
            #########################################################
            g4 = f.create_group('T0B0L2R1')
            BC = util.formCond2D(self.mesh.getRight(), (-1, -1))
            for i in range(1000):
                FC = util.formCond2D(self.mesh.getLeft(),
                                     (randVal100(), randVal100()))
                strainDensity = solveFE2D(self.mesh, BC, FC)
                g4.create_dataset("config_{i}", data=strainDensity)
            #########################################################
            # CASE 5: constrained: BOTTOM, FORCE: LEFT
            #########################################################
            g5 = f.create_group('T0B1L2R0')
            BC = util.formCond2D(self.mesh.getDown(), (-1, -1))
            for i in range(1000):
                FC = util.formCond2D(self.mesh.getLeft(),
                                     (randVal100(), randVal100()))
                strainDensity = solveFE2D(self.mesh, BC, FC)
                g5.create_dataset("config_{i}", data=strainDensity)
            #########################################################
            # CASE 6: constrained: BOTTOM, FORCE: RIGHT
            #########################################################
            g6 = f.create_group('T0B1L0R2')
            BC = util.formCond2D(self.mesh.getDown(), (-1, -1))
            for i in range(1000):
                FC = util.formCond2D(self.mesh.getRight(),
                                     (randVal100(), randVal100()))
                strainDensity = solveFE2D(self.mesh, BC, FC)
                g6.create_dataset("config_{i}", data=strainDensity)
            #########################################################
            # CASE 7: constrained: TOP, FORCE: LEFT
            #########################################################
            g7 = f.create_group('T1B0L2R0')
            BC = util.formCond2D(self.mesh.getUp(), (-1, -1))
            for i in range(1000):
                FC = util.formCond2D(self.mesh.getLeft(),
                                     (randVal100(), randVal100()))
                strainDensity = solveFE2D(self.mesh, BC, FC)
                g7.create_dataset("config_{i}", data=strainDensity)
            #########################################################
            # CASE 8: constrained: TOP, FORCE: RIGHT
            #########################################################
            g8 = f.create_group('T1B0L0R2')
            BC = util.formCond2D(self.mesh.getUp(), (-1, -1))
            for i in range(1000):
                FC = util.formCond2D(self.mesh.getRight(),
                                     (randVal100(), randVal100()))
                strainDensity = solveFE2D(self.mesh, BC, FC)
                g8.create_dataset("config_{i}", data=strainDensity)
            #########################################################
            # CASE 9: constrained: LEFT, FORCE: TOP
            #########################################################
            g9 = f.create_group('T2B0L1R0')
            BC = util.formCond2D(self.mesh.getLeft(), (-1, -1))
            for i in range(1000):
                FC = util.formCond2D(self.mesh.getUp(),
                                     (randVal100(), randVal100()))
                strainDensity = solveFE2D(self.mesh, BC, FC)
                g9.create_dataset("config_{i}", data=strainDensity)
            #########################################################
            # CASE 10: constrained: LEFT, FORCE: BOTTOM
            #########################################################
            g10 = f.create_group('T0B2L1R0')
            BC = util.formCond2D(self.mesh.getLeft(), (-1, -1))
            for i in range(1000):
                FC = util.formCond2D(self.mesh.getDown(),
                                     (randVal100(), randVal100()))
                strainDensity = solveFE2D(self.mesh, BC, FC)
                g10.create_dataset("config_{i}", data=strainDensity)
            #########################################################
            # CASE 11: constrained: RIGHT, FORCE: TOP
            #########################################################
            g11 = f.create_group('T2B0L0R1')
            BC = util.formCond2D(self.mesh.getRight(), (-1, -1))
            for i in range(1000):
                FC = util.formCond2D(self.mesh.getUp(),
                                     (randVal100(), randVal100()))
                strainDensity = solveFE2D(self.mesh, BC, FC)
                g11.create_dataset("config_{i}", data=strainDensity)
            #########################################################
            # CASE 12: constrained: RIGHT, FORCE: BOTTOM
            #########################################################
            g12 = f.create_group('T0B2L0R1')
            BC = util.formCond2D(self.mesh.getRight(), (-1, -1))
            for i in range(100):
                FC = util.formCond2D(self.mesh.getDown(),
                                     (randVal100(), randVal100()))
                strainDensity = solveFE2D(self.mesh, BC, FC)
                g12.create_dataset("config_{i}", data=strainDensity)
            #########################################################

    def augmentData(self):
        """
        Augments Data to increase the volume of the dataset
        """
        with h5py.File(self.fname, 'r') as f1:
            with h5py.File(self.fname, 'w') as f2:
                g1 = f1.get('group1')
                f2.create_group('group1_lflip')

        

    def generateData(self) -> None:
        """
        Generating dataset
        """
        for i in range(self.amount):
            np.random.rand(self.nelx, self.nely)

    def splitTrainTest(self, train: float):
        if 0.5 < train < 1:
            test = 1 - train
