"""
This module generates training data
"""

import h5py
import numpy as np

from ingest.generate.mesh import Plate
from process import util
from process.solve_fe import SolveFE


class DataGen:
    """
    Generates Data for structural optimization

    :type mesh: class
    :param nelx: Generates the plate Geometry for the given x,y elements

    :type amount: int
    :param amount: volume of the dataset
    """

    def __init__(self, mesh: Plate, amount: int) -> None:
        self.mesh = mesh
        self.nelx = mesh.nelx
        self.nely = mesh.nely
        self.amount = amount
        self.fname = "topoNN.h5"
        self.augment_data()

    def generate_forces(self):
        """
        Generating volume strain energy density data for a specific
        pattern of boundary conditions by only altering the magnitudes
        """

        def rand_vals_100() -> int:
            """Generates a random value from 1 to 100"""
            return int(np.round(100 * np.random.rand()))

        #########################################################
        # CASE 1: constrained: BOTTOM, FORCE: TOP
        #########################################################
        with h5py.File(self.fname, "w") as f:
            g1 = f.create_group("T1B2L0R0")
            boundary_condns = util.formulate_condns(self.mesh.get_down(), (-1, -1))
            for _ in range(1000):
                force_condns = util.formulate_condns(
                    self.mesh.get_up(), (rand_vals_100(), rand_vals_100())
                )
                strain_density = SolveFE(self.mesh, boundary_condns, force_condns)
                g1.create_dataset("config_{i}", data=strain_density)
            #########################################################
            # CASE 2: constrained: TOP, FORCE: BOTTOM
            #########################################################
            g2 = f.create_group("T2B1L0R0")
            boundary_condns = util.formulate_condns(self.mesh.get_up(), (-1, -1))
            for i in range(1000):
                force_condns = util.formulate_condns(
                    self.mesh.get_down(), (rand_vals_100(), rand_vals_100())
                )
                strain_density = SolveFE(self.mesh, boundary_condns, force_condns)
                g2.create_dataset("config_{i}", data=strain_density)
            #########################################################
            # CASE 3: constrained: LEFT, FORCE: RIGHT
            #########################################################
            g3 = f.create_group("T0B0L1R2")
            boundary_condns = util.formulate_condns(self.mesh.get_left(), (-1, -1))
            for i in range(1000):
                force_condns = util.formulate_condns(
                    self.mesh.get_right(), (rand_vals_100(), rand_vals_100())
                )
                strain_density = SolveFE(self.mesh, boundary_condns, force_condns)
                g3.create_dataset("config_{i}", data=strain_density)
            #########################################################
            # CASE 4: constrained: RIGHT, FORCE: LEFT
            #########################################################
            g4 = f.create_group("T0B0L2R1")
            boundary_condns = util.formulate_condns(self.mesh.get_right(), (-1, -1))
            for i in range(1000):
                force_condns = util.formulate_condns(
                    self.mesh.get_left(), (rand_vals_100(), rand_vals_100())
                )
                strain_density = SolveFE(self.mesh, boundary_condns, force_condns)
                g4.create_dataset("config_{i}", data=strain_density)
            #########################################################
            # CASE 5: constrained: BOTTOM, FORCE: LEFT
            #########################################################
            g5 = f.create_group("T0B1L2R0")
            boundary_condns = util.formulate_condns(self.mesh.get_down(), (-1, -1))
            for i in range(1000):
                force_condns = util.formulate_condns(
                    self.mesh.get_left(), (rand_vals_100(), rand_vals_100())
                )
                strain_density = SolveFE(self.mesh, boundary_condns, force_condns)
                g5.create_dataset("config_{i}", data=strain_density)
            #########################################################
            # CASE 6: constrained: BOTTOM, FORCE: RIGHT
            #########################################################
            g6 = f.create_group("T0B1L0R2")
            boundary_condns = util.formulate_condns(self.mesh.get_down(), (-1, -1))
            for i in range(1000):
                force_condns = util.formulate_condns(
                    self.mesh.get_right(), (rand_vals_100(), rand_vals_100())
                )
                strain_density = SolveFE(self.mesh, boundary_condns, force_condns)
                g6.create_dataset("config_{i}", data=strain_density)
            #########################################################
            # CASE 7: constrained: TOP, FORCE: LEFT
            #########################################################
            g7 = f.create_group("T1B0L2R0")
            boundary_condns = util.formulate_condns(self.mesh.get_up(), (-1, -1))
            for i in range(1000):
                force_condns = util.formulate_condns(
                    self.mesh.get_left(), (rand_vals_100(), rand_vals_100())
                )
                strain_density = SolveFE(self.mesh, boundary_condns, force_condns)
                g7.create_dataset("config_{i}", data=strain_density)
            #########################################################
            # CASE 8: constrained: TOP, FORCE: RIGHT
            #########################################################
            g8 = f.create_group("T1B0L0R2")
            boundary_condns = util.formulate_condns(self.mesh.get_up(), (-1, -1))
            for i in range(1000):
                force_condns = util.formulate_condns(
                    self.mesh.get_right(), (rand_vals_100(), rand_vals_100())
                )
                strain_density = SolveFE(self.mesh, boundary_condns, force_condns)
                g8.create_dataset("config_{i}", data=strain_density)
            #########################################################
            # CASE 9: constrained: LEFT, FORCE: TOP
            #########################################################
            g9 = f.create_group("T2B0L1R0")
            boundary_condns = util.formulate_condns(self.mesh.get_left(), (-1, -1))
            for i in range(1000):
                force_condns = util.formulate_condns(
                    self.mesh.get_up(), (rand_vals_100(), rand_vals_100())
                )
                strain_density = SolveFE(self.mesh, boundary_condns, force_condns)
                g9.create_dataset("config_{i}", data=strain_density)
            #########################################################
            # CASE 10: constrained: LEFT, FORCE: BOTTOM
            #########################################################
            g10 = f.create_group("T0B2L1R0")
            boundary_condns = util.formulate_condns(self.mesh.get_left(), (-1, -1))
            for i in range(1000):
                force_condns = util.formulate_condns(
                    self.mesh.get_down(), (rand_vals_100(), rand_vals_100())
                )
                strain_density = SolveFE(self.mesh, boundary_condns, force_condns)
                g10.create_dataset("config_{i}", data=strain_density)
            #########################################################
            # CASE 11: constrained: RIGHT, FORCE: TOP
            #########################################################
            g11 = f.create_group("T2B0L0R1")
            boundary_condns = util.formulate_condns(self.mesh.get_right(), (-1, -1))
            for i in range(1000):
                force_condns = util.formulate_condns(
                    self.mesh.get_up(), (rand_vals_100(), rand_vals_100())
                )
                strain_density = SolveFE(self.mesh, boundary_condns, force_condns)
                g11.create_dataset("config_{i}", data=strain_density)
            #########################################################
            # CASE 12: constrained: RIGHT, FORCE: BOTTOM
            #########################################################
            g12 = f.create_group("T0B2L0R1")
            boundary_condns = util.formulate_condns(self.mesh.get_right(), (-1, -1))
            for i in range(100):
                force_condns = util.formulate_condns(
                    self.mesh.get_down(), (rand_vals_100(), rand_vals_100())
                )
                strain_density = SolveFE(self.mesh, boundary_condns, force_condns)
                g12.create_dataset("config_{i}", data=strain_density)

    def augment_data(self):
        """
        Augments Data to increase the volume of the dataset
        """
        with h5py.File(self.fname, "r") as f1:
            with h5py.File(self.fname, "w") as f2:
                _ = f1.get("group1")
                f2.create_group("group1_lflip")
