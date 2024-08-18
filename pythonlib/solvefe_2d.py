"""
This module will solve the 2D-FE problem
"""

from pathlib import Path

import numpy as np

from pythonlib.util import create_dir_if_not_exists, save_contour, split_xy
from pythonlib import element_routine as ER
from settings import PROJECT_DATA_DIR


class SolveFE2D:
    """
    Solves the 2D FE problem

    :type mesh2D: class
    :param mesh2D: the mesh-grid structure

    :type BC: ndarray(N X 3)
    :param BC: Boundary conditions, nodes along with constraints in
                x,y direction

    :type FC: ndarray(N X 3)
    :param FC: Boundary conditions, nodes along with values in
                x,y direction

    :type matlParams: tuple(E, Nu)
    :param matlParams: youngs modulus and poissons ratio
    """

    def __init__(self, mesh_2d, boundary_condns, force_condns, matl_params=(120, 0.3)):
        self.mesh_2d = mesh_2d
        self.node_coords = mesh_2d.node_coords
        self.elem_nodes = mesh_2d.elem_nodes
        self.nelx = mesh_2d.nelx
        self.nely = mesh_2d.nely
        self.matl_params = matl_params
        self.boundary_condns = boundary_condns
        self.force_condns = force_condns

        self.total_nodes = np.shape(self.node_coords)[0]
        self.total_elems = np.shape(self.elem_nodes)[0]
        self.total_dofs = self.total_nodes * 2
        self.elem_dofs = 8

        # Initialize Matrices
        self.f_int_global = np.zeros((self.total_dofs))
        self.f_ext_global = np.zeros((self.total_dofs))
        self.k_global = np.zeros((self.total_dofs, self.total_dofs), dtype=float)
        self.u_global = np.zeros((self.total_dofs))
        self.du_global = np.zeros((self.total_dofs))

        # boundary conditions
        self.u_free_dof = np.array([False for i in range(self.total_dofs)])
        self.u_fixed_dof = np.array([False for i in range(self.total_dofs)])
        self.k_mask = np.array(
            [[True for j in range(self.total_dofs)] for j in range(self.total_dofs)]
        )
        self.set_boundary_condns(self.boundary_condns)
        self.set_force_condns(self.force_condns)

        # attributes
        self.strain_density = np.zeros((self.nelx, self.nely))
        self.u_x = np.zeros((self.nelx + 1, self.nely + 1))
        self.u_y = np.zeros((self.nelx + 1, self.nely + 1))
        self.u_xy = np.zeros((self.nelx + 1, self.nely + 1))

    # @staticmethod
    def get_dof_ids(self, node_ids):
        """
        Returns the [x,y] IDs for the given Node IDs

        :type NodeIDs: ndarray
        :param NodeIDs: the ID of the node
        """

        def xy(x):
            return [2 * x - 2, 2 * x - 1]

        node_dofs = np.concatenate([xy(i) for i in node_ids])
        return node_dofs

    def set_force_condns(self, fc) -> None:
        """
        Sets the force conditions

        :type fc: nd.array(N X 3)
        :param fc: the node IDs along with their force constraints
                    in the x,y direction
        """
        for each in fc:
            self.f_ext_global[int(each[0]) * 2 - 2] = each[1]
            self.f_ext_global[int(each[0]) * 2 - 1] = each[2]

    # def boundarySet(self,U_ID,vals):
    def set_boundary_condns(self, bc) -> None:
        """
        Sets the boundary conditions

        :type bc: nd.array(N X 3)
        :param bc: the node IDs along with their displacements
                        constraints in the x,y direction
        """
        for each in bc:
            if each[1] == -1:
                self.u_fixed_dof[int(each[0]) * 2 - 2] = True
            if each[2] == -1:
                self.u_fixed_dof[int(each[0]) * 2 - 1] = True
            # else:
            #     pass
        self.u_free_dof = ~self.u_fixed_dof

    @staticmethod
    def get_strain_energy(u_el, k_el):
        """
        Calculates the starin energy density of an element

        :type u_el: ndarray (8,1)
        :param u_el: the element displacement

        :type K_el: ndarray (8,8)
        :param K_el: the element stiffness matrix
        """
        return u_el.T @ k_el @ u_el

    def fill_matrix(self, elem_id: int, k_el, f_int_el) -> None:
        """
        Fills the sparse Global stiffness matrix with the local
        stifnesses

        :type K_el: ndarray (8,8)
        :param K_el: the stiffness matrix of an element

        :type Fint_el: ndarray (8,1)
        :param Fint_el: the internal force matrix of an element
        """

        def xy(x):
            return [2 * x - 2, 2 * x - 1]

        dofs = np.concatenate([xy(i) for i in self.elem_nodes[elem_id - 1]])
        for i, j in enumerate(dofs):
            self.k_global[j, j] += k_el[i, i]
            self.f_int_global[j] += f_int_el[i]

    def assembly_global(self) -> None:
        """
        Assembly of the global Force and Stiffness matrices
        """
        quad_el = np.zeros((4, 2), dtype=float)
        u_el = np.zeros((8), dtype=float)

        for i, elem_id in enumerate(self.elem_nodes):
            for j, node_id in enumerate(elem_id):
                quad_el[j] = self.node_coords[node_id - 1]
            u_el = self.get_dof_ids(self.elem_nodes[i])
            element = ER.Quad(gauss_points=1, quad_rc=quad_el)
            k_el, f_int_el = element.quad_el(self.matl_params, u_el)

            self.get_strain_energy(u_el, k_el)
            self.strain_density[self.mesh_2d.get_elem_origin(i)] = (
                self.get_strain_energy(u_el, k_el)
            )
            self.fill_matrix(i + 1, k_el, f_int_el)

    def set_matl_params(self, matl_params) -> None:
        """
        Sets the material Parameters

        :type matlParams: tuple(float) -> (E,Nu,sigY)
        :param matlParams: the material parameters of the element
        """
        self.matl_params = matl_params

    def update_du_global(self, du_global_red) -> None:
        """
        Updates the global displacement vector after each iteration

        :type du_Global_red: numpy.array(int)
        :param du_Global_red: the nodal displacements of IDs with
                                non-zero displacement
        """
        count = 0
        while count < np.size(du_global_red):
            for each, _ in enumerate(self.u_free_dof):
                if self.u_free_dof[each]:
                    self.du_global[each] = du_global_red[count]
                    count += 1

    def solve_problem(self):
        """
        Solves the formulated FE Problem
        """
        # #iterative-solve
        # for step in range(1):
        #     self.assemblyGlobal()
        #     K_Global_red = self.K_Global[self.u_freeDOF]
        #     K_Global_red = (K_Global_red.T[self.u_freeDOF]).T
        #     Fint_Global_red = self.Fint_Global[self.u_freeDOF]
        #     Fext_Global_red = self.Fext_Global[self.u_freeDOF]
        #     G_red = Fext_Global_red - Fint_Global_red

        #     du_Global_red = np.linalg.solve(K_Global_red, G_red)
        #     self.update_duGlobal(du_Global_red)
        #     self.u_Global += self.du_Global

        # direct-solve
        self.assembly_global()
        k_global_red = self.k_global[self.u_free_dof]
        k_global_red = (k_global_red.T[self.u_free_dof]).T
        f_int_global_red = self.f_int_global[self.u_free_dof]
        f_ext_global_red = self.f_ext_global[self.u_free_dof]
        g_red = f_ext_global_red - f_int_global_red
        du_global_red = np.linalg.solve(k_global_red, f_ext_global_red)
        self.update_du_global(du_global_red)
        self.u_global += self.du_global

        # print(np.max(np.abs(self.Fext_Global-self.Fint_Global)))

        fname_ux = "data/ux.jpg"
        fname_uy = "data/uy.jpg"
        fname_uxy = "data/uxy.jpg"
        ux, uy = split_xy(self.u_global)
        uxy = ux**2 + uy**2
        ux = np.flip(ux.reshape(self.nelx + 1, self.nely + 1), 0)
        uy = np.flip(uy.reshape(self.nelx + 1, self.nely + 1), 0)
        uxy = np.flip(uxy.reshape(self.nelx + 1, self.nely + 1), 0)

        create_dir_if_not_exists(fpath=PROJECT_DATA_DIR)
        save_contour(ux, r"$u_x$", fname_ux, r"$x$")
        save_contour(uy, r"$u_y$", fname_uy, r"$y$")
        save_contour(uxy, r"$|u|$", fname_uxy, r"$x$")

        # print(uy)
        fname_k = "data/k_matrix.jpg"
        # saveContour(self.K_Global, fname_k)
        return self.u_global

    def get_euclidean_disp(self):
        """Euclidean distance of u_global"""
        ux = self.u_global[0 : self.total_elems : 3]
        uy = self.u_global[1 : self.total_elems : 3]
        uz = self.u_global[2 : self.total_elems : 3]

        return np.sqrt(ux**2 + uy**2 + uz**2)

    def get_analytical_soln(self):
        stiff = self.matl_params[0] * np.array(
            [
                [+2, +0, -1, +0, -1, +0, +0, +0],
                [+0, +2, +0, -1, +0, -1, +0, +0],
                [-1, +0, +2, +0, +0, +0, -1, +0],
                [+0, -1, +0, +2, +0, +0, +0, -1],
                [-1, +0, +0, +0, +2, +0, -1, +0],
                [+0, -1, +0, +0, +0, +2, +0, -1],
                [+0, +0, -1, +0, -1, +0, +2, +0],
                [+0, +0, +0, -1, +0, -1, +0, +2],
            ]
        )
        stiff_global = np.zeros((self.total_dofs, self.total_dofs))
        u_act = np.zeros((self.total_dofs))

        def xy(x):
            return [2 * x - 2, 2 * x - 1]

        for k, elem_id in enumerate(self.elem_nodes):
            dofs = np.concatenate([xy(l) for l in self.elem_nodes[k]])
            for i, j in enumerate(dofs):
                stiff_global[j, j] += stiff[i, i]
        stiff_red = stiff_global[self.u_free_dof]
        stiff_red = (stiff_red.T[self.u_free_dof]).T
        f_ext_global_red = self.f_ext_global[self.u_free_dof]
        # print(np.linalg.det(stiff))
        # print(np.linalg.det(stiff_red))
        u_act_red = np.linalg.solve(stiff_red, f_ext_global_red)
        count = 0
        while count < np.size(u_act_red):
            for each, _ in enumerate(self.u_free_dof):
                if self.u_free_dof[each]:
                    u_act[each] = u_act_red[count]
                    count += 1
        return u_act
