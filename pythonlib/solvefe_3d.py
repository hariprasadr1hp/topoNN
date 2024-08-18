"""
This module will solve the FE problem
"""

import numpy as np
from pythonlib import element_routine as ER


class SolveFE3D:
    """
    Solves the 3D FE problem

    :type mesh3D: class
    :param nodeCoords: the mesh-grid structure

    :type BC: ndarray(N X 4)
    :param BC: Boundary conditions, nodes along with constraints in
                x,y,z direction

    :type FC: ndarray(N X 4)
    :param FC: Boundary conditions, nodes along with values in
                x,y,z direction

    :type matlParams: tuple(E, Nu)
    :param matlParams: youngs modulus and poissons ratio
    """

    def __init__(
        self, mesh_3d, boundary_condns, force_condns, matl_params=(120, 0.3, 5)
    ):
        self.mesh_3d = mesh_3d
        self.node_coords = mesh_3d.nodeCoords
        self.elem_nodes = mesh_3d.elemNodes
        self.matl_params = matl_params
        self.nelx = mesh_3d.nelx
        self.nely = mesh_3d.nely
        self.nelz = mesh_3d.nelz

        self.boundary_condns = boundary_condns
        self.force_condns = force_condns

        self.totnodes = np.shape(self.node_coords)[0]
        self.totelems = np.shape(self.elem_nodes)[0]
        self.total_dofs = self.totnodes * 3
        self.elem_dofs = 24

        # Initialize Matrices
        self.f_int_global = np.zeros((self.total_dofs))
        self.f_ext_global = np.zeros((self.total_dofs))
        self.k_global = np.zeros((self.total_dofs, self.total_dofs), dtype=float)
        self.u_global = np.zeros((self.total_dofs))
        self.du_global = np.zeros((self.total_dofs))

        # boundary conditions
        self.u_free_dofs = np.array([False for i in range(self.total_dofs)])
        self.u_fixed_dofs = np.array([False for i in range(self.total_dofs)])
        self.k_mask = np.array(
            [[True for j in range(self.total_dofs)] for j in range(self.total_dofs)]
        )
        self.set_boundary_condns(self.boundary_condns)
        self.set_force_condns(self.force_condns)

        # attributes
        self.strain_density = np.zeros((self.nelx, self.nely, self.nelz))
        self.u_x = np.zeros((self.nelx + 1, self.nely + 1, self.nelz + 1))
        self.u_y = np.zeros((self.nelx + 1, self.nely + 1, self.nelz + 1))
        self.u_xy = np.zeros((self.nelx + 1, self.nely + 1, self.nelz + 1))

    # @staticmethod
    def get_dof_ids(self, node_ids):
        """
        Returns the [x,y,z] IDs for the given Node IDs

        :type NodeIDs: ndarray
        :param NodeIDs: the ID of the node
        """

        def xyz(x):
            return [3 * x - 3, 3 * x - 2, 3 * x - 1]

        node_dofs = np.concatenate([xyz(i) for i in node_ids])
        return node_dofs

    def set_force_condns(self, fc) -> None:
        """
        Sets the force conditions

        :type fc: nd.array(N X 4)
        :param fc: the node IDs along with their force constraints
                    in the x,y direction
        """
        for each in fc:
            self.f_ext_global[int(each[0]) * 3 - 3] = each[1]
            self.f_ext_global[int(each[0]) * 3 - 2] = each[2]
            self.f_ext_global[int(each[0]) * 3 - 1] = each[3]

    # def boundarySet(self,U_ID,vals):
    def set_boundary_condns(self, boundary_conds) -> None:
        """
        Sets the boundary conditions

        :type bc: nd.array(N X 4)
        :param bc: the node IDs along with their displacements
                        constraints in the x,y direction
        """
        for each in boundary_conds:
            if each[1] == -1:
                self.u_fixed_dofs[int(each[0]) * 3 - 3] = True
            if each[2] == -1:
                self.u_fixed_dofs[int(each[0]) * 3 - 2] = True
            if each[3] == -1:
                self.u_fixed_dofs[int(each[0]) * 3 - 1] = True
            # else:
            #     pass
        self.u_free_dofs = ~self.u_fixed_dofs

    @staticmethod
    def get_strain_energy(u_el, k_el):
        """
        Calculates the starin energy density of an element

        :type u_el: ndarray (24,1)
        :param u_el: the element displacement

        :type K_el: ndarray (24,24)
        :param K_el: the element stiffness matrix
        """
        return u_el.T @ k_el @ u_el

    def fill_matrix(self, elem_id: int, k_el, f_int_el) -> None:
        """
        Fills the sparse Global stiffness matrix with the local
        stifnesses

        :type K_el: ndarray (24,24)
        :param K_el: the local stiffness matrix of an element

        :type Fint_el: ndarray (24,1)
        :param Fint_el: the internal force matrix of an element
        """

        def xyz(x):
            return [3 * x - 3, 3 * x - 2, 3 * x - 1]

        dofs = np.concatenate([xyz(i) for i in self.elem_nodes[elem_id - 1]])
        for i, j in enumerate(dofs):
            self.k_global[j, j] += k_el[i, i]
            self.f_int_global[j] += f_int_el[i]

    def assembly_global(self):
        """
        Assembly of the global Force and Stiffness matrices
        """
        hex8_el = np.zeros((8, 3), dtype=float)
        u_el = np.zeros((24), dtype=float)

        for i, elem_id in enumerate(self.elem_nodes):
            for j, node_id in enumerate(elem_id):
                hex8_el[j] = self.node_coords[node_id - 1]
            u_el = self.get_dof_ids(self.elem_nodes[i])
            element = ER.Brick(GP=1, hex8_rc=hex8_el)
            k_el, f_int_el = element.hex8(self.matl_params, u_el)

            self.get_strain_energy(u_el, k_el)
            self.strain_density[self.mesh_3d.getElemOrigin(i)] = self.get_strain_energy(
                u_el, k_el
            )
            self.fill_matrix(i + 1, k_el, f_int_el)

    def set_matl_params(self, matl_params):
        """
        Sets the material Parameters

        :type matlParams: tuple(float) -> (E,Nu,sigY)
        :param matlParams: the material parameters of the element
        """
        self.matl_params = matl_params

    def update_du_global(self, du_global_red):
        """
        Updates the global displacement vector after each iteration

        :type du_Global_red: numpy.array(int)
        :param du_Global_red: the nodal displacements of IDs with
                                non-zero displacement
        """
        count = 0
        while count < np.size(du_global_red):
            for each, _ in enumerate(self.u_free_dofs):
                if self.u_free_dofs[each]:
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
        k_global_red = self.k_global[self.u_free_dofs]
        k_global_red = (k_global_red.T[self.u_free_dofs]).T
        f_int_global_red = self.f_int_global[self.u_free_dofs]
        f_ext_global_red = self.f_ext_global[self.u_free_dofs]
        g_red = f_ext_global_red - f_int_global_red
        du_global_red = np.linalg.solve(k_global_red, f_ext_global_red)
        self.update_du_global(du_global_red)
        self.u_global += self.du_global

        # print(np.max(np.abs(self.Fext_Global-self.Fint_Global)))

        # fname_ux = "data/ux.jpg"
        # fname_uy = "data/uy.jpg"
        # fname_uxy = "data/uxy.jpg"
        # ux, uy = util.splitXY(self.u_Global)
        # uxy = ux**2 + uy**2
        # ux = np.flip(ux.reshape(self.nelx+1, self.nely+1), 0)
        # uy = np.flip(uy.reshape(self.nelx+1, self.nely+1), 0)
        # uxy = np.flip(uxy.reshape(self.nelx+1, self.nely+1), 0)
        # util.saveContour(ux, fname_ux)
        # util.saveContour(uy, fname_uy)
        # util.saveContour(uxy, fname_uxy)

        # print(uy)
        # fname_k = "data/k_matrix.jpg"
        # util.saveContour(self.K_Global, fname_k)
        return self.u_global

    def get_euclidean_disp(self):
        """Euclidean distance of u_global"""
        ux = self.u_global[0 : self.totelems : 3]
        uy = self.u_global[1 : self.totelems : 3]
        uz = self.u_global[2 : self.totelems : 3]

        return np.sqrt(ux**2 + uy**2 + uz**2)
