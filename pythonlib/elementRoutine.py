"""
elementRoutine: Defines the characteristics of the element
"""
import numpy as np
from pythonlib import materialRoutine as MR

class Brick:
    def __init__(self,GP,coord):
        self.GP = GP
        self.coord = coord
        self.grids = 8
        self.axes = 3
        self.components = 6
        self.Gtab = self.calcQuadrature()
    #____________________________________________________________________
    #UNDER CONSTRUCTION
    def calcQuadrature(self):
        """
        Returns the Gauss Points table based on the no.of Gauss Points
        """
        self.Gtab = np.zeros(((self.GP**self.axes),5))
        self.Gtab[0,:] = [1,
                          0.5773502692,
                          0.5773502692,
                          0.5773502692,
                          1]
        return self.Gtab
    #____________________________________________________________________

    def natCoords(self):
        """
        hex8-Natural Coordinates[xi1, xi2, xi3] -> (8 X 3)
        """
        hex8_nc = np.zeros((self.grids,self.axes),dtype='int32')
        hex8_nc[0] = (-1, -1, -1)
        hex8_nc[1] = (+1, -1, -1)
        hex8_nc[2] = (-1, +1, -1)
        hex8_nc[3] = (+1, +1, -1)
        hex8_nc[4] = (-1, -1, +1)
        hex8_nc[5] = (+1, -1, +1)
        hex8_nc[6] = (-1, +1, +1)
        hex8_nc[7] = (+1, +1, +1)
        return hex8_nc
    #____________________________________________________________________

    def refCoords(self):
        """
        hex8-reference Coordinates[xi1, xi2, xi3] -> (8 X 3)
        """
        hex8_rc = np.zeros((self.grids,self.axes))
        x1 = self.coord[0]
        y1 = self.coord[1]
        z1 = self.coord[2]
        hex8_rc[0] = (x1,   y1,   z1)
        hex8_rc[1] = (x1+1, y1,   z1)
        hex8_rc[2] = (x1  , y1+1, z1)
        hex8_rc[3] = (x1+1, y1+1, z1)
        hex8_rc[4] = (x1,   y1,   z1+1)
        hex8_rc[5] = (x1+1, y1,   z1+1)
        hex8_rc[6] = (x1  , y1+1, z1+1)
        hex8_rc[7] = (x1+1, y1+1, z1+1)
        return hex8_rc
    #____________________________________________________________________

    def shapeFunc(self,hex8_nc,GP):
        """
        Compute shape functions for an hex8 element -> 8 X 1
        shapefunction = (1/8)*(1 + aa(i))*(1 + bb(i))*(1 + cc(i))

        :type hex8_nc: numpy.array(float,float) -> (8 X 3)
        :param hex8_nc: the natural coordinates of each node of the element

        :type GP: int
        :param GP: Gauss point
        """
        N = np.zeros((self.grids))
        N = (1/8) * (1 + (self.Gtab[GP,1] * hex8_nc[:,0]) ) *  \
                    (1 + (self.Gtab[GP,2] * hex8_nc[:,1]) ) *  \
                    (1 + (self.Gtab[GP,3] * hex8_nc[:,2]) )
        return N
    #____________________________________________________________________

    def dN_nat(self,hex8_nc,GP):
        """
        Computes the derivatives of the shape function -> (8 X 3)
        w.r.t its natural coordinates, for the hex8 element

        :type hex8_nc: numpy.array(float,float) -> (8 X 3)
        :param hex8_nc: the natural coordinates of each node of the element

        :type GP: int
        :param GP: Gauss point
        """
        dN_nat = np.zeros((self.grids,self.axes))
        dN_nat[:,0] = (1/8) * (hex8_nc[:,0]) * \
                              (1 + (self.Gtab[GP,2] * hex8_nc[:,1]) ) * \
                              (1 + (self.Gtab[GP,3] * hex8_nc[:,2]) )
        dN_nat[:,1] = (1/8) * (hex8_nc[:,1]) * \
                              (1 + (self.Gtab[GP,1] * hex8_nc[:,0]) ) * \
                              (1 + (self.Gtab[GP,3] * hex8_nc[:,2]) )
        dN_nat[:,2] = (1/8) * (hex8_nc[:,2]) * \
                              (1 + (self.Gtab[GP,1] * hex8_nc[:,0]) ) * \
                              (1 + (self.Gtab[GP,2] * hex8_nc[:,1]) )
        return dN_nat
    #____________________________________________________________________

    @staticmethod
    def JacobianMat(dN_nat,hex8_rc):
        """
        Computes the Jacobian Matrix  -> (3 X 3)
        J = dx/de = x_i * dN/de

        :type dN_nat: numpy.array(float,float) -> (8 X 3)
        :param dN_nat: the derivates of shape functions w.r.t to its
                        natural coordinates

        :type hex8_nc: numpy.array(float,float) -> (8 X 3)
        :param hex8_nc: the natural coordinates of each node of the element
        """
        Jmat = hex8_rc.T @ dN_nat
        return Jmat
    #____________________________________________________________________

    @staticmethod
    def dN_ref(Jmat,dN_nat):
        """
        Computes the derivatives of the shape function -> (8 X 3)
        w.r.t its reference coordinates, for the hex8 element

        :type Jmat: numpy.array(float,float) -> (3 X 3)
        :param Jmat: the Jacobian matrix of the element

        :type dN_nat: numpy.array(float,float) -> (8 X 3)
        :param dN_nat: the derivates of shape functions w.r.t to its
                        natural coordinates
        """
        Jinv = np.linalg.inv(Jmat)
        dN_ref = dN_nat @ Jinv.T
        return dN_ref
    #____________________________________________________________________

    def B_matrix(self,dN_ref):
        """
        Computes the strain-displacement matrix(B) -> (6 X 24)

        :type dN_ref: numpy.array(float,float) -> (8 X 3)
        :param dN_ref: the derivates of shape functions w.r.t to its
                        reference coordinates
        """
        Bmat = np.zeros((self.components,(self.grids*self.axes)))
        for i in range(self.grids):
            # 9 entries
            Bmat[0,(3*i+0)] = dN_ref[i,0]
            Bmat[1,(3*i+1)] = dN_ref[i,1]
            Bmat[2,(3*i+2)] = dN_ref[i,2]
            Bmat[3,(3*i+0)] = dN_ref[i,0]
            Bmat[3,(3*i+1)] = dN_ref[i,1]
            Bmat[4,(3*i+1)] = dN_ref[i,1]
            Bmat[4,(3*i+2)] = dN_ref[i,2]
            Bmat[5,(3*i+0)] = dN_ref[i,0]
            Bmat[5,(3*i+2)] = dN_ref[i,2]
        return Bmat
    #____________________________________________________________________

    def hex8(self,matlParams,u_el):
        """
        Compute element's local stiffness matrix 'K_el'
        and the internal force vector 'Fint_el'

        K_el     -> 24 X 24
        Fint_el -> 24 X 1
        u_el     -> 24 X 1
        strain_el->  6 X 1
        stress_el->  6 X 1
        matstiff ->  6 X 6

        :type matlParams: tuple(float) -> (E,Nu,sigY)
        :param matlParams: the material parameters of the element

        :type u_el: numpy.array(float) -> (8 X 1)
        :param u_el: the nodal displacements of the element
        """
        Fint_el = np.zeros((self.grids*self.axes))
        K_el = np.zeros(((self.grids*self.axes),(self.grids*self.axes)))

        for GP in range(self.GP**self.axes):
            hex8_nc = self.natCoords()
            hex8_rc = self.refCoords()
            N = self.shapeFunc(hex8_nc,GP)
            dN_nat = self.dN_nat(hex8_nc,GP)
            Jmat = self.JacobianMat(dN_nat,hex8_rc)
            detJ = np.linalg.det(Jmat)
            dN_ref = self.dN_ref(Jmat,dN_nat)
            Bmat = self.B_matrix(dN_ref)

            #strain Matrix
            strain_el = Bmat @ u_el

            #Constitutive Matrix
            matl = MR.matGen(matlParams,strain_el,del_eps=0)
            stress_el, Cmat = matl.LEIsotropic()

            #K_el Formulation
            BtCB = Bmat.T @ Cmat @ Bmat
            K_el_GP = (self.Gtab[GP,4] * detJ) * BtCB
            Fint_el_GP = (self.Gtab[GP,4] * detJ) *  Bmat.T @ stress_el
            K_el += K_el_GP
            Fint_el += Fint_el_GP

        return K_el, Fint_el
