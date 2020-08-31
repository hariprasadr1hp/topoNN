#ifndef ELEMENT_ROUTINE_HPP
#define ELEMENT_ROUTINE_HPP

#include <Eigen/Dense>
#include "material_routine.hpp"
#include <iostream>

using namespace std;
using namespace Eigen;

class Element
{
    private:
        MatrixXd hex8_nc, hex8_rc;
        VectorXd u_local;
        Vector3d matlParams;
        const double xi = 0.5773502692;

    public:
        Element(ArrayXXd&,VectorXd&,Vector3d&);
        void elemInitialize(MatrixXd&,VectorXd&);
        void natCoords();
        void shapeFunc(VectorXd&);
        void dN_natural(MatrixXd&);
        void JacobianMat(Matrix3d&,MatrixXd&);
        void dN_reference(MatrixXd&,Matrix3d&,MatrixXd&);
        void BMatrix(MatrixXd&,MatrixXd&);
};







#endif //ELEMENT_ROUTINE_HPP