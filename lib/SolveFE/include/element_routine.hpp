/**
 * Element routine description
*/

#pragma once

#include <Eigen/Dense>
#include "material_routine.hpp"

class Element
{
    private:
        Eigen::MatrixXd hex8_nc, hex8_rc;
        Eigen::VectorXd u_local;
        Eigen::Vector3d matlParams;
        const double xi = 0.5773502692;

    public:
        Element(Eigen::ArrayXXd&,Eigen::VectorXd&,Eigen::Vector3d&);
        void elemInitialize(Eigen::MatrixXd&,Eigen::VectorXd&);
        void natCoords();
        void shapeFunc(Eigen::VectorXd&);
        void dN_natural(Eigen::MatrixXd&);
        void JacobianMat(Eigen::Matrix3d&,Eigen::MatrixXd&);
        void dN_reference(Eigen::MatrixXd&,Eigen::Matrix3d&,Eigen::MatrixXd&);
        void BMatrix(Eigen::MatrixXd&,Eigen::MatrixXd&);
};
