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
        Element(Eigen::ArrayXXd& eleminfo, Eigen::VectorXd& disp, 
                Eigen::Vector3d& matl);
        void elemInitialize(Eigen::MatrixXd& K_el,Eigen::VectorXd& Fint_el);
        void natCoords();
        void shapeFunc(Eigen::VectorXd& shapeVector);
        void dN_natural(Eigen::MatrixXd& dN_nat);
        void JacobianMat(Eigen::Matrix3d& Jmat, Eigen::MatrixXd& dN_nat);
        void dN_reference(Eigen::MatrixXd& dN_ref, Eigen::Matrix3d& Jmat,
                            Eigen::MatrixXd& dN_nat);
        void BMatrix(Eigen::MatrixXd& Bmat, Eigen::MatrixXd& dN_ref);
};
