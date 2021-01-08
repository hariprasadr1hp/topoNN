/**
 * Material routine description
*/

#pragma once

#include <Eigen/Dense>


class Material
{
    private:
        double E, Nu, sigY;

    public:
        Material();
        Material(Eigen::Vector3d& matl);
        void setParams(Eigen::Vector3d& matl);
        void getParams(Eigen::Vector3d& matl);
        void MaterialInitialize(Eigen::MatrixXd& Cmat,
                                Eigen::VectorXd& Stress,
                                Eigen::VectorXd& Strain);
        void Isotropic(Eigen::MatrixXd& Cmat);

        
};
