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
        Material(Eigen::Vector3d&);
        void setParams(Eigen::Vector3d&);
        void getParams(Eigen::Vector3d&);
        void MaterialInitialize(Eigen::MatrixXd&,
                                Eigen::VectorXd&,
                                Eigen::VectorXd&);
        void Isotropic(Eigen::MatrixXd&);

        
};
