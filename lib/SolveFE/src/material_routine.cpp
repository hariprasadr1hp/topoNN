/**
 * Material routine description
*/

#include "material_routine.hpp"

Material :: Material()
{
    E = 2000000000;
    Nu = 0.3;
    sigY = 415000000;
}

Material :: Material(Eigen::Vector3d& matl)
{
    E = matl(0);
    Nu = matl(1);
    sigY = matl(2);
}

void Material :: setParams(Eigen::Vector3d& matl)
{
    E = matl(0);
    Nu = matl(1);
    sigY = matl(2);
}

void Material :: getParams(Eigen::Vector3d& matl)
{
    matl(0) = E;
    matl(1) = Nu;
    matl(2) = sigY;
}

void Material :: MaterialInitialize(Eigen::MatrixXd& Cmat, Eigen::VectorXd& Stress, 
                                                Eigen::VectorXd& Strain)
{
    Isotropic(Cmat);
    Stress = Cmat * Strain;
}

void Material :: Isotropic(Eigen::MatrixXd& Cmat)
{
    double constant,x,y,z;
    constant = E / ( (1+Nu) * (1-(2*Nu)) );
    x = constant * Nu;
    y = constant * (1-Nu);
    z = 0.5 * (1-(2*Nu));
    Cmat.topLeftCorner(3,3) << y,x,x,
                                x,y,x,
                                x,x,y;
    Cmat.bottomRightCorner(3,3) = z * Eigen::Matrix3d::Identity();
    Cmat.topRightCorner(3,3).setZero();
    Cmat.bottomLeftCorner(3,3).setZero();
}
