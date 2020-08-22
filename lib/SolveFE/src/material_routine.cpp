#include "material_routine.hpp"

Material :: Material()
{
    E = 2000000000;
    Nu = 0.3;
    sigY = 415000000;
}

Material :: Material(Vector3d& matl)
{
    E = matl(0);
    Nu = matl(1);
    sigY = matl(2);
}

void Material :: setParams(Vector3d& matl)
{
    E = matl(0);
    Nu = matl(1);
    sigY = matl(2);
}

void Material :: getParams(Vector3d& matl)
{
    matl(0) = E;
    matl(1) = Nu;
    matl(2) = sigY;
}

void Material :: MaterialInitialize(MatrixXd& Cmat, VectorXd& Stress, 
                                                VectorXd& Strain)
{
    Isotropic(Cmat);
}

void Material :: Isotropic(MatrixXd& Cmat)
{
    double constant,x,y,z;
    constant = E / ( (1+Nu) * (1-(2*Nu)) );
    x = constant * Nu;
    y = constant * (1-Nu);
    z = (x-y)/2;
    Cmat.topLeftCorner(3,3) << y,x,x,x,y,x,x,x,y;
    Cmat.topRightCorner(3,3).setZero();
    Cmat.bottomLeftCorner(3,3).setZero();
    Cmat.bottomRightCorner(3,3) = z * Matrix3d::Identity();
}
