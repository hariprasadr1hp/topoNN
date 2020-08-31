#include "element_routine.hpp"

Element :: Element(ArrayXXd& eleminfo,VectorXd& disp,Vector3d& matl)
{ 
    natCoords();
    hex8_rc = eleminfo;
    u_local = disp;
    matlParams = matl;
    // elemInitialize();
}

/**
 * Sets the natural coordinates for an hex8 element
 */
void Element :: natCoords()
{
    hex8_nc.resize(8,3);
    hex8_nc.setZero();
    hex8_nc <<  -1, -1, -1,
                +1, -1, -1,
                -1, +1, -1,
                +1, +1, -1,
                -1, -1, +1,
                +1, -1, +1,
                -1, +1, +1,
                +1, +1, +1;
}


void Element :: elemInitialize(MatrixXd& K_el, VectorXd& Fint_el)
{
    //initialize vectors and matrices
    VectorXd shapeVector(8);
    MatrixXd dN_nat(8,3);
    Matrix3d Jmat;
    MatrixXd dN_ref(8,3);
    MatrixXd Bmat(6,24);
    VectorXd Strain_el(6);
    MatrixXd Cmat(6,6);
    VectorXd Stress_el(6);
    Material matl(matlParams);

    //simulate
    shapeFunc(shapeVector);
    dN_natural(dN_nat);
    JacobianMat(Jmat,dN_nat);
    dN_reference(dN_ref,Jmat,dN_nat);
    BMatrix(Bmat,dN_ref);
    Strain_el = Bmat * u_local;
    matl.MaterialInitialize(Cmat,Stress_el,Strain_el);
    K_el = Jmat.determinant() * Bmat.transpose() * Cmat * Bmat;
    Fint_el = Jmat.determinant() * Bmat.transpose() * Stress_el;
    // cout << hex8_rc << endl;
}

/**
 * computes the Shape Function of the element,
 * vector of size (8 X 1)
 */
void Element :: shapeFunc(VectorXd& shapeVector)
{
    shapeVector = (
        ( (hex8_nc.col(0).array() * xi) + 1 ) *
        ( (hex8_nc.col(1).array() * xi) + 1 ) *
        ( (hex8_nc.col(2).array() * xi) + 1 )
    )/8;
}

/**
 * Computes the derivatives of the shape function -> (8 X 3)
 * w.r.t its natural coordinates, for the hex8 element
 */
void Element :: dN_natural(MatrixXd& dN_nat)
{
    dN_nat.col(0) = (
        ( (hex8_nc.col(0).array() *  1) + 1 ) *
        ( (hex8_nc.col(1).array() * xi) + 1 ) *
        ( (hex8_nc.col(2).array() * xi) + 1 )
    )/8;
    
    dN_nat.col(1) = (
        ( (hex8_nc.col(0).array() * xi) + 1 ) *
        ( (hex8_nc.col(1).array() *  1) + 1 ) *
        ( (hex8_nc.col(2).array() * xi) + 1 )
    )/8;

    dN_nat.col(2) = (
        ( (hex8_nc.col(0).array() * xi) + 1 ) *
        ( (hex8_nc.col(1).array() * xi) + 1 ) *
        ( (hex8_nc.col(2).array() *  1) + 1 )
    )/8;
}

/**
 * computes the Jacobian matrix of the element,
 * of size (3 X 3)
 */
void Element :: JacobianMat(Matrix3d& Jmat , MatrixXd& dN_nat)
{
    Jmat = hex8_rc.transpose() * dN_nat;
}

/**
 * Computes the derivatives of the shape function -> (8 X 3)
 * w.r.t its reference coordinates, for the hex8 element
 */
void Element :: dN_reference(MatrixXd& dN_ref, Matrix3d& Jmat, MatrixXd& dN_nat)
{
    dN_ref = dN_nat * Jmat.inverse().transpose();
}

/**
 * Computes the strain-displacement matrix(B) -> (6 X 24)
 */
void Element :: BMatrix(MatrixXd& Bmat, MatrixXd& dN_ref)
{
    //e11
    Bmat.setZero();
    Bmat.row(0).head(8) = dN_ref.col(0);
    //e22
    Bmat.row(1).segment(8,8) = dN_ref.col(1);
    //e33
    Bmat.row(2).segment(16,8) = dN_ref.col(2);
    //e23
    Bmat.row(3).segment(8,8) = dN_ref.col(1);
    Bmat.row(3).segment(16,8) = dN_ref.col(2);
    //e13
    Bmat.row(4).head(8) = dN_ref.col(0);
    Bmat.row(4).segment(16,8) = dN_ref.col(2);
    //e12
    Bmat.row(5).head(8) = dN_ref.col(0);
    Bmat.row(5).segment(8,8) = dN_ref.col(1);    
}
