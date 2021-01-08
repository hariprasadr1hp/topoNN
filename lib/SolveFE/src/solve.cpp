/**
 * Solving the FE
*/

#include "solve.hpp"

SolveFE :: SolveFE(Eigen::ArrayXXi& _elemNodes, Eigen::ArrayXXd& _nodeCoords, 
                        Eigen::ArrayXXd& _BC, Eigen::ArrayXXd& _FC, Eigen::Vector3d& matl){
    matlParams = matl;
    elemNodes = _elemNodes;
    nodeCoords = _nodeCoords;
    BC = _BC;
    FC = _FC;
    totelems = elemNodes.col(0).size();
    totnodes = nodeCoords.col(0).size();
    K_global.resize(totnodes*3, totnodes*3);
    u_global.resize(totnodes*3); u_global.setZero();
    Fint_global.resize(totnodes*3);
    Fext_global.resize(totnodes*3);
    solveInitialize();
}

/**
 * Initialize the solver
 */
void SolveFE :: solveInitialize(){
 
    connectivityMatrix();

    // VectorXd disp;
    // ArrayXXd elemcoord(8,3);
    // MatrixXd K_el(24,24);
    // VectorXd Fint_el(24);
    // elemCoordinates(elemcoord,6);
    // disp = VectorXd :: Random(24);
    // Element elem(elemcoord,disp,matlParams);
    // elem.elemInitialize(K_el,Fint_el);
}

/**
 * Assembles local stiffness matrices to form a global one
 */
void SolveFE :: connectivityMatrix(){
    Eigen::VectorXd u_local(24); u_local.setZero();
    Eigen::ArrayXXd hex8_rc(8,3);
    Eigen::MatrixXd K_el(24,24);
    Eigen::VectorXd Fint_el(24);
    Eigen::VectorXi nodes(8);
    // u_local.setRandom();

    for (int i=0; i<totelems; ++i){
        nodes = elemNodes.row(i);
        elemCoordinates(hex8_rc,nodes);
        localDisp(u_local,nodes);
        Element elem(hex8_rc,u_local,matlParams);
        elem.elemInitialize(K_el,Fint_el);
    }
}

/**
 * Sets the nodal coordinates for the given element
 */
void SolveFE :: elemCoordinates(Eigen::ArrayXXd& elemcoord, Eigen::VectorXi& nodes){
    for (int i=0;i<8;++i){
        elemcoord.row(i) = nodeCoords.row(nodes(i)-1);
    }
}

/**
 * Returns the x,y,z nodes of given 8-nodes of a hex-element
 * (24 X 1)
 */
void SolveFE :: localDisp(Eigen::VectorXd& u_local, Eigen::VectorXi& nodes){   
    int temp;

    for (int i=0;i<8;++i){
        temp = nodes(i) - 1;
        u_local(i) = u_global(temp);
    }
    for (int i=0;i<8;++i){
        temp = totnodes + nodes(i) - 1;
        u_local(8+i) = u_global(temp);
    }
    for (int i=0;i<8;++i){
        temp = (2*totnodes) + nodes(i) - 1;
        u_local(16+i) = u_global(temp);
    }
}