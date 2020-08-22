#include "solve.hpp"

SolveFE :: SolveFE(ArrayXXi& eN, ArrayXXd& nC, 
                        ArrayXXd& bc, ArrayXXd& fc, Vector3d& matl)
{
    E = matl(0);
    Nu = matl(1);
    sigY = matl(2);
    elemNodes = eN;
    nodeCoords = nC;
    FC = fc;
    BC = bc;
    totelems = elemNodes.col(0).size();
    totnodes = nodeCoords.col(0).size();
    solveInitialize();
}

void SolveFE :: solveInitialize()
{
    ArrayXXd elemcoord(8,3);
    elemCoordinates(elemcoord,6);
}

void SolveFE :: elemCoordinates(ArrayXXd& elemcoord, int elem)
{
    VectorXi nodes(8);
    nodes = elemNodes.row(elem-1);
    for (int i=0;i<8;++i){
        elemcoord.row(i) = nodeCoords.row(nodes(i)-1);
    }
}