#include "element_routine.hpp"
#include <iostream>

Element :: Element(ArrayXXd& eleminfo)
{
    elemInitialize(eleminfo);
}

void Element :: elemInitialize(ArrayXXd& eleminfo)
{
    
}

// void JacobianMat(MatrixXd& Jmat ,const MatrixXd& dN_nat, const MatrixXd& hex8_rc)
// {
//     hex8_rc.dot(dN_nat);
//     // Jmat = hex8_rc;
// }