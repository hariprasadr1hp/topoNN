#ifndef ELEMENT_ROUTINE_HPP
#define ELEMENT_ROUTINE_HPP

#include <Eigen/Dense>

using namespace Eigen;

class Element
{
    private:

    public:
        Element(ArrayXXd&);
        void elemInitialize(ArrayXXd&);
        // void JacobianMat(MatrixXd&,MatrixXd&,MatrixXd&);

};







#endif //ELEMENT_ROUTINE_HPP