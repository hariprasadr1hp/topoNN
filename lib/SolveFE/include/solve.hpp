#ifndef SOLVE_HPP
#define SOLVE_HPP

#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "element_routine.hpp"

using namespace std;
using namespace Eigen;

struct dim{
    int x;
    int y;
    int z;
};

struct xyz{
    double x;
    double y;
    double z;
};

typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::Triplet<double> T;

class SolveFE
{
    private:


    public:
        //constructor
        SolveFE(ArrayXXi&,ArrayXXd&,ArrayXXd&,ArrayXXd&,Vector3d&);

        //public variables
        int totelems, totnodes;
        ArrayXXi elemNodes;
        ArrayXXd nodeCoords;
        ArrayXXd BC;
        ArrayXXd FC;
        Vector3d matlParams;
        VectorXd Fint_global;
        VectorXd Fext_global;
        VectorXd u_global;
        SpMat K_global;

        // double E,Nu,sigY;
        
        void solveInitialize();
        void elemCoordinates(ArrayXXd&,VectorXi&);
        void localDisp(VectorXd&,VectorXi&);
        void connectivityMatrix();
};





#endif //SOLVE_FE_HPP
