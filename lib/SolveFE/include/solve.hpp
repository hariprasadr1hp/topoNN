/**
 * Solving the FE
*/

#pragma once

#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "element_routine.hpp"


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
        SolveFE(Eigen::ArrayXXi&,Eigen::ArrayXXd&,Eigen::ArrayXXd&,Eigen::ArrayXXd&,Eigen::Vector3d&);

        //public variables
        int totelems, totnodes;
        Eigen::ArrayXXi elemNodes;
        Eigen::ArrayXXd nodeCoords;
        Eigen::ArrayXXd BC;
        Eigen::ArrayXXd FC;
        Eigen::Vector3d matlParams;
        Eigen::VectorXd Fint_global;
        Eigen::VectorXd Fext_global;
        Eigen::VectorXd u_global;
        SpMat K_global;

        // double E,Nu,sigY;
        
        void solveInitialize();
        void elemCoordinates(Eigen::ArrayXXd&,Eigen::VectorXi&);
        void localDisp(Eigen::VectorXd&,Eigen::VectorXi&);
        void connectivityMatrix();
};
