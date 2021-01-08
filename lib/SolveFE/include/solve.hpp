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

class SolveFE{
    
    private:


    public:
        //constructor
        SolveFE(Eigen::ArrayXXi& _elemNodes,
                Eigen::ArrayXXd& _nodeCoords,
                Eigen::ArrayXXd& _BC,
                Eigen::ArrayXXd& _FC,
                Eigen::Vector3d& matl);

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
        void elemCoordinates(Eigen::ArrayXXd& elemCoord,
                                Eigen::VectorXi& nodes);
        void localDisp(Eigen::VectorXd& u_local, Eigen::VectorXi& nodes);
        void connectivityMatrix();
};
