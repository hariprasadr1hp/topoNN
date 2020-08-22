#ifndef SOLVE_HPP
#define SOLVE_HPP

#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "element_routine.hpp"
#include "material_routine.hpp"

using namespace std;
using namespace Eigen;

struct elems{
    int x;
    int y;
    int z;
};

struct xyz{
    double x;
    double y;
    double z;
};

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
        double E,Nu,sigY;
        
        void solveInitialize();
        void elemCoordinates(ArrayXXd&,int);

};





#endif //SOLVE_FE_HPP
