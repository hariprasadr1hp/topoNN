#include<iostream>
#include<Eigen/Core>
#include "plate.hpp"
#include "solve.hpp"

using namespace std;
using namespace Eigen;

void setParams();


int main(int argc, char *argv[])
{    
    setParams();
    return 0;
}

void setParams()
{
    //define elements
    Vector3i nel = {4,1,2};
    
    //initialize Plate Object
    Plate plate(nel);

    //define boundary conditions
    Vector3d disp = {-1,0,0};
    Vector3d force = {0,500,0};

    ArrayXXd BC,FC;
    VectorXd left, right;
    plate.getLeft(left);
    plate.getRight(right);

    plate.setValues(BC,left,disp);
    plate.setValues(FC,right,force);
    
    //define material parameters
    Vector3d matl = {2000000000, 0.3, 415000000};
    
    //initialize solver object  
    SolveFE solve(
        plate.elemNodes,
        plate.nodeCoords,
        BC,
        FC,
        matl
    );

    // cout << solve.elemNodes << endl;

    // cout << solve.totelems << endl;
    // cout << solve.totnodes << endl;
    // cout << "------------------" << endl;
    // cout << solve.elemNodes << endl;
    // cout << "------------------" << endl;
    // cout << solve.nodeCoords << endl;
    // cout << "------------------" << endl;
    
}

