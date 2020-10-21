/**
 * Main file
*/

#include<iostream>
#include<Eigen/Core>
#include "plate_rect.hpp"
#include "solve.hpp"

//#####################################################################

void setParams();

int main(int argc, char *argv[])
{    
    setParams();
    std::cout << 5 << std::endl;
    return 0;
}

void setParams()
{
    //define elements
    Eigen::Vector3i nel = {4,1,2};
    
    //initialize Plate Object
    Plate plate(nel);

    //define boundary conditions
    Eigen::Vector3d disp = {-1,0,0};
    Eigen::Vector3d force = {0,500,0};

    Eigen::ArrayXXd BC,FC;
    Eigen::VectorXd left, right;
    plate.getLeft(left);
    plate.getRight(right);

    plate.setValues(BC,left,disp);
    plate.setValues(FC,right,force);
    
    //define material parameters
    Eigen::Vector3d matl = {2000000000, 0.3, 415000000};
    
    //initialize solver object  
    SolveFE solve(
        plate.elemNodes,
        plate.nodeCoords,
        BC,
        FC,
        matl
    );

    std::cout << solve.elemNodes << std::endl;

    // cout << solve.totelems << endl;
    // cout << solve.totnodes << endl;
    // cout << "------------------" << endl;
    // cout << solve.elemNodes << endl;
    // cout << "------------------" << endl;
    // cout << solve.nodeCoords << endl;
    // cout << "------------------" << endl;
    
}

//#####################################################################
