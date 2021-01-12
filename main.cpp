/**
 * Main file
*/

#include <iostream>
#include <Eigen/Core>
#include "plate_rect.hpp"
#include "solve.hpp"

//#####################################################################

void setParams();

int main(int argc, char *argv[]){
    std::cout << "-------------------------" << std::endl;
    setParams();

    return 0;
}

void setParams(){
    //define elements
    Eigen::Vector3i nel = {4,1,2};
    
    //initialize Plate Object
    Plate plate(nel);

    //define boundary conditions
    Eigen::Vector3d disp = {-1,0,0};
    Eigen::Vector3d force = {0,500,0};

    Eigen::ArrayXXd BC,FC;
    Eigen::VectorXd left, right, up, down;
    plate.getLeft(left);
    plate.getRight(right);
    plate.getUp(up);
    plate.getDown(down);

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

    // printf("Total elements: %d\n",solve.totelems);
    // printf("Total nodes: %d\n",solve.totnodes);
    // std::cout << "------------------" << std::endl;
    // std::cout << solve.elemNodes << std::endl;
    // std::cout << "------------------" << std::endl;
    // std::cout << solve.nodeCoords << std::endl;
    // std::cout << "------------------" << std::endl;
    
    std::cout << left << std::endl;
    std::cout << "------------------" << std::endl;
    std::cout << right << std::endl;
    std::cout << "------------------" << std::endl;
    std::cout << up << std::endl;
    std::cout << "------------------" << std::endl;
    std::cout << down << std::endl;
    std::cout << "------------------" << std::endl;
}

//#####################################################################
