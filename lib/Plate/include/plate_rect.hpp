/**
 * Defining mesh for reactangular plate using cartesian coordinates
*/

#pragma once

#include <Eigen/Dense>

struct Elems{
    int x;
    int y;
    int z;
};

class Plate{
    
    private:
        Elems nel;

    public:
        Plate(Eigen::Vector3i&);

        //variables
        int totelems;
        int totnodes;
        Eigen::VectorXd nodelist;
        Eigen::ArrayXXi elemNodes;
        Eigen::ArrayXXd nodeCoords;

        //structured-grid
        void structInitialize();
        void setElemXYZ(Elems& elem);
        void getElemXYZ(Elems& elem);
        int CoordtoNode(Eigen::Vector3d& coord);
        void NodetoCoord(Eigen::Vector3d& coord, int node);
        void getElemOrigin(Eigen::Vector3d& origin,int elem);
        int getElemID(Eigen::Vector3d& origin);
        void getHex8IDs(Eigen::VectorXi& hexid, Eigen::Vector3d& origin);
        void getLeft(Eigen::VectorXd& left);
        void getRight(Eigen::VectorXd& right);
        void getUp(Eigen::VectorXd& up);
        void getDown(Eigen::VectorXd& down);
        void setValues(Eigen::ArrayXXd& matx, Eigen::VectorXd& pos, 
                        Eigen::Vector3d& val);

        //generate Data
        void genElemNodes();
        void genNodeCoords();

        //sanity checks
        bool checkElemID(int elem);
        bool checkNodeID(int node);
        bool checkCoords(Eigen::Vector3d& coord);
        bool checkOrigin(Eigen::Vector3d& origin);

};
