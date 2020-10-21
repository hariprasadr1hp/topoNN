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

class Plate
{
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
        void setElemXYZ(Elems&);
        void getElemXYZ(Elems&);
        int CoordtoNode(Eigen::Vector3d&);
        void NodetoCoord(Eigen::Vector3d&,int);
        void getElemOrigin(Eigen::Vector3d&,int);
        int getElemID(Eigen::Vector3d&);
        void getHex8IDs(Eigen::VectorXi&,Eigen::Vector3d&);
        void getLeft(Eigen::VectorXd&);
        void getRight(Eigen::VectorXd&);
        void getUp(Eigen::VectorXd&);
        void getDown(Eigen::VectorXd&);
        void setValues(Eigen::ArrayXXd&,Eigen::VectorXd&,Eigen::Vector3d&);

        //generate Data
        void genElemNodes();
        void genNodeCoords();

        //sanity checks
        bool checkElemID(int);
        bool checkNodeID(int);
        bool checkCoords(Eigen::Vector3d&);
        bool checkOrigin(Eigen::Vector3d&);

};
