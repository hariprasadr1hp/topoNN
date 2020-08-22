#ifndef PLATE_HPP
#define PLATE_HPP

#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

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
        Plate(Vector3i&);

        //variables
        int totelems;
        int totnodes;
        VectorXd nodelist;
        ArrayXXi elemNodes;
        ArrayXXd nodeCoords;

        //structured-grid
        void structInitialize();
        void setElemXYZ(Elems&);
        void getElemXYZ(Elems&);
        int CoordtoNode(Vector3d&);
        void NodetoCoord(Vector3d&,int);
        void getElemOrigin(Vector3d&,int);
        int getElemID(Vector3d&);
        void getHex8IDs(VectorXi&,Vector3d&);
        void getLeft(VectorXd&);
        void getRight(VectorXd&);
        void getUp(VectorXd&);
        void getDown(VectorXd&);
        void setValues(ArrayXXd&,VectorXd&,Vector3d&);

        //generate Data
        void genElemNodes();
        void genNodeCoords();

        //sanity checks
        bool checkElemID(int);
        bool checkNodeID(int);
        bool checkCoords(Vector3d&);
        bool checkOrigin(Vector3d&);

};






#endif //PLATE_HPP