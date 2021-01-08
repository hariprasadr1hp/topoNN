/**
 * Defining mesh for reactangular plate using cartesian coordinates
*/

#include "plate_rect.hpp"
#include <iostream>


Plate :: Plate(Eigen::Vector3i& elem){
    nel.x = elem(0);
    nel.y = elem(1);
    nel.z = elem(2);
    structInitialize();
}

void Plate :: structInitialize(){
    totelems = nel.x * nel.y * nel.z;
    totnodes = (nel.x + 1) * (nel.y + 1) * (nel.z + 1);
    nodelist.resize(totnodes);
    nodelist = Eigen::VectorXd::LinSpaced(totnodes,1,totnodes);
    genElemNodes();
    genNodeCoords();
}

/**
 * sets the no. of elements in each direction
 */
void Plate :: setElemXYZ(Elems& elem){
    nel.x = elem.x;
    nel.y = elem.y;
    nel.z = elem.z;
    structInitialize();
}

/**
 * retrieves the no. of elements in each direction
 */
void Plate :: getElemXYZ(Elems& elem){
    elem.x = nel.x;
    elem.y = nel.y;
    elem.z = nel.z;
}

/**
 * Returns the Global node ID for the given origin coordinates
 * 
 * @param coord The origin coordinates of the Element
*/
int Plate :: CoordtoNode(Eigen::Vector3d& coord){   
    if (checkCoords(coord)){
        int node = (coord(2) * (nel.x+1) * (nel.y+1)) + 
                    (coord(1) * (nel.x+1)) + coord(0) + 1;
        return node;
    }
    else{
        return -1;
    }
}


/**
 * Returns the coordinates (x,y,z) for the given Global ID
 * 
 * @param coord The coordinates of the Node
 * @param node node ID
*/
void Plate :: NodetoCoord(Eigen::Vector3d& coord, int node){
    if (checkNodeID(node)){
        node = node - 1;
        coord(2) = node / ((nel.x+1) * (nel.y+1));
        int temp = node % ((nel.x+1) * (nel.y+1));
        coord(1) = temp / (nel.x + 1);
        coord(0) = temp % (nel.x + 1);
    }
    else{
        coord = {-1,-1,-1};
    }
}

/**
 * Returns coordinates (x,y,z) of the element's origin
 * 
 * @param origin The origin coordinates of the Node
 * @return elem Element ID
*/    
void Plate :: getElemOrigin(Eigen::Vector3d& origin, int elem){
        int x,y,z,temp1,temp2;
        temp1 = elem - 1;
        z = temp1 / (nel.x*nel.y);
        temp2 = temp1 % (nel.x*nel.y);
        y = temp2 / nel.x;
        x = temp2 % nel.x;
        origin  = {double(x),double(y),double(z)};
}

/**
 * Returns the ID of the element, given its origin
 * 
 * @param origin The origin coordinates of the Element
 * @return elem Element ID
*/
int Plate :: getElemID(Eigen::Vector3d& origin){
    if (checkOrigin(origin)){
        int elem;
        elem = int(
            (origin(2) * nel.x * nel.y) + 
            (origin(1) * nel.x) + 
            origin(0) + 1
        );
        return elem;
    }
    else{
        return -1;
    }
}

/**
 * Writes the global IDs of the nodes of an element, given the 
 * coordinates of an element at its origin
 * 
 * @param hexid The node IDs of the element
 * @param origin The origin coordinates of the element
*/
void Plate :: getHex8IDs(Eigen::VectorXi& hexid, Eigen::Vector3d& origin){
    // hexid(8);
    if (hexid.size() != 8){
        std::cout << "Size doesn't match" << std::endl;
    }
    if (checkOrigin(origin)){
        hexid(0) = CoordtoNode(origin);
        hexid(1) = hexid(0) + 1;
        hexid(2) = hexid(0) + (nel.x + 1);
        hexid(3) = hexid(2) + 1;
        hexid(4) = hexid(0) + ((nel.x + 1) * (nel.y + 1));
        hexid(5) = hexid(4) + 1;
        hexid(6) = hexid(4) + (nel.x + 1);
        hexid(7) = hexid(6) + 1;
    }
    else{
        std::cout << "Origin not valid" << std::endl;
    }
}

/**
 * checks whether an element with the following ID exist
 */
bool Plate :: checkElemID(int elem){
    // return (elemNodes.col(0) == elem).any();
    return true ? (elem > 0 && elem <= totelems) : false;
}

/**
 * checks whether a node with the following ID exist
 */
bool Plate :: checkNodeID(int node){
    // return (nodeCoords.col(0) == node).any();
    return true ? (node > 0 && node <= totnodes) : false;
}

/**
 * checks whether the node coordinates exist
 */
bool Plate :: checkCoords(Eigen::Vector3d& coords){
    if ((coords(0) >= nel.x+1) || (coords(0) < 0)){
        return false;
    }
    if ((coords(1) >= nel.y+1) || (coords(1) < 0)){
        return false;
    }
    if ((coords(2) >= nel.z+1) || (coords(2) < 0)){
        return false;
    }
    return true;
}

/**
 * checks whether the origin coordinates exist
 */
bool Plate :: checkOrigin(Eigen::Vector3d& origin){
    if ((origin(0) >= nel.x) || (origin(0) < 0)){
        return false;
    }
    if ((origin(1) >= nel.y) || (origin(1) < 0)){
        return false;
    }
    if ((origin(2) >= nel.z) || (origin(2) < 0)){
        return false;
    }
    return true;
}

/// Generates Element-Nodes Connectivity Database
void Plate :: genElemNodes(){
    Eigen::VectorXi hex8(8);
    Eigen::Vector3d origin;
    // elemNodes.resize(totelems,9);
    elemNodes.resize(totelems,8);
    elemNodes.setZero();

    // elemNodes.col(0) = ArrayXi::LinSpaced(totelems,1,totelems);
    for (int i=0; i<totelems; ++i){
        getElemOrigin(origin, i+1);
        getHex8IDs(hex8, origin);
        // elemNodes.row(i).tail(8) = hex8.array();
        elemNodes.row(i) = hex8.array();
    }

}

/**
 * Generates Node-Coordinates Database
 */
void Plate :: genNodeCoords(){
    Eigen::Vector3d coords;
    // nodeCoords.resize(totnodes,4);
    nodeCoords.resize(totnodes,3);
    nodeCoords.setZero();

    // nodeCoords.col(0) = ArrayXd::LinSpaced(totnodes,1,totnodes);
    for (int i=0; i<totnodes; i++){
        NodetoCoord(coords,i+1);
        // nodeCoords.row(i).tail(3) = coords.array();
        nodeCoords.row(i) = coords.array();
    }

}

/**
 * Writes the node IDs on the left side to a vector
 */
void Plate :: getLeft(Eigen::VectorXd& left){   
    int temp,count;
    temp = 2*(nel.z+1);
    left.resize(temp);
    count=0;
    for (int i=1;i<totnodes;i+=nel.x+1){
        left(count) = double(i);
        count+=1;
    }

}

/**
 * Writes the node IDs on the right side to a vector
 */
void Plate :: getRight(Eigen::VectorXd& right){
    int temp,count;
    temp = 2*(nel.z+1);
    right.resize(temp);
    count=0;
    for (int i=nel.x+1;i<=totnodes;i+=nel.x+1){
        right(count) = double(i);
        count+=1;
    }
}

/**
 * Writes the node IDs on the top side to a vector
 */
void Plate :: getUp(Eigen::VectorXd& top){
    int temp,count;
    temp = 2*(nel.x+1);
    top.resize(temp);
    count=0;
    for (int i=1;i<=((nel.x+1)*(nel.y+1));++i){
        top(count) = double(i);
        count+=1;
    }
}

/**
 * Writes the node IDs on the bottom side to a vector
 */
void Plate :: getDown(Eigen::VectorXd& bottom){
    int temp,count;
    temp = (nel.x+1)*(nel.y+1);
    bottom.resize(temp);
    count=0;
    for (int i=totnodes-temp+1;i<=totnodes;++i){
        bottom(count) = double(i);
        count+=1;
    }
}

void Plate :: setValues(Eigen::ArrayXXd& matx, Eigen::VectorXd& pos, Eigen::Vector3d& val){
    int nodes;
    nodes = pos.size();
    matx.resize(nodes,4);
    matx.setZero();
    pos.data();
    matx.col(0) = pos.array();
    for (int i=0; i<nodes; i++){
        matx.row(i).tail(3) = val.array();
    }
}