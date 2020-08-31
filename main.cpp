#include<iostream>
#include <pybind11/pybind11.h>
#include<pybind11/eigen.h>
#include<Eigen/Core>
#include "plate.hpp"
#include "solve.hpp"

namespace py = pybind11;
using namespace std;
using namespace Eigen;


int add(int a, int b){
    return a+b;
}

void say_hello(){
    cout << "Hello, World!" << endl;
}

// PYBIND11_MODULE(example,m){
//     m.doc() = "pybind11 example plugin";
//     m.def("add",&add,"A function that add two numbers");
// }

PYBIND11_MODULE(topoNN,my_module)
{
    using namespace pybind11::literals;
    my_module.doc() = "example module to export code";
    my_module.def("say_hello", &say_hello);

    py::class_<Plate>(my_module,"Plate")
		// .def(py::init<Vector3i&>())
        .def("__repr__",[]{cout << 5 << endl;});
        // .def("getElemID",&Plate::getElemID);
        // .def("copy_matrix", &MyClass::getMatrix) // Makes a copy!
		// .def_property("ElemXYX", &Plate::getElemXYZ, &Plate::setElemXYZ);

}


//#####################################################################

void setParams();

int main(int argc, char *argv[])
{    
    setParams();
    cout << 5 << endl;
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

//#####################################################################
