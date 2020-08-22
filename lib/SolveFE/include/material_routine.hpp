#ifndef MATERIAL_ROUTINE_HPP
#define MATERIAL_ROUTINE_HPP

#include <Eigen/Dense>

using namespace Eigen;

class Material
{
    private:
        double E, Nu, sigY;

    public:
        Material();
        Material(Vector3d&);
        void setParams(Vector3d&);
        void getParams(Vector3d&);
        void MaterialInitialize(MatrixXd&,
                                VectorXd&,
                                VectorXd&);
        void Isotropic(MatrixXd&);

        
};






#endif //MATERIAL_ROUTINE_HPP