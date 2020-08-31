#include <gtest/gtest.h>
#include <plate.hpp>

struct TestPlate : testing :: Test
{
    Plate* plate;

    TestPlate(){
        Vector3i nel = {4,1,2};
        plate = new Plate(nel);
    }

    ~TestPlate(){
        delete plate;
    }
};


TEST_F(TestPlate,elem1){
    Vector3d test_coord1 = {0,0,0};
    Vector3d test_coord2 = {2,0,0};
    EXPECT_EQ(plate->getElemID(test_coord1), 1);
}

TEST_F(TestPlate,elem2){
    Vector3d test_coord = {3,0,0};
    EXPECT_EQ(plate->getElemID(test_coord),4);
}

TEST_F(TestPlate,node1){
    Vector3d coord;
    Vector3d test_coord = {3,0,0};
    plate->NodetoCoord(coord,4);
    EXPECT_EQ(coord,test_coord);
}