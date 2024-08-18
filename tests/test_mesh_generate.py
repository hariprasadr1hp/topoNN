import pytest
import numpy as np
from pythonlib.mesh_generate import Plate3D
from pythonlib.mesh_generate import Plate2D


@pytest.fixture
def type_3d():
    mesh = Plate3D(4, 1, 2)
    return mesh


def test_getElem_xyz_3d(type_3d):
    assert type_3d.getElemXYZ() == (4, 1, 2)


def test_coord_to_node_3d(type_3d):
    assert type_3d.coordToNode((3, 1, 2)) == 29


def test_node_to_coord_3d(type_3d):
    assert type_3d.nodeToCoord(29) == (3, 1, 2)


def test_get_elem_origin_3d(type_3d):
    assert type_3d.getElemOrigin(7) == (2, 0, 1)


def test_get_elem_id_3d(type_3d):
    assert type_3d.getElemID((2, 0, 1)) == 7


def test_get_hex_8ids_3d(type_3d):
    np.testing.assert_array_equal(
        type_3d.getHex8IDs((2, 0, 1)), np.array([13, 14, 18, 19, 23, 24, 28, 29])
    )
    np.testing.assert_array_equal(
        type_3d.getHex8IDs((0, 0, 0)), np.array([1, 2, 6, 7, 11, 12, 16, 17])
    )


def test_check_elemid_3d(type_3d):
    assert not type_3d.checkElemID(-2)
    assert not type_3d.checkElemID(0)
    assert type_3d.checkElemID(1)
    assert type_3d.checkElemID(5)
    assert type_3d.checkElemID(8)
    assert not type_3d.checkElemID(9)
    assert not type_3d.checkElemID(24)


def test_check_nodeid_3d(type_3d):
    assert not type_3d.checkNodeID(-2)
    assert not type_3d.checkNodeID(0)
    assert type_3d.checkNodeID(1)
    assert type_3d.checkNodeID(12)
    assert type_3d.checkNodeID(30)
    assert not type_3d.checkNodeID(31)
    assert not type_3d.checkNodeID(112)


def test_check_coords_3d(type_3d):
    assert not type_3d.checkCoords((-1, 0, 0))
    assert not type_3d.checkCoords((0, -1, 0))
    assert not type_3d.checkCoords((0, 0, -1))
    assert type_3d.checkCoords((0, 0, 0))
    assert type_3d.checkCoords((2, 1, 1))
    assert type_3d.checkCoords((4, 1, 2))
    assert not type_3d.checkCoords((5, 1, 2))
    assert not type_3d.checkCoords((4, 2, 2))
    assert not type_3d.checkCoords((4, 1, 3))


def test_check_origin_3d(type_3d):
    assert type_3d.checkOrigin((0, 0, 0))
    assert type_3d.checkOrigin((1, 0, 0))
    assert type_3d.checkOrigin((0, 0, 1))
    assert not type_3d.checkOrigin((4, 1, 2))
    assert not type_3d.checkOrigin((0, 1, 0))
    assert not type_3d.checkOrigin((5, 1, 2))


def test_get_up_3d(type_3d):
    np.testing.assert_array_equal(
        type_3d.getUp(), np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    )


def test_get_down_3d(type_3d):
    np.testing.assert_array_equal(
        type_3d.getDown(), np.array([21, 22, 23, 24, 25, 26, 27, 28, 29, 30])
    )


def test_get_left_3d(type_3d):
    np.testing.assert_array_equal(type_3d.getLeft(), np.array([1, 6, 11, 16, 21, 26]))


def test_get_right_3d(type_3d):
    np.testing.assert_array_equal(type_3d.getRight(), np.array([5, 10, 15, 20, 25, 30]))


###############################################################


@pytest.fixture
def type2D():
    mesh = Plate2D(10, 10)
    return mesh


def test_getElemXY_2D(type2D):
    assert type2D.getElemXY() == (10, 10)


def test_coordToNode_2D(type2D):
    assert type2D.coordToNode((2, 1)) == 14


def test_nodeToCoord_2D(type2D):
    assert type2D.nodeToCoord(14) == (2, 1)


def test_getElemOrigin_2D(type2D):
    assert type2D.getElemOrigin(13) == (2, 1)
    assert type2D.getElemOrigin(20) == (9, 1)


def test_getElemID_2D(type2D):
    assert type2D.getElemID((2, 1)) == 13


def test_getQuadIDs_2D(type2D):
    np.testing.assert_array_equal(type2D.getQuadIDs((3, 1)), np.array([15, 16, 26, 27]))
    np.testing.assert_array_equal(type2D.getQuadIDs((9, 1)), np.array([21, 22, 32, 33]))


def test_checkElemID_2D(type2D):
    assert not type2D.checkElemID(-2)
    assert not type2D.checkElemID(0)
    assert type2D.checkElemID(1)
    assert type2D.checkElemID(35)
    assert type2D.checkElemID(100)
    assert not type2D.checkElemID(101)
    assert not type2D.checkElemID(124)


def test_checkNodeID_2D(type2D):
    assert not type2D.checkNodeID(-2)
    assert not type2D.checkNodeID(0)
    assert type2D.checkNodeID(1)
    assert type2D.checkNodeID(82)
    assert type2D.checkNodeID(121)
    assert not type2D.checkNodeID(122)
    assert not type2D.checkNodeID(132)


def test_checkCoords_2D(type2D):
    assert not type2D.checkCoords((-1, 0))
    assert not type2D.checkCoords((0, -1))
    assert type2D.checkCoords((0, 0))
    assert type2D.checkCoords((2, 1))
    assert type2D.checkCoords((10, 10))
    assert not type2D.checkCoords((11, 10))
    assert not type2D.checkCoords((10, 11))
    assert not type2D.checkCoords((4, 11))


def test_checkOrigin_2D(type2D):
    assert not type2D.checkOrigin((-1, -1))
    assert type2D.checkOrigin((0, 0))
    assert type2D.checkOrigin((1, 0))
    assert type2D.checkOrigin((0, 1))
    assert not type2D.checkOrigin((10, 10))
    assert not type2D.checkOrigin((10, 5))
    assert not type2D.checkOrigin((5, 10))


def test_getLeft_2D(type2D):
    np.testing.assert_array_equal(
        type2D.getLeft(), np.array([1, 12, 23, 34, 45, 56, 67, 78, 89, 100, 111])
    )


def test_getRight_2D(type2D):
    np.testing.assert_array_equal(
        type2D.getRight(), np.array([11, 22, 33, 44, 55, 66, 77, 88, 99, 110, 121])
    )


def test_getUp_2D(type2D):
    np.testing.assert_array_equal(
        type2D.getUp(),
        np.array([111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121]),
    )


def test_getDown_2D(type2D):
    np.testing.assert_array_equal(
        type2D.getDown(), np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    )


# _____________________________________________________________________
