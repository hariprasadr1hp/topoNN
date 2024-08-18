import pytest
import numpy as np
from pythonlib.mesh_generate import Plate2D


@pytest.fixture
def type_2d():
    mesh = Plate2D(10, 10)
    return mesh


def test_get_elem_xy(type_2d):
    assert type_2d.getElemXY() == (10, 10)


def test_coord_to_node(type_2d):
    assert type_2d.coordToNode((2, 1)) == 14


def test_node_to_coord(type_2d):
    assert type_2d.nodeToCoord(14) == (2, 1)


def test_get_elem_origin(type_2d):
    assert type_2d.getElemOrigin(13) == (2, 1)
    assert type_2d.getElemOrigin(20) == (9, 1)


def test_get_elem_id(type_2d):
    assert type_2d.getElemID((2, 1)) == 13


def test_get_quad_ids(type_2d):
    np.testing.assert_array_equal(
        type_2d.getQuadIDs((3, 1)), np.array([15, 16, 26, 27])
    )
    np.testing.assert_array_equal(
        type_2d.getQuadIDs((9, 1)), np.array([21, 22, 32, 33])
    )


def test_check_elem_id(type_2d):
    assert not type_2d.checkElemID(-2)
    assert not type_2d.checkElemID(0)
    assert type_2d.checkElemID(1)
    assert type_2d.checkElemID(35)
    assert type_2d.checkElemID(100)
    assert not type_2d.checkElemID(101)
    assert not type_2d.checkElemID(124)


def test_check_node_id(type_2d):
    assert not type_2d.checkNodeID(-2)
    assert not type_2d.checkNodeID(0)
    assert type_2d.checkNodeID(1)
    assert type_2d.checkNodeID(82)
    assert type_2d.checkNodeID(121)
    assert not type_2d.checkNodeID(122)
    assert not type_2d.checkNodeID(132)


def test_check_coords(type_2d):
    assert not type_2d.checkCoords((-1, 0))
    assert not type_2d.checkCoords((0, -1))
    assert type_2d.checkCoords((0, 0))
    assert type_2d.checkCoords((2, 1))
    assert type_2d.checkCoords((10, 10))
    assert not type_2d.checkCoords((11, 10))
    assert not type_2d.checkCoords((10, 11))
    assert not type_2d.checkCoords((4, 11))


def test_check_origin(type_2d):
    assert not type_2d.checkOrigin((-1, -1))
    assert type_2d.checkOrigin((0, 0))
    assert type_2d.checkOrigin((1, 0))
    assert type_2d.checkOrigin((0, 1))
    assert not type_2d.checkOrigin((10, 10))
    assert not type_2d.checkOrigin((10, 5))
    assert not type_2d.checkOrigin((5, 10))


def test_get_left(type_2d):
    np.testing.assert_array_equal(
        type_2d.getLeft(), np.array([1, 12, 23, 34, 45, 56, 67, 78, 89, 100, 111])
    )


def test_get_right(type_2d):
    np.testing.assert_array_equal(
        type_2d.getRight(), np.array([11, 22, 33, 44, 55, 66, 77, 88, 99, 110, 121])
    )


def test_get_up(type_2d):
    np.testing.assert_array_equal(
        type_2d.getUp(),
        np.array([111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121]),
    )


def test_get_down(type_2d):
    np.testing.assert_array_equal(
        type_2d.getDown(), np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    )
