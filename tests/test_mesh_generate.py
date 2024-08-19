import numpy as np
import pytest

from ingest.generate.mesh import Plate


@pytest.fixture
def type_2d():
    mesh = Plate(10, 10)
    return mesh


def test_get_elem_xy(type_2d):
    assert type_2d.get_elem_xy() == (10, 10)


def test_coord_to_node(type_2d):
    assert type_2d.coord_to_node((2, 1)) == 14


def test_node_to_coord(type_2d):
    assert type_2d.node_to_coord(14) == (2, 1)


def test_get_elem_origin(type_2d):
    assert type_2d.get_elem_origin(13) == (2, 1)
    assert type_2d.get_elem_origin(20) == (9, 1)


def test_get_elem_id(type_2d):
    assert type_2d.get_elem_id((2, 1)) == 13


def test_get_quad_ids(type_2d):
    np.testing.assert_array_equal(
        type_2d.get_quad_ids((3, 1)), np.array([15, 16, 26, 27])
    )
    np.testing.assert_array_equal(
        type_2d.get_quad_ids((9, 1)), np.array([21, 22, 32, 33])
    )


def test_check_elem_id(type_2d):
    assert not type_2d.check_elem_id(-2)
    assert not type_2d.check_elem_id(0)
    assert type_2d.check_elem_id(1)
    assert type_2d.check_elem_id(35)
    assert type_2d.check_elem_id(100)
    assert not type_2d.check_elem_id(101)
    assert not type_2d.check_elem_id(124)


def test_check_node_id(type_2d):
    assert not type_2d.check_node_id(-2)
    assert not type_2d.check_node_id(0)
    assert type_2d.check_node_id(1)
    assert type_2d.check_node_id(82)
    assert type_2d.check_node_id(121)
    assert not type_2d.check_node_id(122)
    assert not type_2d.check_node_id(132)


def test_check_coords(type_2d):
    assert not type_2d.check_coords((-1, 0))
    assert not type_2d.check_coords((0, -1))
    assert type_2d.check_coords((0, 0))
    assert type_2d.check_coords((2, 1))
    assert type_2d.check_coords((10, 10))
    assert not type_2d.check_coords((11, 10))
    assert not type_2d.check_coords((10, 11))
    assert not type_2d.check_coords((4, 11))


def test_check_origin(type_2d):
    assert not type_2d.check_origin((-1, -1))
    assert type_2d.check_origin((0, 0))
    assert type_2d.check_origin((1, 0))
    assert type_2d.check_origin((0, 1))
    assert not type_2d.check_origin((10, 10))
    assert not type_2d.check_origin((10, 5))
    assert not type_2d.check_origin((5, 10))


def test_get_left(type_2d):
    np.testing.assert_array_equal(
        type_2d.get_left(), np.array([1, 12, 23, 34, 45, 56, 67, 78, 89, 100, 111])
    )


def test_get_right(type_2d):
    np.testing.assert_array_equal(
        type_2d.get_right(), np.array([11, 22, 33, 44, 55, 66, 77, 88, 99, 110, 121])
    )


def test_get_up(type_2d):
    np.testing.assert_array_equal(
        type_2d.get_up(),
        np.array([111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121]),
    )


def test_get_down(type_2d):
    np.testing.assert_array_equal(
        type_2d.get_down(), np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    )
