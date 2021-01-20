import pytest
import numpy as np
from pythonlib.meshGen import Plate3D


@pytest.fixture
def typeA():
    mesh = Plate3D(4,1,2)
    return mesh

@pytest.fixture
def typeB():
    mesh = Plate3D(10,1,10)
    return mesh


def test_getUp_A(typeA):
    np.testing.assert_array_equal(typeA.getUp(), np.array([1,2,3,4,5,6,7,8,9,10]))

def test_getDown_A(typeA):
    np.testing.assert_array_equal(typeA.getDown(), np.array([21,22,23,24,25,26,27,28,29,30]))

def test_getLeft_A(typeA):
    np.testing.assert_array_equal(typeA.getLeft(), np.array([1,6,11,16,21,26]))

def test_getRight_A(typeA):
    np.testing.assert_array_equal(typeA.getRight(), np.array([5,10,15,20,25,30]))
