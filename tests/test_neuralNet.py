import pytest
import numpy as np
from pythonlib import neuralNet


@pytest.fixture
def Net1():
    Net = neuralNet.neuralNet([5, 4, 3, 2], 10)
    return Net


def test_variableGen_A(Net1):
    assert Net1.layerSizes == [5,4,3,2]
    assert Net1.batchSize == 10
    assert Net1.numLayers == 4

    assert np.shape(Net1.y[0]) == (5,1)
    assert np.shape(Net1.y[1]) == (4,1)
    assert np.shape(Net1.y[2]) == (3,1)
    assert np.shape(Net1.y[3]) == (2,1)
    with pytest.raises(IndexError):
        assert np.shape(Net1.y[4]) == (2,)

    assert np.shape(Net1.w[0]) == (5, 4)
    assert np.shape(Net1.w[1]) == (4, 3)
    assert np.shape(Net1.w[2]) == (3, 2)
    with pytest.raises(IndexError):
        assert np.shape(Net1.w[3]) == (2,)

    assert np.shape(Net1.b[0]) == (4, 1)
    assert np.shape(Net1.b[1]) == (3, 1)
    assert np.shape(Net1.b[2]) == (2, 1)
    with pytest.raises(IndexError):
        assert np.shape(Net1.b[3]) == (2,)

    assert np.shape(Net1.dw[0]) == (5, 4)
    assert np.shape(Net1.dw[1]) == (4, 3)
    assert np.shape(Net1.dw[2]) == (3, 2)
    with pytest.raises(IndexError):
        assert np.shape(Net1.dw[3]) == (2,)

    assert np.shape(Net1.df[0]) == (4, 1)
    assert np.shape(Net1.df[1]) == (3, 1)
    assert np.shape(Net1.df[2]) == (2, 1)
    with pytest.raises(IndexError):
        assert np.shape(Net1.db[3]) == (2,)


def test_architecture(Net1):
    assert np.shape(Net1.y[0]) == (5,1)

