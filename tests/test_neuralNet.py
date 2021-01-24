import pytest
import numpy as np
from pythonlib.neuralNet import neuralNet


@pytest.fixture
def Net1():
    Net = neuralNet([5, 4, 3, 2])
    return Net

def test_variableGen_A(Net1):
    assert Net1.layerSizes == [5,4,3,2]
    assert Net1.batchSize == 100
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


def test_feedforward1():
    """
    test for a single layer, no hidden layers
    """
    Net = neuralNet([1,1])
    Net.w[0] = 2.0
    Net.b[0] = 3.0
    act_val = (2.0 * 3.0) + 3.0
    act_val = act_val * (act_val > 0)
    assert Net.applyNet([3]) == act_val


def test_feedforward2():
    """
    test for 1 hidden layers, one neuron each
    """
    Net = neuralNet([1,1,1])
    Net.w[0] = 2.0
    Net.b[0] = 3.0
    Net.w[1] = 4.0
    Net.b[1] = 5.0
    act_val = 4.0 * (2.0 * 6 + 3.0) + 5.0
    act_val = act_val * (act_val > 0)
    assert Net.applyNet([6]) == act_val

def test_feedforward3():
    """
    test for 0 hidden layers, two input neurons
    """
    Net = neuralNet([2,1])
    Net.w[0][0,0] = 2.0
    Net.b[0][0] = 3.0
    Net.w[0][1,0] = 4.0
    act_val = (6.0 * 2.0) + (7.0 * 4.0) + 3.0
    act_val = act_val * (act_val > 0)
    assert Net.applyNet([6,7]) == act_val

def test_feedforward4():
    """
    test for a single layer, no hidden layers
    testinf for negative values
    """
    Net = neuralNet([1,1])
    Net.w[0] = 2.0
    Net.b[0] = 3.0
    act_val = (2.0 * -5.0) + 3.0
    act_val = act_val * (act_val > 0)
    assert Net.applyNet([-5]) == act_val

def test_backprop():
    """
    Test for no hidden layers, with lr=0.005 and y=10
    """
    Net = neuralNet([1,1], batchSize=1)
    Net.applyNet([5])
    Net.w[0] = 2.0
    Net.b[0] = 3.0
    y_act = 10
    lr = 0.005

    y_pred = (2.0 * 5.0) + 3.0
    y_pred = y_pred * (y_pred > 0)
    cost = 0.5 * (y_act - y_pred)**2
    dCdy = (-y_pred)
    dydz = (y_pred > 0)
    dzdw = 5
    dzdb = 1
    dCdw = dCdy * dydz * dzdw
    dCdb = dCdy * dydz * dzdb
    w0 = 2.0 - (lr * dCdw)
    b0 = 3.0 - (lr * dCdb)
    
    Net.trainNet([5], y_act, lr=0.005)
    np.testing.assert_approx_equal(Net.w[0], 1.925)
    np.testing.assert_approx_equal(Net.b[0], 2.985)
