import pytest
import numpy as np
from pythonlib.neural_net import NeuralNet


@pytest.fixture
def net_1():
    net = NeuralNet([5, 4, 3, 2])
    return net


def test_generate_variables_1(net_1):
    assert net_1.layerSizes == [5, 4, 3, 2]
    assert net_1.batchSize == 100
    assert net_1.numLayers == 4

    assert np.shape(net_1.y[0]) == (5, 1)
    assert np.shape(net_1.y[1]) == (4, 1)
    assert np.shape(net_1.y[2]) == (3, 1)
    assert np.shape(net_1.y[3]) == (2, 1)
    with pytest.raises(IndexError):
        assert np.shape(net_1.y[4]) == (2,)

    assert np.shape(net_1.w[0]) == (5, 4)
    assert np.shape(net_1.w[1]) == (4, 3)
    assert np.shape(net_1.w[2]) == (3, 2)
    with pytest.raises(IndexError):
        assert np.shape(net_1.w[3]) == (2,)

    assert np.shape(net_1.b[0]) == (4, 1)
    assert np.shape(net_1.b[1]) == (3, 1)
    assert np.shape(net_1.b[2]) == (2, 1)
    with pytest.raises(IndexError):
        assert np.shape(net_1.b[3]) == (2,)

    assert np.shape(net_1.dw[0]) == (5, 4)
    assert np.shape(net_1.dw[1]) == (4, 3)
    assert np.shape(net_1.dw[2]) == (3, 2)
    with pytest.raises(IndexError):
        assert np.shape(net_1.dw[3]) == (2,)

    assert np.shape(net_1.df[0]) == (4, 1)
    assert np.shape(net_1.df[1]) == (3, 1)
    assert np.shape(net_1.df[2]) == (2, 1)
    with pytest.raises(IndexError):
        assert np.shape(net_1.db[3]) == (2,)


def test_architecture(net_1):
    assert np.shape(net_1.y[0]) == (5, 1)


def test_feedforward1():
    """
    test for a single layer, no hidden layers
    """
    net = NeuralNet([1, 1])
    net.w[0] = 2.0
    net.b[0] = 3.0
    act_val = (2.0 * 3.0) + 3.0
    act_val = act_val * (act_val > 0)
    assert net.apply_net([3]) == act_val


def test_feedforward2():
    """
    test for 1 hidden layers, one neuron each
    """
    net = NeuralNet([1, 1, 1])
    net.w[0] = 2.0
    net.b[0] = 3.0
    net.w[1] = 4.0
    net.b[1] = 5.0
    act_val = 4.0 * (2.0 * 6 + 3.0) + 5.0
    act_val = act_val * (act_val > 0)
    assert net.apply_net([6]) == act_val


def test_feedforward3():
    """
    test for 0 hidden layers, two input neurons
    """
    net = NeuralNet([2, 1])
    net.w[0][0, 0] = 2.0
    net.b[0][0] = 3.0
    net.w[0][1, 0] = 4.0
    act_val = (6.0 * 2.0) + (7.0 * 4.0) + 3.0
    act_val = act_val * (act_val > 0)
    assert net.apply_net([6, 7]) == act_val


def test_feedforward4():
    """
    test for a single layer, no hidden layers
    testinf for negative values
    """
    net = NeuralNet([1, 1])
    net.w[0] = 2.0
    net.b[0] = 3.0
    act_val = (2.0 * -5.0) + 3.0
    act_val = act_val * (act_val > 0)
    assert net.apply_net([-5]) == act_val


def test_backpropogate():
    """
    Test for no hidden layers, with lr=0.005 and y=10
    """
    net = NeuralNet([1, 1], batch_size=1)
    net.apply_net([5])
    net.w[0] = 2.0
    net.b[0] = 3.0
    y_act = 10
    lr = 0.005

    y_pred = (2.0 * 5.0) + 3.0
    y_pred = y_pred * (y_pred > 0)
    cost = 0.5 * (y_act - y_pred) ** 2
    dc_dy = -y_pred
    dy_dz = y_pred > 0
    dz_dw = 5
    dz_db = 1
    dc_dw = dc_dy * dy_dz * dz_dw
    dc_db = dc_dy * dy_dz * dz_db
    w0 = 2.0 - (lr * dc_dw)
    b0 = 3.0 - (lr * dc_db)

    net.train_net([5], y_act, lr=0.005)
    np.testing.assert_approx_equal(net.w[0], 1.925)
    np.testing.assert_approx_equal(net.b[0], 2.985)
