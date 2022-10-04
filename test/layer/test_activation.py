import numpy as np
import pytest
import idealflow as flow

@pytest.mark.parametrize("activation_layer, expect_range",
                         [(iflow.layer.Sigmoid(), (0, 1)),
                          (iflow.layer.Tanh(), (-1, 1)),
                          (iflow.layer.ReLU(), (0, np.inf)),
                          (iflow.layer.LeakyReLU(), (-np.inf, np.inf)),
                          (iflow.layer.Softplus(), (0, np.inf))])
def test_activation(activation_layer, expect_range):
    """Test expected output range of activation layers"""
    input_ = np.random.normal(size=(100, 5))
    net = iflow.net.Net([iflow.layer.Dense(1), activation_layer])
    output = net.forward(input_)
    lower_bound, upper_bound = expect_range
    assert np.all((output >= lower_bound) & (output <= upper_bound))
  
