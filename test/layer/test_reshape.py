import numpy as np
import pytest
import idealflow as iflow

def test_reshape():
    batch_size = 1
    input_ = np.random.randn(batch_size, 2, 3, 4, 5)
    target_shape = (5, 4, 3, 2)
    layer = iflow.layer.Reshape(*target_shape)
    output = layer.forward(input_)
    assert output.shape[1:] == target_shape
