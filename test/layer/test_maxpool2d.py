import numpy as np
import pytest
import idealflow as iflow

def test_max_pool_2d():
    batch_size = 1
    channel = 2
    input_ = np.random.randn(batch_size, 4, 4, channel).astype(np.float32)

    layer = iflow.layer.MaxPool2D(pool_size=[2, 2], stride=[2, 2])
    output = layer.forward(input_)
    assert output.shape == (batch_size, 2, 2, channel)

    layer = iflow.layer.MaxPool2D(pool_size=[4, 4], stride=[2, 2])
    output = layer.forward(input_)
    answer = np.max(np.reshape(input_, (batch_size, -1, 2)), axis=1)
    assert (output.ravel() == answer.ravel()).all() # Test whether all array elements along a given axis evaluate to True.

if __name__=="__main__":
    test_max_pool_2d()