import numpy as np
import pytest
import idealflow as iflow


def test_dropout():
    batch_size, input_dim = 100, 1000
    input_ = np.ones((batch_size, input_dim))
    keep_prob = 0.5
    layer = iflow.layer.Dropout(keep_prob=keep_prob)
    forward_out = layer.forward(input_)
    assert forward_out.shape == input_.shape
    keep_rate = 1. - (forward_out == 0.).sum() / (batch_size * input_dim)
    # varify keep_prob
    assert np.abs(keep_rate - keep_prob) < 1e-1
    # constent expectations
    assert np.abs(forward_out.mean() - input_.mean()) < 1e-1

    backward_out = layer.backward(input_)
    assert (backward_out == forward_out).all()

    layer.is_training = False
    forward_out = layer.forward(input_)
    assert (forward_out == input_).all()
