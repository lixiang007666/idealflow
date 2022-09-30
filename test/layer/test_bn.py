import numpy as np
import pytest
import idealflow as iflow

def test_batch_normalization():
    input_ = np.array([[1.0, 2.0, 3.0, 4.0, 5.0],
                       [5.0, 4.0, 3.0, 2.0, 1.0]])
    mom, epsilon = 0.9, 1e-5
    layer = iflow.layer.BatchNormalization(momentum=mom, epsilon=epsilon)
    for i in range(3):
        layer.forward(input_)
        mean = input_.mean(0, keepdims=True)
        var = input_.var(0, keepdims=True)
        if i == 0:
            r_mean = mean
            r_var = var
        else:
            r_mean = mom * r_mean + (1 - mom) * mean
            r_var = mom * r_var + (1 - mom) * var
        assert np.allclose(layer.ctx["X_norm"],
                           (input_ - mean) / (var + epsilon) ** 0.5)

    layer.is_training = False
    layer.forward(input_)
    assert np.allclose(layer.ctx["X_norm"],
                       (input_ - r_mean) / (r_var + epsilon) ** 0.5)

if __name__=="__main__":
    test_batch_normalization()
