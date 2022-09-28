import numpy as np
import pytest
import idealflow as iflow

def test_conv_2d():
    batch_size = 1
    input_ = np.random.randn(batch_size, 16, 16, 1)

    # test forward and backward correctness
    """
    padding=valid表示without padding(dropped), padding=same表示with zero padding

    设输入图像尺寸为W, 卷积核尺寸为F, 步幅为S, Padding使用P, 则经过卷积层或池化层之后的图像尺寸为 (W-F+2P)/S+1。

    (1)如果参数是 SAME, 那么计算只与步长有关，直接除以步长 W / S(除不尽，向上取整)
    (2)如果参数是 VALID, 那么计算公式如上：(W - F + 1) / S （结果向上取整）
    
    """
    layer = iflow.layer.Conv2D(
        kernel=[4, 4, 1, 2], stride=[3, 3], padding="VALID")
    output = layer.forward(input_)
    assert output.shape == (batch_size, 5, 5, 2)
    input_grads = layer.backward(output)
    assert input_grads.shape == input_.shape

    layer = iflow.layer.Conv2D(
        kernel=[4, 4, 1, 2], stride=[3, 3], padding="SAME")
    output = layer.forward(input_)
    assert output.shape == (batch_size, 6, 6, 2)
    input_grads = layer.backward(output)
    assert input_grads.shape == input_.shape

def test_conv_transpose_2d():
    batch_size = 1
    input_ = np.random.randn(batch_size, 7, 7, 2)

    # test forward and backward correctness
    layer = iflow.layer.ConvTranspose2D(
        kernel=[4, 4, 2, 1], stride=[3, 3], padding="VALID")
    output = layer.forward(input_)
    assert output.shape == (batch_size, 22, 22, 1)
    input_grads = layer.backward(output)
    assert input_grads.shape == input_.shape

    layer = iflow.layer.ConvTranspose2D(
        kernel=[4, 4, 2, 1], stride=[3, 3], padding="SAME")
    output = layer.forward(input_)
    assert output.shape == (batch_size, 21, 21, 1)
    input_grads = layer.backward(output)
    assert input_grads.shape == input_.shape

if __name__=="__main__":
    test_conv_2d()
    test_conv_transpose_2d()