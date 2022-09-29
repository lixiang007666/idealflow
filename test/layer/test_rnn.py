import numpy as np
import pytest
import idealflow as iflow

def test_rnn():
    batch_size = 1
    n_steps, input_dim = 10, 20
    n_hidden = 30
    # num_step是这批数据序列最长的长度, 也就是样本的序列长度（时间步长）；
    input_ = np.random.randn(batch_size, n_steps, input_dim)
    layer = iflow.layer.RNN(num_hidden=n_hidden)
    forward_out = layer.forward(input_)
    assert forward_out.shape == (batch_size, n_hidden)

    fake_grads = np.random.randn(batch_size, n_hidden)
    backward_out = layer.backward(fake_grads)
    # should has the same shape as input_
    assert backward_out.shape == (batch_size, n_steps, input_dim)


def test_lstm():
    batch_size = 1
    n_steps, input_dim = 10, 20
    n_hidden = 30
    input_ = np.random.randn(batch_size, n_steps, input_dim)
    layer = iflow.layer.LSTM(num_hidden=n_hidden)
    forward_out = layer.forward(input_)
    assert forward_out.shape == (batch_size, n_hidden)

    fake_grads = np.random.randn(batch_size, n_hidden)
    backward_out = layer.backward(fake_grads)
    # should has the same shape as input_
    assert backward_out.shape == (batch_size, n_steps, input_dim)

if __name__=="__main__":
    test_rnn()
    test_lstm()
