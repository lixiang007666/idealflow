"""Network layers."""

import numpy as np
from idealflow.core.initializer import Ones
from idealflow.core.initializer import XavierUniform, HeUniform
from idealflow.core.initializer import Zeros
from idealflow.utils.math import sigmoid
from idealflow.core.utils import get_padding_2d, im2col, empty


class Layer:
    """Base class for layers.
       网络层需要有提供 forward 和 backward 接口进行对应的运算,同时还应该将该层的参数和梯度记录下来。
    """

    def __init__(self):
        self.params = {p: None for p in self.param_names}
        self.nt_params = {p: None for p in self.nt_param_names}
        self.initializers = {}

        self.grads = {}
        self.shapes = {}

        self._is_training = True  # used in BatchNorm/Dropout layers
        self._is_init = False

        self.ctx = {}

    # Use: print(repr(Layer()))
    def __repr__(self):
        shape = None if not self.shapes else self.shapes
        return f"layer: {self.name}\tshape: {shape}"

    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError

    @property
    def is_init(self):
        return self._is_init

    # @*.setter装饰器必须在@property装饰器的后面，且两个被修饰的函数的名称必须保持一致。
    """
       Example:
       ```
        class User():
            def __init__(self, name, age):
                self.name = name
                self._age = age

            @property
            def age(self):
                return self._age

            @age.setter
            def age(self,n):
                self._age = n + 5

        user = User('xiao',0)
        user.age = 5

        print(user.age)
        # 当执行 user.age = 5 时，@age.setter装饰器下的age函数会将数据+5后再存入类属性_age中, 实现了存入前对数据的预处理。
       ```
    """
    @is_init.setter
    def is_init(self, is_init):
        self._is_init = is_init
        for name in self.param_names:
            self.shapes[name] = self.initializers[name](self.shapes[name])

    @property
    def is_training(self):
        return self._is_training

    @is_training.setter
    def is_training(self, is_train):
        self._is_training = is_train

    # Return impl layer name.
    @property
    def name(self):
        return self.__class__.__name__

    @property
    def param_names(self):
        return ()

    @property
    def nt_param_names(self):
        return ()

    def _init_params(self):
        for name in self.param_names:
            # param_names: ('w', 'b')
            # print(self.param_names)
            # print(self.initializers[name])
            # print(self.shapes[name])
            # # print(self.initializers[name](self.shapes[name]))
            # print(self.params['w'])
            # init = self.initializers[name]
            # print(init)
            self.params[name] = self.initializers[name](self.shapes[name])
        self.is_init = True


class Dense(Layer):
    """A dense layer operates `outputs = dot(intputs, weight) + bias`
    :param num_out: A positive integer, number of output neurons
    :param w_init: Weight initializer
    :param b_init: Bias initializer
    最基础的一种网络层是全连接网络层, 实现如下。forward 方法接收上层的输入 inputs, 实现 wx+b 的运算; 
    backward 的方法接收来自上层的梯度，计算关于参数 w,b 和输入的梯度, 然后返回关于输入的梯度。
    """
    def __init__(self,
                 num_out,
                 w_init=XavierUniform(),
                 b_init=Zeros()):
        super().__init__()

        self.initializers = {"w": XavierUniform(), "b": Zeros()}
        self.shapes = {"w": [None, num_out], "b": [num_out]}

    def forward(self, inputs):
        if not self.is_init:
            # 初始化 self.shapes，自动获取上一层的 num_out, EX: Dense(784, 400)。
            self.shapes["w"][0] = inputs.shape[1]
            self._init_params()
        self.ctx = {"X": inputs}
        return inputs @ self.params["w"] + self.params["b"]

    def backward(self, grad):
        self.grads["w"] = self.ctx["X"].T @ grad
        self.grads["b"] = np.sum(grad, axis=0)
        return grad @ self.params["w"].T

    @property
    def param_names(self):
        return "w", "b"

"""
激活函数可以看做是一种网络层，同样需要实现 forward 和 backward 方法。
通过继承 Layer 类实现激活函数类，这里实现了最常用的 ReLU 激活函数。
func 和 derivation_func 方法分别实现对应激活函数的正向计算和梯度计算
"""
class Activation(Layer):

    def __init__(self):
        super().__init__()
        self.inputs = None

    def forward(self, inputs):
        self.ctx["X"] = inputs
        return self.func(inputs)

    def backward(self, grad):
        return self.derivative(self.ctx["X"]) * grad

    def func(self, x):
        raise NotImplementedError

    def derivative(self, x):
        raise NotImplementedError

class ReLU(Activation):

    def func(self, x):
        return np.maximum(x, 0.0)

    def derivative(self, x):
        return x > 0.0


class Conv2D(Layer):
    """Implement 2D convolution layer
    :param kernel: A list/tuple of int that has length 4 (height, width,
        in_channels, out_channels)
    :param stride: A list/tuple of int that has length 2 (height, width)
    :param padding: String ["SAME", "VALID"]
    :param w_init: Weight initializer
    :param b_init: Bias initializer
    """
    def __init__(self,
                 kernel,
                 stride=(1, 1),
                 padding="SAME",
                 w_init=XavierUniform(),
                 b_init=Zeros()):
        super().__init__()

        self.kernel_shape = kernel
        self.stride = stride
        self.initializers = {"w": XavierUniform(), "b": Zeros()}
        self.shapes = {"w": self.kernel_shape, "b": self.kernel_shape[-1]}

        self.padding_mode = padding
        self.padding = None

    def forward(self, inputs):
        """Accelerate convolution via im2col trick.
        Blog: https://blog.csdn.net/LoseInVain/article/details/109906065
        An example (assuming only one channel and one filter):
         input = | 43  16  78 |         kernel = | 4  6 |
          (X)    | 34  76  95 |                  | 7  9 |
                 | 35   8  46 |
        After im2col and kernel flattening:
         col  = | 43  16  34  76 |     kernel = | 4 |
                | 16  78  76  95 |      (W)     | 6 |
                | 34  76  35   8 |              | 7 |
                | 76  95   8  46 |              | 9 |
        """
        if not self.is_init:
            self._init_params()

        k_h, k_w, _, out_c = self.kernel_shape
        s_h, s_w = self.stride
        X = self._inputs_preprocess(inputs)

        # padded inputs to column matrix
        col = im2col(X, k_h, k_w, s_h, s_w)
        # perform convolution by matrix product.
        W = self.params["w"].reshape(-1, out_c)
        Z = col @ W
        # reshape output
        batch_sz, in_h, in_w, _ = X.shape
        # separate the batch size and feature map dimensions
        Z = Z.reshape(batch_sz, Z.shape[0] // batch_sz, out_c)
        # further divide the feature map in to (h, w) dimension
        out_h = (in_h - k_h) // s_h + 1
        out_w = (in_w - k_w) // s_w + 1
        Z = Z.reshape(batch_sz, out_h, out_w, out_c)

        # plus the bias for every filter
        Z += self.params["b"]
        # save results for backward function
        self.ctx = {"X_shape": X.shape, "col": col, "W": W}
        return Z

    def backward(self, grad):
        """Compute gradients w.r.t. layer parameters and backward gradients.
        :param grad: gradients from previous layer
            with shape (batch_sz, out_h, out_w, out_c)
        :return d_in: gradients to next layers
            with shape (batch_sz, in_h, in_w, in_c)
        """
        # read size parameters
        k_h, k_w, in_c, out_c = self.kernel_shape
        s_h, s_w = self.stride
        batch_sz, in_h, in_w, in_c = self.ctx["X_shape"]
        pad_h, pad_w = self.padding[1:3]

        # grads w.r.t. parameters
        flat_grad = grad.reshape((-1, out_c))
        d_W = self.ctx["col"].T @ flat_grad
        self.grads["w"] = d_W.reshape(self.kernel_shape)
        self.grads["b"] = np.sum(flat_grad, axis=0)

        # grads w.r.t. inputs
        d_X = grad @ self.ctx["W"].T
        # cast gradients back to original shape as d_in
        d_in = np.zeros(shape=self.ctx["X_shape"], dtype=np.float32)
        for i, r in enumerate(range(0, in_h - k_h + 1, s_h)):
            for j, c in enumerate(range(0, in_w - k_w + 1, s_w)):
                patch = d_X[:, i, j, :]
                patch = patch.reshape((batch_sz, k_h, k_w, in_c))
                d_in[:, r:r+k_h, c:c+k_w, :] += patch

        # cut off gradients of padding
        d_in = d_in[:, pad_h[0]:in_h-pad_h[1], pad_w[0]:in_w-pad_w[1], :]
        return self._grads_postprocess(d_in)

    def _inputs_preprocess(self, inputs):
        _, in_h, in_w, _ = inputs.shape
        k_h, k_w, _, _ = self.kernel_shape
        # padding calculation
        if self.padding is None:
            self.padding = get_padding_2d(
                (in_h, in_w), (k_h, k_w), self.padding_mode)
        return np.pad(inputs, pad_width=self.padding, mode="constant")

    def _grads_postprocess(self, grads):
        return grads

    @property
    def param_names(self):
        return "w", "b"


class ConvTranspose2D(Conv2D):
    """
        https://blog.csdn.net/qq_27261889/article/details/86304061
    """
    def __init__(self,
                 kernel,
                 stride=(1, 1),
                 padding="SAME",
                 w_init=XavierUniform(),
                 b_init=Zeros()):
        super().__init__(kernel, stride, padding, XavierUniform(), Zeros())
        self.origin_stride = stride
        self.stride = (1, 1)

    def _inputs_preprocess(self, inputs):
        k_h, k_w = self.kernel_shape[:2]
        # insert zeros to inputs
        inputs = self._insert_zeros(
            inputs, *self.origin_stride, self.padding_mode)
        _, in_h, in_w, _ = inputs.shape
        # padding calculation
        if self.padding is None:
            if self.padding_mode == "SAME":
                self.padding = get_padding_2d(
                    (in_h, in_w), (k_h, k_w), self.padding_mode)
            else:
                self.padding = ((0, 0), (k_h - 1, k_h - 1),
                                (k_w - 1, k_w - 1), (0, 0))
        return np.pad(inputs, pad_width=self.padding, mode="constant")

    def _grads_postprocess(self, grads):
        return grads[:, ::self.origin_stride[0], ::self.origin_stride[1], :]

    @staticmethod
    def _insert_zeros(inputs, s_h, s_w, mode):
        batch_sz, in_h, in_w, in_c = inputs.shape
        if mode == "SAME":
            out_h = in_h * s_h
            out_w = in_w * s_w
        else:
            out_h = (in_h - 1) * s_h + 1
            out_w = (in_w - 1) * s_h + 1
        expand = np.zeros((batch_sz, out_h, out_w, in_c))
        expand[:, ::s_h, ::s_w, :] = inputs
        return expand


class MaxPool2D(Layer):

    def __init__(self,
                 pool_size=(2, 2),
                 stride=None,
                 padding="VALID"):
        """Implement 2D max-pooling layer
        :param pool_size: A list/tuple of 2 integers (pool_height, pool_width)
        :param stride: A list/tuple of 2 integers (stride_height, stride_width)
        :param padding: A string ("SAME", "VALID")
        """
        super().__init__()
        self.kernel_shape = pool_size
        self.stride = stride if stride is not None else pool_size

        self.padding_mode = padding
        self.padding = None

    def forward(self, inputs):
        s_h, s_w = self.stride
        k_h, k_w = self.kernel_shape
        batch_sz, in_h, in_w, in_c = inputs.shape

        # zero-padding
        if self.padding is None:
            self.padding = get_padding_2d(
                (in_h, in_w), (k_h, k_w), self.padding_mode)
        X = np.pad(inputs, pad_width=self.padding, mode="constant")
        padded_h, padded_w = X.shape[1:3]

        out_h = (padded_h - k_h) // s_h + 1
        out_w = (padded_w - k_w) // s_w + 1

        # construct output matrix and argmax matrix
        max_pool = empty((batch_sz, out_h, out_w, in_c))
        argmax = empty((batch_sz, out_h, out_w, in_c), dtype=int)
        for r in range(out_h):
            r_start = r * s_h
            for c in range(out_w):
                c_start = c * s_w
                pool = X[:, r_start: r_start+k_h, c_start: c_start+k_w, :]
                pool = pool.reshape((batch_sz, -1, in_c))

                _argmax = np.argmax(pool, axis=1)[:, np.newaxis, :]
                argmax[:, r, c, :] = _argmax.squeeze()

                # get max elements
                _max_pool = np.take_along_axis(pool, _argmax, axis=1).squeeze()
                max_pool[:, r, c, :] = _max_pool

        self.ctx = {"X_shape": X.shape, "out_shape": (out_h, out_w),
                    "argmax": argmax}
        return max_pool

    def backward(self, grad):
        batch_sz, in_h, in_w, in_c = self.ctx["X_shape"]
        out_h, out_w = self.ctx["out_shape"]
        s_h, s_w = self.stride
        k_h, k_w = self.kernel_shape
        k_sz = k_h * k_w
        pad_h, pad_w = self.padding[1:3]

        d_in = np.zeros(shape=(batch_sz, in_h, in_w, in_c), dtype=np.float32)
        for r in range(out_h):
            r_start = r * s_h
            for c in range(out_w):
                c_start = c * s_w
                _argmax = self.ctx["argmax"][:, r, c, :]
                mask = np.eye(k_sz)[_argmax].transpose((0, 2, 1))
                _grad = grad[:, r, c, :][:, np.newaxis, :]
                patch = np.repeat(_grad, k_sz, axis=1) * mask
                patch = patch.reshape((batch_sz, k_h, k_w, in_c))
                d_in[:, r_start:r_start+k_h, c_start:c_start+k_w, :] += patch

        # cut off gradients of padding
        return d_in[:, pad_h[0]: in_h-pad_h[1], pad_w[0]: in_w-pad_w[1], :]


class RNN(Layer):

    def __init__(self,
                 num_hidden,
                 w_init=XavierUniform(),
                 b_init=Zeros()):
        super().__init__()
        self.n_h = num_hidden
        self.initializers = {"W": XavierUniform(), "b": Zeros()}


    def forward(self, inputs):
        batch_size, n_ts, input_dim = inputs.shape
        if not self.is_init:
            self.shapes = {"W": [self.n_h, self.n_h + input_dim],
                           "b": [self.n_h]}
            self._init_params()

        a = empty((batch_size, n_ts, self.n_h))
        h = empty((batch_size, n_ts + 1, self.n_h))

        h[:, -1] = 0.0
        for t in range(n_ts):
            z = np.hstack([h[:, t - 1], inputs[:, t]])
            a[:, t] = z @ self.params["W"].T + self.params["b"]
            h[:, t] = np.tanh(a[:, t])
        self.ctx = {"h": h, "X": inputs}
        return h[:, -2]

    def backward(self, grad):
        n_ts = self.ctx["X"].shape[1]
        for p in self.param_names:
            self.grads[p] = np.zeros_like(self.params[p])

        d_in = np.empty_like(self.ctx["X"], dtype=np.float32)
        d_h = grad
        for t in reversed(range(n_ts)):
            d_a = d_h * (1 - self.ctx["h"][:, t] ** 2)
            d_in[:, t] = d_a @ self.params["W"][:, self.n_h:]
            self.grads["W"][:, self.n_h:] += d_a.T @ self.ctx["X"][:, t]
            self.grads["W"][:, :self.n_h] += d_a.T @ self.ctx["h"][:, t - 1]
            self.grads["b"] += d_a.sum(axis=0)
            d_h = d_a @ self.params["W"][:, :self.n_h]
        return d_in

    @property
    def param_names(self):
        return "W", "b"


class LSTM(Layer):

    def __init__(self,
                 num_hidden,
                 w_init=XavierUniform(),
                 b_init=Zeros()):
        super().__init__()
        self.n_h = num_hidden
        self.initializers = {"W_g": XavierUniform(), "W_c": XavierUniform(),
                             "b_g": Zeros(), "b_c": Zeros()}

    def forward(self, inputs):
        batch_size, n_ts, input_dim = inputs.shape
        if not self.is_init:
            self.shapes = {"W_g": [3 * self.n_h, input_dim + self.n_h],
                           "b_g": [3 * self.n_h],
                           "W_c": [self.n_h, input_dim + self.n_h],
                           "b_c": [self.n_h]}
            self._init_params()

        h = empty((batch_size, n_ts + 1, self.n_h))
        h[:, -1] = 0.0
        c = empty((batch_size, n_ts + 1, self.n_h))
        c[:, -1] = 0.0
        gates = empty((batch_size, n_ts, 3 * self.n_h))
        c_hat = empty((batch_size, n_ts + 1, self.n_h))

        for t in range(n_ts):
            z = np.hstack([h[:, t-1], inputs[:, t]])

            gates[:, t] = sigmoid(z @ self.params["W_g"].T + self.params["b_g"])
            o_gate, i_gate, f_gate = np.split(gates[:, t], 3, axis=1)

            c_hat[:, t] = np.tanh(z @ self.params["W_c"].T + self.params["b_c"])

            c[:, t] = f_gate * c[:, t - 1] + i_gate * c_hat[:, t]
            h[:, t] = o_gate * np.tanh(c[:, t])

        self.ctx = {"h": h, "c": c, "X": inputs, "gates": gates, "c_hat": c_hat}
        return h[:, -2]

    def backward(self, grad):
        for p in self.param_names:
            self.grads[p] = np.zeros_like(self.params[p])

        _, n_ts, input_dim = self.ctx["X"].shape

        d_in = np.empty_like(self.ctx["X"], dtype=np.float32)
        d_c_prev = 0
        d_h_prev = grad
        for t in reversed(range(n_ts)):
            z = np.hstack([self.ctx["h"][:, t-1], self.ctx["X"][:, t]])
            tanhc = np.tanh(self.ctx["c"][:, t])

            g_o, g_i, g_f = np.split(self.ctx["gates"][:, t], 3, axis=1)

            d_h = d_h_prev
            d_o = d_h * tanhc
            d_a_o = d_o *  g_o * (1 - g_o)

            self.grads["W_g"][:self.n_h] += d_a_o.T @ z
            self.grads["b_g"][:self.n_h] += d_a_o.sum(axis=0)

            d_c = d_h * g_o * (1 - tanhc ** 2)
            d_c += d_c_prev

            d_c_hat = d_c * g_i
            d_a_c = d_c_hat * (1 - self.ctx["c_hat"][:, t] ** 2)
            self.grads["W_c"] += d_a_c.T @ z
            self.grads["b_c"] += d_a_c.sum(axis=0)

            d_i = d_c * self.ctx["c_hat"][:, t]
            d_a_i = d_i * g_i * (1 - g_i)
            self.grads["W_g"][self.n_h: 2 * self.n_h] += d_a_i.T @ z
            self.grads["b_g"][self.n_h: 2 * self.n_h] += d_a_i.sum(axis=0)

            d_f = d_c * self.ctx["c"][:, t-1]
            d_a_f = d_f * g_f * (1 - g_f)
            self.grads["W_g"][-self.n_h:] += d_a_f.T @ z
            self.grads["b_g"][-self.n_h:] += d_a_f.sum(axis=0)

            d_z = (np.hstack([d_a_o, d_a_i, d_a_f]) @ self.params["W_g"] +
                   d_a_c @ self.params["W_c"])
            d_h = d_z[:, :self.n_h]
            d_in[:, t] = d_z[:, -input_dim:]

            d_h_prev = d_h
            d_c_prev = g_f * d_c
        return d_in

    @property
    def param_names(self):
        return "W_g", "b_g", "W_c", "b_c"

class BatchNormalization(Layer):

    def __init__(self,
                 momentum=0.99,
                 gamma_init=Ones(),
                 beta_init=Zeros(),
                 epsilon=1e-5):
        super().__init__()
        self.m = momentum
        self.epsilon = epsilon
        # gamma 扩展参数
        # beta 平移参数
        self.initializers = {"gamma": Ones(), "beta": Zeros()}
        self.reduce = None

    def forward(self, inputs):
        # self.reduce = (0,) if inputs.ndim == 2 else (0, 1, 2)
        self.reduce = (0,)
        if not self.is_init:
            for p in self.param_names:
                self.shapes[p] = inputs.shape[-1]
            self._init_params()

        if self.nt_params["r_mean"] is None:
            self.nt_params["r_mean"] = inputs.mean(self.reduce, keepdims=True)
            self.nt_params["r_var"] = inputs.var(self.reduce, keepdims=True)

        if self.is_training:
            mean = inputs.mean(self.reduce, keepdims=True)
            var = inputs.var(self.reduce, keepdims=True)
            self.nt_params["r_mean"] = (self.m * self.nt_params["r_mean"] +
                                        (1.0 - self.m) * mean)
            self.nt_params["r_var"] = (self.m * self.nt_params["r_var"] +
                                       (1.0 - self.m) * var)
        else:
            mean = self.nt_params["r_mean"]
            var = self.nt_params["r_var"]

        # standardize
        X_center = inputs - mean
        std = (var + self.epsilon) ** 0.5
        X_norm = X_center / std
        self.ctx = {"X_norm": X_norm, "std": std, "X_center": X_center}
        return self.params["gamma"] * X_norm + self.params["beta"]

    def backward(self, grad):
        # grads w.r.t. params
        self.grads["gamma"] = (self.ctx["X_norm"] * grad).sum(self.reduce)
        self.grads["beta"] = grad.sum(self.reduce)

        # N = grad.shape[0]
        N = np.prod([grad.shape[d] for d in self.reduce])
        std_inv = 1.0 / self.ctx["std"]
        # grads w.r.t. inputs
        # ref: http://cthorey.github.io./backpropagation/
        d_in = (1.0 / N) * self.params["gamma"] * std_inv * (
            N * grad - np.sum(grad, axis=self.reduce, keepdims=True) -
            self.ctx["X_center"] * std_inv ** 2 *
            (grad * self.ctx["X_center"]).sum(axis=self.reduce, keepdims=True))
        return d_in

    @property
    def param_names(self):
        return "gamma", "beta"

    @property
    def nt_param_names(self):
        return "r_mean", "r_var"
