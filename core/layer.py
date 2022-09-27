"""Network layers."""

import numpy as np
from idealflow.core.initializer import Ones
from idealflow.core.initializer import XavierUniform
from idealflow.core.initializer import Zeros
from idealflow.utils.math import sigmoid


def empty(shape, dtype=np.float32):
    return np.empty(shape, dtype=dtype)


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
            self.shapes[name] = self.params[name].shape

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

        self.initializers = {"w": w_init, "b": b_init}
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
