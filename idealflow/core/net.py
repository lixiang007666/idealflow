"""Feed-forward Neural Network class.
   net 类负责管理 tensor 在 layers 之间的前向和反向传播。
   forward 方法很简单，按顺序遍历所有层，每层计算的输出作为下一层的输入；
   backward 则逆序遍历所有层，将每层的梯度作为下一层的输入。
   这里我们还将每个网络层参数的梯度保存下来返回，后面参数更新需要用到。
   另外 net 类还实现了获取参数、设置参数、获取梯度的接口，也是后面参数更新时需要用到。
   EX:
    net = Net([
    Dense(784, 400),
    ReLU(),
    Dense(400, 100),
    ReLU(),
    Dense(100, 10)
    ])
"""

import copy

from idealflow.utils.structured_param import StructuredParam


class Net:

    def __init__(self, layers):
        self.layers = layers
        self._is_training = True

    def __repr__(self):
        return "\n".join([str(layer) for layer in self.layers])

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, grad):
        # back propagation
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

        # structured gradients
        param_grads = [copy.deepcopy(layer.grads) for layer in self.layers]
        struct_grads = StructuredParam(param_grads)
        # save the gradients w.r.t the input
        struct_grads.wrt_input = grad
        return struct_grads

    # def get_params_and_grads(self):
    #     for layer in self.layers:
    #         yield layer.params, layer.grads

    # def get_parameters(self):
    #     return [layer.params for layer in self.layers]

    # def set_parameters(self, params):
    #     for i, layer in enumerate(self.layers):
    #         for key in layer.params.keys():
    #             layer.params[key] = params[i][key]

    @property
    def params(self):
        trainable = [layer.params for layer in self.layers]
        non_trainable = [layer.nt_params for layer in self.layers]
        return StructuredParam(trainable, non_trainable)

    @params.setter
    def params(self, params):
        self.params.values = params.values
        self.params.nt_values = params.nt_values

    @property
    def is_training(self):
        return self._is_training

    @is_training.setter
    def is_training(self, is_training):
        for layer in self.layers:
            layer.is_training = is_training
        self._is_training = is_training
