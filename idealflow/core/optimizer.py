"""Various optimization algorithms and learning rate schedulers.
   optimizer 主要实现一个接口 compute_step, 这个方法根据当前的梯度, 计算返回实际优化时每个参数改变的步长。
"""

import numpy as np


class Optimizer:

    def __init__(self, lr, weight_decay):
        self.lr = lr
        self.weight_decay = weight_decay

    def step(self, grads, params):
        # compute the gradient step
        grads = self.compute_step(grads)
        # apply weight_decay if specified
        if self.weight_decay:
            grads -= self.lr * self.weight_decay * params
        # take a step
        params += grads

    def compute_step(self, grads):
        grads.values = self._compute_step(grads.values)
        return grads

    def _compute_step(self, grads):
        raise NotImplementedError


# 随机梯度下降
class SGD(Optimizer):

    def __init__(self, lr=0.01, weight_decay=0.0):
        super().__init__(lr, weight_decay)

    def _compute_step(self, grads):
        return -self.lr * grads


class Adam(Optimizer):

    def __init__(self,
                 lr=0.001,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-8,
                 weight_decay=0.0):
        super().__init__(lr, weight_decay)
        self._b1 = beta1
        self._b2 = beta2
        self._epsilon = epsilon

        self._t = 0
        self._m = 0
        self._v = 0

    def _compute_step(self, grads):
        self._t += 1

        self._m += (1.0 - self._b1) * (grads - self._m)
        self._v += (1.0 - self._b2) * (grads ** 2 - self._v)

        # bias correction
        _m = self._m / (1 - self._b1 ** self._t)
        _v = self._v / (1 - self._b2 ** self._t)

        step = -self.lr * _m / (_v ** 0.5 + self._epsilon)
        return step

