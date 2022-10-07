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
   
 class RAdam(Optimizer):
    """Rectified Adam. Ref: https://arxiv.org/pdf/1908.03265v1.pdf """
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

        self.rho = 2.0 / (1 - self._b2) - 1.0

    def _compute_step(self, grads):
        self._t += 1

        self._m += (1.0 - self._b1) * (grads - self._m)
        self._v += (1.0 - self._b2) * (grads ** 2 - self._v)

        # bias correction
        _m = self._m / (1 - self._b1 ** self._t)

        _rho = self.rho - 2 * self._b2 ** self._t / (1 - self._b2 ** self._t)
        if _rho > 4.0:
            _v = self._v / (1 - self._b2 ** self._t)
            _r = (((_rho - 4) * (_rho - 2) * self.rho) / \
                    ((self.rho - 4) * (self.rho - 2) * _rho)) ** 0.5
            step = -self.lr * _m * _r / (_v ** 0.5 + self._epsilon)
        else:
            step = -self.lr * _m
        return step


class RMSProp(Optimizer):
    """Root Mean Square Prop optimizer
    mean_square = decay * mean_square{t-1} + (1-decay) * grad_t**2
    mom = momentum * mom{t-1} + lr * grad_t / sqrt(mean_square + epsilon)
    """
    def __init__(self,
                 lr=0.01,
                 decay=0.99,
                 momentum=0.0,
                 epsilon=1e-8,
                 weight_decay=0.0):
        super().__init__(lr, weight_decay)
        self._rho = decay
        self._momentum = momentum
        self._epsilon = epsilon

        self._rms = 0
        self._mom = 0

    def _compute_step(self, grads):
        self._rms += (1 - self._rho) * (grads ** 2 - self._rms)
        self._mom = self._momentum * self._mom + self.lr * grads / \
                (self._rms + self._epsilon) ** 0.5
        step = -self._mom
        return step


class Momentum(Optimizer):
    """accumulation = momentum * accumulation + gradient
    variable -= learning_rate * accumulation
    """
    def __init__(self, lr, momentum=0.9, weight_decay=0.0):
        super().__init__(lr, weight_decay)
        self._momentum = momentum
        self._acc = 0

    def _compute_step(self, grads):
        self._acc = self._momentum * self._acc + grads
        step = -self.lr * self._acc
        return step


class Adagrad(Optimizer):
    """AdaGrad optimizer
    accumulation = - (learning_rate / sqrt(G + epsilon)) * gradient
    where G is the element-wise sum of square gradient
    ref: http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf
    """
    def __init__(self, lr, epsilon=1e-8, weight_decay=0.0):
        super().__init__(lr, weight_decay)
        self._g = 0
        self._epsilon = epsilon

    def _compute_step(self, grads):
        self._g += grads ** 2
        adjust_lr = self.lr / (self._g + self._epsilon) ** 0.5
        step = -adjust_lr * grads
        return step


class Adadelta(Optimizer):
    """Adadelta algorithm (https://arxiv.org/abs/1212.5701)"""
    def __init__(self, lr=1.0, decay=0.9, epsilon=1e-8, weight_decay=0.0,):
        super().__init__(lr, weight_decay)
        self._epsilon = epsilon
        self._rho = decay
        self._rms = 0  # running average of square gradient
        self._delta = 0  # running average of delta

    def _compute_step(self, grads):
        self._rms += (1 - self._rho) * (grads ** 2 - self._rms)
        std = (self._delta + self._epsilon) ** 0.5
        delta = grads * (std / (self._rms + self._epsilon) ** 0.5)
        step = - self.lr * delta
        self._delta += (1 - self._rho) * (delta ** 2 - self._delta)
        return step


