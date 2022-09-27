"""Model class manage the network, loss function and optimizer.
   最后 model 类实现了我们一开始设计的三个接口 forward、backward 和 apply_grad,
   先计算损失 loss, 然后反向传播得到梯度，然后 optimizer 计算步长，由 apply_grad 对参数进行更新。
"""

import pickle


class Model:

    def __init__(self, net, loss, optimizer):
        self.net = net
        self.loss = loss
        self.optimizer = optimizer

    # 前向传播
    def forward(self, inputs):
        return self.net.forward(inputs)

    # loss 的反向传播，计算梯度。
    def backward(self, predictions, targets):
        loss = self.loss.loss(predictions, targets)
        grad_from_loss = self.loss.grad(predictions, targets)
        struct_grad = self.net.backward(grad_from_loss)
        return loss, struct_grad

    # 根据参数变换的步长，参数更新。
    def apply_grads(self, grads):
        params = self.net.params
        self.optimizer.step(grads, params)

    # 模型参数的加载与保存。
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.net.params, f)

    def load(self, path):
        with open(path, "rb") as f:
            params = pickle.load(f)

        self.net.params = params
        for layer in self.net.layers:
            layer.is_init = True

    @property
    def is_training(self):
        return self.net.is_training

    @is_training.setter
    def is_training(self, is_training):
        self.net.is_training = is_training
