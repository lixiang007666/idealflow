"""Data Iterator class."""

from collections import namedtuple

import numpy as np

Batch = namedtuple("Batch", ["inputs", "targets"])


class BaseIterator:

    def __call__(self, inputs, targets):
        raise NotImplementedError


class BatchIterator(BaseIterator):

    def __init__(self, batch_size=32, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle

    # 该方法的功能类似于在类中重载 () 运算符，使得类实例对象可以像调用普通函数那样，以“对象名()”的形式使用。
    """
    EX:
        class CLanguage:
            # 定义__call__方法
            def __call__(self,name,add):
                print("调用__call__()方法",name,add)

        clangs = CLanguage()
        clangs("C语言中文网","http://c.biancheng.net")
    """
    def __call__(self, inputs, targets):
        indices = np.arange(len(inputs))
        if self.shuffle:
            np.random.shuffle(indices)

        starts = np.arange(0, len(inputs), self.batch_size)
        for start in starts:
            end = start + self.batch_size
            batch_inputs = inputs[indices[start: end]]
            batch_targets = targets[indices[start: end]]
            yield Batch(inputs=batch_inputs, targets=batch_targets)
