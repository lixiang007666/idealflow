import idealflow as iflow 
from idealflow import layer

import numpy as np
import time

mnist = iflow.dataset.MNIST(".", one_hot=True)
train_x, train_y = mnist.train_set
test_x, test_y = mnist.test_set
# print(train_x, train_y)

net = iflow.net.Net([
  layer.Dense(784, 400),
  layer.ReLU(),
  layer.Dense(400, 100),
  layer.ReLU(),
  layer.Dense(100, 10)
])

#print(net)

loss = iflow.loss.SoftmaxCrossEntropy()
optimizer = iflow.optimizer.Adam(lr=0.001)
model = iflow.model.Model(net=net, loss=loss, optimizer=optimizer)
print(model)

iterator = iflow.data_loader.BatchIterator(batch_size=1)
print(iterator)


for epoch in range(2):
    t_start = time.time()
    for batch in iterator(train_x, train_y):
        # print(train_x)
        # print(train_y)
        # print(batch.inputs)
        pred = model.forward(batch.inputs)
        loss, grads = model.backward(pred, batch.targets)
        model.apply_grads(grads)
    print(f"Epoch {epoch} time cost: {time.time() - t_start}")

