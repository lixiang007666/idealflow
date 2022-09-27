import idealflow as iflow 
from idealflow import layer

import numpy as np
import time

def evaluate(model, test_x, test_y):
    model.is_training = False
    test_pred = model.forward(test_x)
    test_pred_idx = np.argmax(test_pred, axis=1)
    test_y_idx = np.argmax(test_y, axis=1)
    accuracy, info = iflow.metric.accuracy(test_pred_idx, test_y_idx)
    model.is_training = True
    print(f"accuracy: {accuracy:.4f} info: {info}")


if __name__=="__main__":
    mnist = iflow.dataset.MNIST(".", one_hot=True)
    train_x, train_y = mnist.train_set
    test_x, test_y = mnist.test_set
    # print(train_x, train_y)

    net = iflow.net.Net([
      layer.Dense(784),
      layer.ReLU(),
      layer.Dense(400),
      layer.ReLU(),
      layer.Dense(100),
      layer.ReLU(),
      layer.Dense(10),
    ])

    #print(net)

    loss = iflow.loss.MSE()
    optimizer = iflow.optimizer.Adam(lr=0.001)
    model = iflow.model.Model(net=net, loss=loss, optimizer=optimizer)
    print(model)

    iterator = iflow.data_loader.BatchIterator(batch_size=64)


    for epoch in range(2):
        t_start = time.time()
        for batch in iterator(train_x, train_y):
            # print(train_x)
            # print(train_y)
            # print(batch.inputs)
            pred = model.forward(batch.inputs)
            loss, grads = model.backward(pred, batch.targets)
            print(loss)
            model.apply_grads(grads)
        print(f"Epoch {epoch} time cost: {time.time() - t_start}")
        evaluate(model, test_x, test_y)

