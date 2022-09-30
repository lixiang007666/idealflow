[![License](https://img.shields.io/npm/l/@nrwl/workspace.svg?style=flat-square)]()
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-pink.svg)](https://github.com/lixiang007666/idealflow/pulls)

# A Lightweight Deep Learning Library: IdealFlow

IdealFlow is a lightweight deep learning framework used for personal learning (building wheels is important), use: `import idealflow as iflow`.

## How to install？

Execute the following commands in the root dict:

```
python setup.py build
python setup.py install
```

## Supported Features

| Layer and Activation | Loss | Optimizer | Initializer | 
|--|--|--|--|
| Dense | MSE | SGD | Normal/TruncatedNormal |
| ReLU | MAE | Adam | HeUniform/HeNormal |
| Conv2d | Huber | | Uniform |
| TransposedConv2d | SoftmaxCrossEntropy | | XavierUniform/XavierNormal |
| Maxpool2d | SigmoidCrossEntropy | | Zeros/Ones |
| RNN/LSTM | | |


## Model Zoo and Benchmark

 - [mnist](https://github.com/lixiang007666/idealflow/blob/main/test/example/mnist.py)

## Related Tutorials

 - [img2col](https://cloud.tencent.com/developer/article/2127875)
 - [AD code](https://cloud.tencent.com/developer/article/2129649)
 - [Several common opt methods](https://www.cnblogs.com/shixiangwan/p/7532830.html)
 - [Momentum](https://zhuanlan.zhihu.com/p/34240246)
 - [Adam](https://www.jiqizhixin.com/graph/technologies/f41c192d-9c93-4306-8c47-ce4bf10030dd)
 - [Initializer](https://zhuanlan.zhihu.com/p/40175178)
 - [Batch Normalization](https://zhuanlan.zhihu.com/p/24810318)


## References

 - [tinynn](https://zhuanlan.zhihu.com/p/78713744)
 - [AD](https://zhuanlan.zhihu.com/p/82582926)


