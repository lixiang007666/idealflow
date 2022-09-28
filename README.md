[![License](https://img.shields.io/npm/l/@nrwl/workspace.svg?style=flat-square)]()
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


## Model Zoo and Benchmark

 - [mnist](https://github.com/lixiang007666/idealflow/blob/main/test/example/mnist.py)

## Related Tutorials

 - [img2col](https://cloud.tencent.com/developer/article/2127875)

## Test

TODO
