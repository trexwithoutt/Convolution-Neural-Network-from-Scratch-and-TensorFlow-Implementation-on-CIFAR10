## Data

CIFAR dataset:

	- `CIFAR` https://www.cs.toronto.edu/~kriz/cifar.html

## Data Description

The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. 

The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class. 

<img src="https://github.com/trexwithoutt/Convolution-Neural-Network-from-Scratch-and-TensorFlow-Implementation-on-CIFAR10/blob/master/cifar10.png" width="600">

## Files

```
.
├── cnn-tf.ipynb
├── cnn_tf.py
├── cnn.py
├── cnn.ipynb
└── README.md
```
## Introduction

`.ipynb` are demo for CNN implementation and classification on CIFAR-10 dataset.

`cnn` is an implementation from scratch with feed forward and backprobagation

`cnn-tf` contains 5 convolution layers with relu activation and 2 fully-connected layers output with a softmax probabilities.

## Result

`test-accuracy` can reach 0.7527 with `test-loss` 1.98008

Training can be down in 29 min with GPU