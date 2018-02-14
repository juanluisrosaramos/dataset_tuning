# Dataset Tuning
Face alignment and Data augmentation are two tasks necessaries for trainining a Deep CNN when you can't have enough data available.

Considering how a Convolutional Neural Network learns its weights it is priority to scale in a uniform way the input training vectors. Normalisation improves the performance and stability of neural networks many facial recognition algorithms. This is due to the fact that a CNN learns by continually adding gradient error vectors (multiplied by a learning rate) computed from backpropagation to various weight matrices throughout the network as training examples are passed through. If we didn't scale our input training vectors, the ranges of our distributions of feature values would likely be different for each feature, and thus the learning rate would cause corrections in each dimension that would differ (proportionally speaking) from one another.

## Face alignment
[Based in Adrian Rosebock code](https://www.pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/)
For normalising the dataset it will be employed a computer vision technique called warping transformations that works detecting semantic facial landmarks (eg, eyes, nose, mouth corners) and rotating that such eyes lie on a horizontal line. The code lies on DLIB library that includes face deteccctor and landmarks detectorsace. Face will be detected with HOG descriptors and then landmarks will detected and eyes selected, a central point will be detected (up in the nose) and then image rotated around this central point. Face will be cropped using DLIB library for just use a 25% of the face.

Apart from having installed openCV2 it is necessary the libraries: imutils and dlib

$ pip install --upgrade imutils
$ pip install --upgrade dlib

![alt tag](https://raw.githubusercontent.com/juanluisrosaramos/dataset_tuning/master/alignment.png)

## Dataset augmentation
