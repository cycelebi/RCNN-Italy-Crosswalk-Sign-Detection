# Italy Crosswalk Sign Detection with R-CNN

This project implements Region Based Convolution Neural Network (RCNN) to detect crosswalk signs in Italian traffic signs. The code is written in MATLAB and the implementation includes the following steps:

-   Loading the ground truth information about traffic signs.
-   Defining the layers for the object detector.
-   Setting the options for the object detector.
-   Training the object detector.
-   Detecting crosswalk signs in test images.

A detailed explanation of each step is provided in the comments of the MATLAB code.

## Please cite this work as:

M. Celebi, "RCNN Italy Crosswalk Sign Detection,"
 University of Rome "Tor Vergata", Rome, Italy, 2023. 
 Available at: [https://github.com/cycelebi/RCNN-Italy-Crosswalk-Sign-Detection](https://github.com/cycelebi/RCNN-Italy-Crosswalk-Sign-Detection).

## Data Set:

**DITS - Data set of Italian Traffic Signs** used as dataset. You can find dataset [here](http://www.diag.uniroma1.it/~bloisi/ds/dits.html).

> *Youssef, A., Albani, D., Nardi, D., & Bloisi, D. D. (n.d.).* Fast Traffic Sign Recognition Using Color Segmentation and Deep Convolutional Networks. 
> Department of Computer, Control, and Management Engineering, 
> Sapienza University of Rome.

## Preparing Data Set:
Each image used from data set resized to 64x64pixel format. Images are converted with MATLAB Batch Processor.

