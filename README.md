# MTCNN-pi

We are aiming to implement the real-time face recognition based on Multi-task Convolution Neuron Network on Raspberry Pi 3B. 

This project was developed by our develop team based on 
- Previous work from the [Github Project](https://github.com/Pi-DeepLearning/RaspberryPi-FaceDetection-MTCNN-Caffe-With-Motion)
- Original Paper:
[*Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks*](https://arxiv.org/abs/1604.02878) (K. Zhang, Z. Zhang, Z. Li, Y. Qiao, 2016).

Credit to our team members: 
* Jinghong Chen, [@EriChen0615](https://github.com/EriChen0615) on Github; jc2124@cam.ac.uk
* Xiaoqiao Hu, xh297@cam.ac.uk
* Connor Wang, [@wonnor-pro](https://github.com/wonnor-pro) on Github; xw345@cam.ac.uk

- - -

## Hardware

* Raspberry Pi 3B (Operating system: Raspbian)
* Camera Module

## Software

* Python 3.5.3

## Settings 

Click the link to see the installastion instruction.

1. [**Environment Setup and Dependencies**](https://tech.connorx.wang/2019/08/06/MTCNN-dependencies/)
- Install dependencies
- Check `Python` version and install `pip3`
- Install `Protobuf 2.6.1`
- Install `OpenCV 3.3.0`
- Install `caffe`
2. [**Install the Camera Module**](https://tech.connorx.wang/2019/08/06/MTCNN-camera)
- Install the camera hardware
- Configure the Raspberry Pi
- Test the camera module

## File Structrues

You should be able to fine three demo files in this repo as we use different methods to speed up the process. It is noteworthy that the accuracy varies from method to method.

- demo_slow.py
- demo_scales.py
- demo_multiprocess.py

Requisitions regarding reposting please contact wonnor.cam@gmail.com.