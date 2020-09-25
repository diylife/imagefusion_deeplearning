# An Infrared and Visible Image Fusion Algorithm Based on ResNet-152
To improve the details of the fusion image from the infrared and visible images by reducing artifacts and noise, an infrared and visible image fusion algorithm based on ResNet-152 is proposed. First, the source images are decomposed into the low-frequency part and the high-frequency part. The low-frequency part is processed by the average weighting strategy. Second, the multi-layer features are extracted from high-frequency part by using the ResNet-152 network. Regularization L1, convolution operation, bilinear interpolation upsampling and maximum selection strategy on the feature layers to obtain the maximum weight layer. Multiplying the maximum weight layer and the high-frequency as new high-frequency. Finally, the fusion image is reconstructed by the low-frequency and the high-frequency. Experiments show that the proposed method can get more details from the image texture by retaining the significant features of the images. In addition, this method can effectively reduce artifacts and noise. The consistency in the subjective evaluation and objective evaluation performs superior to the comparative algorithms.

Developer Liming Zhang(zhanglm8@gmail.com)

Contact address Faculty of Geomatics, Lanzhou Jiaotong University, Lanzhou 730070, China

Year first available 2020

Telephone number 0086-931-4957229

Program language Matlab language

Hardware required and software required Matlab language compiler

Program size 8.62 KB
