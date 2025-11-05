
<img width="512" height="512" alt="0" src="https://github.com/user-attachments/assets/93e5b3c6-03ab-40aa-86b6-17759665250e" /> <img width="512" height="512" alt="0" src="https://github.com/user-attachments/assets/798338d0-0ba5-4f5c-abce-f1a75fc7c41f" />


# Latex-Clusters-Detection
Detection with UNet of clusters of latex
## TB1 : Introduction au traitement d'image
Karim Bekhti Galvao
This project has been done in the context of the class "Introduction to Image Processing" in Mines Saint-Etienne (EMSE). 
___
This projects consists in an example of utilisation of a deep learning architecture (Unet) in order to realize picture's segmentation. The different codes below produce detailed results of the use of such an algorithm, on a relatively small training sample. The results are detailed below and in the PDF document given with this work.

Ce projet consiste en un exemple d'utilisation de deep learning (UNet) pour la segmentation d'images. 
Les différents codes permettant d'obtenir les résultats détaillés dans ci-dessous et dans le PDF sont donnés ci-dessous et commentés. 
___


The architecture of the project is fairly simple, as its an introduction project:
We define important parameters, like the number of epochs, batchsize, testsize in the jupyterfile

## Preprocessing:
The images are resized to 128x128 and are already black and white. They are normalized.

## Hyperparameters
epochs = 16
batch_size = 32
test_size = 0.3
image_size = (128,128)
optimizer = adam
loss = binary_crossentropy
num_class = 1

we add an early stopping mechanism with keras. 

## UNet Structure
### Encoder:
4 layers with: 2 convolutions (2D), RELU activation, no padding + pooling with size (2,2)
### Bottleneck
Dropout of 0.5
### Decoder
4 layers with: 2 convolution (2D), RELU activation, no padding + stride (2,2)
### Output
Sigmoid activatiom

## Result
<img width="251" height="264" alt="image" src="https://github.com/user-attachments/assets/3a9408ba-d071-47e1-a887-247e6a4636e3" />
<img width="251" height="264" alt="image" src="https://github.com/user-attachments/assets/4936c053-5619-4ff3-b575-1daf8c2055f5" />
<img width="306" height="262" alt="image" src="https://github.com/user-attachments/assets/8d960b2c-6d44-4463-8259-cabee913ee32" />

              precision    recall  f1-score   support

           0       0.99      0.89      0.94    219353
           1       0.52      0.95      0.67     26407

    accuracy                           0.90    245760
   macro avg       0.76      0.92      0.81    245760
weighted avg       0.94      0.90      0.91    245760

