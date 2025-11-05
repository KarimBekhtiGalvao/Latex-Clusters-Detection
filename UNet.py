#Encoder followed by a decoder 
#symetry
import numpy as np
import imageio as im
import os, glob
import tensorflow as tf

def image_loading(path_X, path_Y, flatten_X, flatten_Y):
    X_data = []
    os.chdir(path_X) 
    if flatten_X :
        for file in glob.glob("*.png"):
            X_data+=[im.imread(file).flatten()]  #.flatten()] #The data is stored as the flattened version of the images (no longer flattened with CNN)
    else :
        for file in glob.glob("*.png"):
            X_data+=[im.imread(file)]

    Y_data = []
    os.chdir(path_Y)
    if flatten_Y :
        for file in glob.glob("*.png"):
            Y_data+=[im.imread(file).flatten()]  #.flatten()] #The data is stored as the flattened version of the images (no longer flattened with CNN)
    else :
        for file in glob.glob("*.png"):
            Y_data+=[im.imread(file)]
    return X_data, Y_data


#https://www.youtube.com/watch?v=NhdzGfB1q74
#Decoder
#repeated 3*3 convolutional + relu layers, 2*2 max pooling layers to downsample (5:54)
#NB channel are doubled between each downsample iteration

#Encoder
#repeated 3*3 convolutional + relu layers 2*2 upsampling 2*2
#NB Halve the channels 

#Bottleneck
#downsample 2*2 max pooling, convolution 3*3 + RELU, upsampling again

#Connexions
#concatenate the symetrical connexions 

def unet(input_shape=(256, 256, 3), num_classes=1):
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Encoder
    conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')(conv4)
    drop4 = tf.keras.layers.Dropout(0.5)(conv4)
    pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(drop4)

    # Bottleneck
    conv5 = tf.keras.layers.Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = tf.keras.layers.Conv2D(1024, 3, activation='relu', padding='same')(conv5)
    drop5 = tf.keras.layers.Dropout(0.5)(conv5)

    # Decoder
    up6 = tf.keras.layers.Conv2DTranspose(512, 2, strides=(2, 2), padding='same')(drop5)
    up6 = tf.keras.layers.concatenate([up6, drop4], axis=3)
    conv6 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')(up6)
    conv6 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')(conv6)

    up7 = tf.keras.layers.Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(conv6)
    up7 = tf.keras.layers.concatenate([up7, conv3], axis=3)
    conv7 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(up7)
    conv7 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(conv7)

    up8 = tf.keras.layers.Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv7)
    up8 = tf.keras.layers.concatenate([up8, conv2], axis=3)
    conv8 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(up8)
    conv8 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(conv8)

    up9 = tf.keras.layers.Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv8)
    up9 = tf.keras.layers.concatenate([up9, conv1], axis=3)
    conv9 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(up9)
    conv9 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(conv9)

    outputs = tf.keras.layers.Conv2D(num_classes, 1, activation='sigmoid')(conv9)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model