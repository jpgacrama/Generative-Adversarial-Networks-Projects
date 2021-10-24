# Complete Generator Adversarial Network which I typed

import os
from keras.engine import training
from keras.optimizer_v1 import Adam
import scipy.io as io
import numpy as np
import scipy.ndimage as nd
import matplotlib.pyplot as plt
from os import system, name  
from keras.backend import shape
from keras.layers import Input, LeakyReLU
from keras.layers.convolutional import Deconv3D, Conv3D
from keras.layers.core import Activation
from keras.layers.normalization.batch_normalization import BatchNormalization
from keras.models import Model
from tensorflow.keras import Sequential

def clear():
  
    # for windows
    if name == 'nt':
        _ = system('cls')
  
    # for mac and linux(here, os.name is 'posix')
    else:
        _ = system('clear')

def build_generator():
    """
    Create a Generator Model with hyperparameters values defined as follows
    :return: Generator network
    """
    z_size = 200
    gen_filters = [512, 256, 128, 64, 1]
    gen_kernel_sizes = [4, 4, 4, 4, 4]
    gen_strides = [1, 2, 2, 2, 2]
    gen_input_shape = (1, 1, 1, z_size)
    gen_activations = ['relu', 'relu', 'relu', 'relu', 'sigmoid']
    gen_convolutional_blocks = 5

    # Create the input layer
    input_layer = Input(shape=gen_input_shape)

    # First 3D transpose convolution otherwise known in Keras as Deconvolution
    layer = Deconv3D(filters=gen_filters[0],
                     kernel_size=gen_kernel_sizes[0],
                     strides=gen_strides[0])(input_layer) 
    layer = BatchNormalization()(layer, training=True)
    layer = Activation(activation=gen_activations[0])(layer)

    # Add 4 3D transpose convolution blocks
    for i in range(gen_convolutional_blocks - 1):
        layer = Deconv3D(
                    filters=gen_filters[i+1],
                    kernel_size=gen_kernel_sizes[i+1],
                    strides=gen_strides[i+1],
                    padding='same')(layer)
        layer = BatchNormalization()(layer, training=True)
        layer = Activation(activation=gen_activations[i+1])(layer)

    # Create a Keras Model
    gen_model = Model(inputs=input_layer, outputs=layer)
    gen_model.summary()
    return gen_model

def build_discriminator():
    """
    Create a Discriminator Model using hyperparameters values defined as follows
    :return: Discriminator network
    """
    dis_input_shape = (64, 64, 64, 1)
    dis_filters = [64, 128, 256, 512, 1]
    dis_kernel_sizes = [4, 4, 4, 4, 4]
    dis_strides = [2, 2, 2, 2, 1]
    dis_paddings = ['same', 'same', 'same', 'same', 'valid']
    dis_alphas = [0.2, 0.2, 0.2, 0.2, 0.2]
    dis_activations = ['leaky_relu', 'leaky_relu', 'leaky_relu', 'leaky_relu', 'sigmoid']
    dis_convolutional_blocks = 5
    
    # Create the input layer
    dis_input_layer = Input(shape=dis_input_shape)

    # The first 3D convolution block
    layer = Conv3D(filters=dis_filters[0],
                   kernel=dis_kernel_sizes[0],
                   strides=dis_strides[0],
                   padding=dis_paddings[0])(dis_input_layer)
    layer = BatchNormalization()(layer, training=True)
    layer = LeakyReLU(dis_alphas[0])(layer)

    # Add 4 more convolutional blocks
    for i in range(dis_convolutional_blocks - 1):
        layer = Conv3D(filters=dis_filters[i+1],
                       kernel_size=dis_kernel_sizes[i+1],
                       strides=dis_strides[i+1],
                       padding=dis_paddings[i+1])(layer)
        layer = BatchNormalization()(layer, training=True)
        if dis_activations[i+1] == 'leaky_relu':
            layer = LeakyReLU(dis_alphas[i+1])(layer)

    dis_model = Model(inputs=dis_input_layer, outputs=layer)
    print(dis_model.summary())
    return dis_model

def train_gan():
    gen_learning_rate = 0.0025
    dis_learning_rate = 0.00001
    gen_beta = 0.5
    dis_beta = 0.9
    batch_size = 32
    z_size = 200
    DIR_PATH = './data/3DShapeNets'
    generated_volumes_dir = 'generated_volumes'
    log_dir = 'logs'

    # Create Instances
    generator = build_generator()
    discriminator = build_discriminator()

    # Specify Optimizer
    gen_optimizer = Adam(lr=gen_learning_rate, beta_1=gen_beta)
    dis_optimimzer = Adam(lr=dis_learning_rate, beta_1=dis_beta)

    # Compile networks
    generator.compile(loss='binary_crossentropy', optimizer=gen_optimizer)
    discriminator.compile(loss='binary_crossentropy', optimizer=dis_optimimzer)

    # Create and compile the adversarial model
    discriminator.trainable = False
    adversarial_model = Sequential()
    adversarial_model.add(generator)
    adversarial_model.add(discriminator)
    adversarial_model.compile(loss="binary_crossentropy", optimizer=Adam(lr=gen_learning_rate, beta_1=beta))

def main():
    clear()
    print(f'Current Working Directory: {os.getcwd()}')
    voxels = io.loadmat('./3DShapeNets/volumetric_data/airplane/30/train/3e73b236f62d05337678474be485ca_12.mat')['instance']
    voxels = np.pad(voxels, (1,1), 'constant', constant_values = (0,0))
    voxels = nd.zoom(voxels, 2, mode='constant', order=0)
    print(f'Shape of Voxels: {np.shape(voxels)}')

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d', aspect='auto')
    ax.voxels(voxels, edgecolor="red")
    plt.show()

if __name__ == "__main__":
    main()
