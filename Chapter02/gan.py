# Complete Generator Adversarial Network which I typed

import os
import time
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import scipy.io as io
import numpy as np
import scipy.ndimage as nd
import matplotlib.pyplot as plt
from os import system, name  
from tensorflow.keras.backend import shape
from tensorflow.keras.layers import Input, LeakyReLU
from tensorflow.keras.layers import Conv3DTranspose, Conv3D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.python.keras.backend import dropout

DIR_PATH = './data/3DShapeNets/volumetric_data/'

# Fixed problems with Error #15: Initializing libiomp5md.dll, but found libiomp5 already initialized.
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def clear():
    # for windows
    if os.name == 'nt':
        _ = os.system('cls')
  
    # for mac and linux(here, os.name is 'posix')
    else:
        _ = os.system('clear')

def build_generator():
    '''
    Create a Generator Model with hyperparameters values defined as follows
    :return: Generator network
    '''
    z_size = 200
    gen_filters = [512, 256, 128, 64, 1]
    gen_kernel_sizes = [4, 4, 4, 4, 4]
    gen_strides = [1, 2, 2, 2, 2]
    gen_input_shape = (1, 1, 1, z_size)
    gen_activations = ['relu', 'relu', 'relu', 'relu', 'sigmoid']
    gen_convolutional_blocks = 5
    gen_dropout_rate = 0.2

    # Create the input layer
    input_layer = Input(shape=gen_input_shape, name='Generator Input Layer')

    # First 3D transpose convolution otherwise known in Keras as Deconvolution
    layer = Conv3DTranspose(
                filters=gen_filters[0],
                kernel_size=gen_kernel_sizes[0],
                strides=gen_strides[0], dropout=gen_dropout_rate)(input_layer) 
    layer = BatchNormalization()(layer, training=True)
    layer = Activation(activation=gen_activations[0])(layer)

    # Add 4 3D transpose convolution blocks
    for i in range(gen_convolutional_blocks - 1):
        layer = Conv3DTranspose(
                    filters=gen_filters[i+1],
                    kernel_size=gen_kernel_sizes[i+1],
                    strides=gen_strides[i+1],
                    padding='same',
                    dropout=gen_dropout_rate)(layer)
        layer = BatchNormalization()(layer, training=True)
        layer = Activation(activation=gen_activations[i+1])(layer)

    # Create a Keras Model
    gen_model = Model(inputs=[input_layer], outputs=layer)
    gen_model.summary()
    return gen_model

def build_discriminator():
    '''
    Create a Discriminator Model using hyperparameters values defined as follows
    :return: Discriminator network
    '''
    dis_input_shape = (64, 64, 64, 1)
    dis_filters = [64, 128, 256, 512, 1]
    dis_kernel_sizes = [4, 4, 4, 4, 4]
    dis_strides = [2, 2, 2, 2, 1]
    dis_paddings = ['same', 'same', 'same', 'same', 'valid']
    dis_alphas = [0.2, 0.2, 0.2, 0.2, 0.2]
    dis_activations = ['leaky_relu', 'leaky_relu', 'leaky_relu', 'leaky_relu', 'sigmoid']
    dis_convolutional_blocks = 5
    dis_dropout_rate = 0.2
    
    # Create the input layer
    dis_input_layer = Input(shape=dis_input_shape, name='Discriminator Input Layer')

    # The first 3D convolution block
    layer = Conv3D(filters=dis_filters[0],
                   kernel_size=dis_kernel_sizes[0],
                   strides=dis_strides[0],
                   padding=dis_paddings[0],
                   dropout = dis_dropout_rate)(dis_input_layer)
    layer = BatchNormalization()(layer, training=True)
    layer = LeakyReLU(dis_alphas[0])(layer)

    # Add 4 more convolutional blocks
    for i in range(dis_convolutional_blocks - 1):
        layer = Conv3D(filters=dis_filters[i+1],
                       kernel_size=dis_kernel_sizes[i+1],
                       strides=dis_strides[i+1],
                       padding=dis_paddings[i+1],
                       dropout = dis_dropout_rate)(layer)
        layer = BatchNormalization()(layer, training=True)
        if dis_activations[i+1] == 'leaky_relu':
            layer = LeakyReLU(dis_alphas[i+1])(layer)
        elif dis_activations[i + 1] == 'sigmoid':
            layer = Activation(activation='sigmoid')(layer)

    dis_model = Model(inputs=[dis_input_layer], outputs=layer)
    dis_model.summary()
    return dis_model

def getVoxelsFromMat(path, cube_len=64):
    voxels = io.loadmat(path)['instance']
    voxels = np.pad(voxels, (1, 1), 'constant', constant_values=(0, 0))
    if cube_len != 32 and cube_len == 64:
        voxels = nd.zoom(voxels, (2, 2, 2), mode='constant', order=0)
    return voxels

def get3ImagesForACategory(obj='airplane', train=True, cube_len=64, obj_ratio=1.0):
    obj_path = DIR_PATH + obj + '/30/'
    obj_path += 'train/' if train else 'test/'
    fileList = [f for f in os.listdir(obj_path) if f.endswith('.mat')]
    fileList = fileList[0:int(obj_ratio * len(fileList))]
    volumeBatch = np.asarray([getVoxelsFromMat(obj_path + f, cube_len) for f in fileList], dtype=bool)
    return volumeBatch

def saveFromVoxels(voxels, path):
    z, x, y = voxels.nonzero()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, -z, zdir='z', c='red')
    plt.savefig(path)
    plt.close()

def write_log(callback, name, value, batch_no):
    writer = tf.summary.create_file_writer(callback.log_dir)
    with writer.as_default():
        # other model code would go here
        tf.summary.scalar(name, value, description=batch_no)
        writer.flush()

def main():
    clear() # Clears the terminal 
    
    # Hyperparameters
    gen_learning_rate = 0.0025
    dis_learning_rate = 0.00001
    gen_beta = 0.5
    dis_beta = 0.9
    adversarialModel_beta = 0.5
    batch_size = 32
    z_size = 200
    generated_volumes_dir = 'generated_volumes'
    log_dir = 'logs'
    epochs = 10
    MODE = 'train'
    
    # Create two lists to store losses
    gen_losses = []
    dis_losses = []

    # Build the Generator and Discriminator
    generator = build_generator()
    discriminator = build_discriminator()

    # Specify Optimizer
    gen_optimizer = Adam(learning_rate=gen_learning_rate, beta_1=gen_beta)
    dis_optimimzer = Adam(learning_rate=dis_learning_rate, beta_1=dis_beta)

    # Compile networks
    generator.compile(loss='binary_crossentropy', optimizer=gen_optimizer)
    discriminator.compile(loss='binary_crossentropy', optimizer=dis_optimimzer)

    # Create and compile the adversarial model
    discriminator.trainable = False
    adversarial_model = Sequential(name='AdversarialModel')
    adversarial_model.add(generator)
    adversarial_model.add(discriminator)
    adversarial_model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=gen_learning_rate, beta_1=adversarialModel_beta))
    adversarial_model.summary()

    # Getting images
    print('\nGetting Images')
    volumes = get3ImagesForACategory(obj='airplane', train=True, obj_ratio=1.0)
    volumes = volumes[..., np.newaxis].astype(float)
    print(f'Obtained {len(volumes)} images')

    # Creating the Tensorflow callback class
    print('\nCreating TensorBoard callback')
    tensorboard = TensorBoard(log_dir='{}/{}'.format(log_dir, time.time()))
    tensorboard.set_model(generator)
    tensorboard.set_model(discriminator)

    # Run the simulation for a specified number of epochs
    print('\n\n ############################# Starting Simulation #############################\n\n')
    for epoch in range(epochs):
        print(f'Epoch: {epoch}')

        number_of_batches = int(volumes.shape[0] / batch_size)
        print(f'Number of batches: {number_of_batches}')
        
        for index in range(number_of_batches):
            print(f'Batch: {index + 1}')

            z_sample = np.random.normal(0, 0.33, size=[batch_size, 1, 1, 1, z_size]).astype(np.float32)
            volumes_batch = volumes[index * batch_size:(index + 1) * batch_size, :, :, :]

            # Generate Fake images'
            gen_volumes = generator.predict(z_sample,verbose=1)

            # Make the discriminator network trainable
            discriminator.trainable = True
                    
            # Create fake and real labels
            labels_real = np.reshape([1] * batch_size, (-1, 1, 1, 1, 1))
            labels_fake = np.reshape([0] * batch_size, (-1, 1, 1, 1, 1))
                    
            # Train the discriminator network
            loss_real = discriminator.train_on_batch(volumes_batch, labels_real)
            loss_fake = discriminator.train_on_batch(gen_volumes, labels_fake)
                    
            # Calculate total discriminator loss
            d_loss = 0.5 * (loss_real + loss_fake)
            z = np.random.normal(0, 0.33, size=[batch_size, 1, 1, 1, z_size]).astype(np.float32)

            # Train the adversarial model
            g_loss = adversarial_model.train_on_batch(z, np.reshape([1] * batch_size, (-1, 1, 1, 1, 1)))
        
            # Append the losses
            gen_losses.append(g_loss)
            dis_losses.append(d_loss)

            # Generate and save the 3D images after each epoch
            if index % 10 == 0:
                z_sample2 = np.random.normal(0, 0.33, size=[batch_size, 1, 1, 1, z_size]).astype(np.float32)
                generated_volumes = generator.predict(z_sample2, verbose=1)
            
            for i, generated_volume in enumerate(generated_volumes[:5]):
                voxels = np.squeeze(generated_volume)
                voxels[voxels < 0.5] = 0.
                voxels[voxels >= 0.5] = 1.
                saveFromVoxels(voxels, f'results/img_{epoch}_{index}_{i}')

        # Save losses to Tensorboard
        write_log(tensorboard, 'g_loss', np.mean(gen_losses), epoch)
        write_log(tensorboard, 'd_loss', np.mean(dis_losses), epoch)

    # Save the models
    generator.save_weights(os.path.join(generated_volumes_dir, 'generator_weights.h5'))
    discriminator.save_weights(os.path.join(generated_volumes_dir, 'discriminator_weights.h5'))

    if MODE == 'predict':
        # Create models
        generator = build_generator()
        discriminator = build_discriminator()

        # Load model weights
        generator.load_weights(os.path.join('models', 'generator_weights.h5'), True)
        discriminator.load_weights(os.path.join('models', 'discriminator_weights.h5'), True)

        # Generate 3D models
        z_sample = np.random.normal(0, 1, size=[batch_size, 1, 1, 1, z_size]).astype(np.float32)
        generated_volumes = generator.predict(z_sample, verbose=1)

        for i, generated_volume in enumerate(generated_volumes[:2]):
            voxels = np.squeeze(generated_volume)
            voxels[voxels < 0.5] = 0.
            voxels[voxels >= 0.5] = 1.
            saveFromVoxels(voxels, 'results/gen_{}'.format(i))

if __name__ == '__main__':
    main()