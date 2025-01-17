import os
import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import Input, Model
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.callbacks import TensorBoard
from keras.layers import Conv2D, Flatten, Dense, BatchNormalization, Reshape, concatenate, LeakyReLU, Lambda
from tensorflow.keras import backend as K
from keras.layers import Activation, UpSampling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from keras_preprocessing import image
from scipy.io import loadmat
from tqdm import tqdm
import pickle

# Fixed problems with Error #15: Initializing libiomp5md.dll, but found libiomp5 already initialized.
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Pickle Object to store variables
PICKLE_IMAGES_FILE_NAME = 'pickle_images.pkl'
PICKLE_TRAINING_GAN_FILE_NAME = 'pickle_training_gan.pkl'
PICKLE_TRAINING_ENCODER_FILE_NAME = 'pickle_training_encoder.pkl'
PICKLE_TRAINING_GAN_WITH_FR_FILE_NAME = 'pickle_training_gan_with_fr.pkl'


def clear():
    # for windows
    if os.name == 'nt':
        _ = os.system('cls')
  
    # for mac and linux(here, os.name is 'posix')
    else:
        _ = os.system('clear')

def build_encoder():
    """
    Encoder Network
    """
    input_layer = Input(shape=(64, 64, 3))

    # 1st Convolutional Block
    enc = Conv2D(filters=32, kernel_size=5, strides=2, padding='same')(input_layer)
    # enc = BatchNormalization()(enc)
    enc = LeakyReLU(alpha=0.2)(enc)

    # 2nd Convolutional Block
    enc = Conv2D(filters=64, kernel_size=5, strides=2, padding='same')(enc)
    enc = BatchNormalization()(enc)
    enc = LeakyReLU(alpha=0.2)(enc)

    # 3rd Convolutional Block
    enc = Conv2D(filters=128, kernel_size=5, strides=2, padding='same')(enc)
    enc = BatchNormalization()(enc)
    enc = LeakyReLU(alpha=0.2)(enc)

    # 4th Convolutional Block
    enc = Conv2D(filters=256, kernel_size=5, strides=2, padding='same')(enc)
    enc = BatchNormalization()(enc)
    enc = LeakyReLU(alpha=0.2)(enc)

    # Flatten layer
    enc = Flatten()(enc)

    # 1st Fully Connected Layer
    enc = Dense(4096)(enc)
    enc = BatchNormalization()(enc)
    enc = LeakyReLU(alpha=0.2)(enc)

    # Second Fully Connected Layer
    enc = Dense(100)(enc)

    # Create a model
    model = Model(inputs=[input_layer], outputs=[enc])
    return model


def build_generator():
    """
    Create a Generator Model with hyperparameters values defined as follows
    """
    latent_dims = 100
    num_classes = 6

    input_z_noise = Input(shape=(latent_dims,), name='Generator Z Noise')
    input_label = Input(shape=(num_classes,), name='Generator Input Label')

    x = concatenate([input_z_noise, input_label])

    x = Dense(2048, input_dim=latent_dims + num_classes)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.2)(x)

    x = Dense(256 * 8 * 8)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.2)(x)

    x = Reshape((8, 8, 256))(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(filters=128, kernel_size=5, padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(filters=64, kernel_size=5, padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(filters=3, kernel_size=5, padding='same')(x)
    x = Activation('tanh')(x)

    model = Model(inputs=[input_z_noise, input_label], outputs=[x])
    return model


def expand_label_input(x):
    x = K.expand_dims(x, axis=1)
    x = K.expand_dims(x, axis=1)
    x = K.tile(x, [1, 32, 32, 1])
    return x


def build_discriminator():
    """
    Create a Discriminator Model with hyperparameters values defined as follows
    """
    input_shape = (64, 64, 3)
    label_shape = (6,)
    image_input = Input(shape=input_shape, name='Discriminator Input Image')
    label_input = Input(shape=label_shape, name='Discriminator Label Input')

    x = Conv2D(64, kernel_size=3, strides=2, padding='same')(image_input)
    x = LeakyReLU(alpha=0.2)(x)

    label_input1 = Lambda(expand_label_input)(label_input)
    x = concatenate([x, label_input1], axis=3)

    x = Conv2D(128, kernel_size=3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(256, kernel_size=3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(512, kernel_size=3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Flatten()(x)
    x = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[image_input, label_input], outputs=[x])
    return model


def build_fr_combined_network(encoder, generator, fr_model):
    input_image = Input(shape=(64, 64, 3))
    input_label = Input(shape=(6,))

    latent0 = encoder(input_image)

    gen_images = generator([latent0, input_label])

    fr_model.trainable = False

    resized_images = Lambda(lambda x: K.resize_images(gen_images, height_factor=2, width_factor=2,
                                                      data_format='channels_last'))(gen_images)
    embeddings = fr_model(resized_images)

    model = Model(inputs=[input_image, input_label], outputs=[embeddings])
    return model


def build_fr_model(input_shape):
    resent_model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=input_shape, pooling='avg')
    image_input = resent_model.input
    x = resent_model.layers[-1].output
    out = Dense(128)(x)
    embedder_model = Model(inputs=[image_input], outputs=[out])

    input_layer = Input(shape=input_shape)

    x = embedder_model(input_layer)
    output = Lambda(lambda x: K.l2_normalize(x, axis=-1))(x)

    model = Model(inputs=[input_layer], outputs=[output])
    return model


def build_image_resizer():
    input_layer = Input(shape=(64, 64, 3))

    resized_images = Lambda(lambda x: K.resize_images(x, height_factor=3, width_factor=3,
                                                      data_format='channels_last'))(input_layer)

    model = Model(inputs=[input_layer], outputs=[resized_images])
    return model


def calculate_age(taken, dob):
    birth = datetime.fromordinal(max(int(dob) - 366, 1))

    if birth.month < 7:
        return taken - birth.year
    else:
        return taken - birth.year - 1


def load_data(wiki_dir, dataset='wiki'):
    # Load the wiki.mat file
    meta = loadmat(os.path.join(wiki_dir, f"{dataset}.mat"))

    # Load the list of all files
    full_path = meta[dataset][0, 0]["full_path"][0]

    # List of Matlab serial date number
    dob = meta[dataset][0, 0]["dob"][0]

    # List of years when photo was taken
    photo_taken = meta[dataset][0, 0]["photo_taken"][0]  # year

    # Calculate age for all dobs
    age = [calculate_age(photo_taken[i], dob[i]) for i in range(len(dob))]

    # Create a list of tuples containing a pair of an image path and age
    images = []
    age_list = []
    for index, image_path in enumerate(full_path):
        images.append(image_path[0])
        age_list.append(age[index])

    # Return a list of all images and respective age
    middle_index = len(images) // 8 # Truncating the size for faster processing
    return images[:middle_index], age_list[:middle_index]


def age_to_category(age_list):
    age_list1 = []

    for age in age_list:
        if 0 < age <= 18:
            age_category = 0
        elif 18 < age <= 29:
            age_category = 1
        elif 29 < age <= 39:
            age_category = 2
        elif 39 < age <= 49:
            age_category = 3
        elif 49 < age <= 59:
            age_category = 4
        elif age >= 60:
            age_category = 5

        age_list1.append(age_category)

    return age_list1


def load_images(data_dir, image_paths, image_shape):
    images = None
    number_of_images = len(image_paths)
    pbar = tqdm(total=number_of_images) # Init pbar
    pickle_file_exists = False

    print(f'Loading {number_of_images} images')

    try:
        with open(PICKLE_IMAGES_FILE_NAME, 'rb') as pickle_in:
            images = pickle.load(pickle_in)

        pickle_file_exists = True
        print(f'{PICKLE_IMAGES_FILE_NAME} is loaded successfully')
    except FileNotFoundError:
        print(f'{PICKLE_IMAGES_FILE_NAME} does not exist yet. Creating it.')

    except EOFError:
        print(f'{PICKLE_IMAGES_FILE_NAME} is corrupted. Deleting it.')
        os.remove(PICKLE_IMAGES_FILE_NAME)

    except: # Handles all other exceptions
        pass

    if not pickle_file_exists:
        for i, image_path in enumerate(image_paths):
            try:
                # Load image
                loaded_image = image.load_img(os.path.join(data_dir, image_path), target_size=image_shape)

                # Convert PIL image to numpy ndarray
                loaded_image = image.img_to_array(loaded_image)

                # Add another dimension (Add batch dimension)
                loaded_image = np.expand_dims(loaded_image, axis=0)

                # Concatenate all images into one tensor
                if images is None:
                    images = loaded_image
                else:
                    images = np.concatenate([images, loaded_image], axis=0)
                
                pbar.update(n=1) # Increments counter
            except Exception as e:
                print(f"Error at {i} with Exception {e}")
        
        with open(PICKLE_IMAGES_FILE_NAME, 'wb') as pickle_out:
            pickle.dump(images, pickle_out)

    print('Finished loading all images')
    return images


def euclidean_distance_loss(y_true, y_pred):
    """
    Euclidean distance loss
    https://en.wikipedia.org/wiki/Euclidean_distance
    :param y_true: TensorFlow/Theano tensor
    :param y_pred: TensorFlow/Theano tensor of the same shape as y_true
    :return: float
    """
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))


def write_log(callback, name, value, batch_no):
    writer = tf.summary.create_file_writer(callback.log_dir)
    with writer.as_default():
        tf.summary.scalar(name, value, step=batch_no)

def load_data_when_training_gan():
    epochs_start = 0
    gen_losses = []
    dis_losses = []

    try:
        with open(PICKLE_TRAINING_GAN_FILE_NAME, 'rb') as pickle_in:
            epochs_start = pickle.load(pickle_in)
            gen_losses = pickle.load(pickle_in)
            dis_losses = pickle.load(pickle_in)

        print(f'{PICKLE_TRAINING_GAN_FILE_NAME} is loaded successfully')
    except FileNotFoundError:
        print(f'{PICKLE_TRAINING_GAN_FILE_NAME} does not exist yet. Creating it.')

    except EOFError:
        print(f'{PICKLE_TRAINING_GAN_FILE_NAME} is corrupted. Deleting it.')
        os.remove(PICKLE_TRAINING_GAN_FILE_NAME)

    except: # Handles all other exceptions
        pass

    return epochs_start, gen_losses, dis_losses

def load_data_when_training_encoder():
    epochs_start = 0
    encoder_losses = []
    try:
        with open(PICKLE_TRAINING_ENCODER_FILE_NAME, 'rb') as pickle_in:
            epochs_start = pickle.load(pickle_in)
            encoder_losses = pickle.load(pickle_in)

        print(f'{PICKLE_TRAINING_ENCODER_FILE_NAME} is loaded successfully')
    except FileNotFoundError:
        print(f'{PICKLE_TRAINING_ENCODER_FILE_NAME} does not exist yet. Creating it.')

    except EOFError:
        print(f'{PICKLE_TRAINING_ENCODER_FILE_NAME} is corrupted. Deleting it.')
        os.remove(PICKLE_TRAINING_ENCODER_FILE_NAME)

    except: # Handles all other exceptions
        pass

    return epochs_start, encoder_losses

def load_data_when_training_gan_with_fr():
    epochs_start = 0
    reconstruction_losses = []
    try:
        with open(PICKLE_TRAINING_GAN_WITH_FR_FILE_NAME, 'rb') as pickle_in:
            epochs_start = pickle.load(pickle_in)
            reconstruction_losses = pickle.load(pickle_in)

        print(f'{PICKLE_TRAINING_GAN_WITH_FR_FILE_NAME} is loaded successfully')
    except FileNotFoundError:
        print(f'{PICKLE_TRAINING_GAN_WITH_FR_FILE_NAME} does not exist yet. Creating it.')

    except EOFError:
        print(f'{PICKLE_TRAINING_GAN_WITH_FR_FILE_NAME} is corrupted. Deleting it.')
        os.remove(PICKLE_TRAINING_GAN_WITH_FR_FILE_NAME)

    except: # Handles all other exceptions
        pass

    return epochs_start, reconstruction_losses

def save_data_when_training_gan(epochs, gen_losses, dis_losses):
    with open(PICKLE_TRAINING_GAN_FILE_NAME, 'wb') as pickle_out:
        pickle.dump(epochs, pickle_out)
        pickle.dump(gen_losses, pickle_out)
        pickle.dump(dis_losses, pickle_out)

def save_data_when_training_encoder(epoch, encoder_losses):
    with open(PICKLE_TRAINING_ENCODER_FILE_NAME, 'wb') as pickle_out:
        pickle.dump(epoch, pickle_out)
        pickle.dump(encoder_losses, pickle_out)

def save_data_when_training_gan_with_fr(epoch, encoder_losses):
    with open(PICKLE_TRAINING_GAN_WITH_FR_FILE_NAME, 'wb') as pickle_out:
        pickle.dump(epoch, pickle_out)
        pickle.dump(encoder_losses, pickle_out)


def save_rgb_img(img, path):
    """
    Save an rgb image
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(img)
    ax.axis("off")
    ax.set_title("Image")

    plt.savefig(path)
    plt.close()

def resized_images_layer(gen_images):
    return K.resize_images(gen_images, height_factor=3, width_factor=3, data_format='channels_last')


if __name__ == '__main__':
    # Define hyperparameters
    data_dir = "data"
    wiki_dir = os.path.join(data_dir, "wiki_crop")
    epochs = 1
    batch_size = 2
    image_shape = (64, 64, 3)
    z_shape = 100
    TRAIN_GAN = True
    TRAIN_ENCODER = True
    TRAIN_GAN_WITH_FR = True
    fr_image_shape = (192, 192, 3)

    # Clear screen
    clear()

    # Define optimizers
    dis_optimizer = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.999, epsilon=10e-8)
    gen_optimizer = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.999, epsilon=10e-8)
    adversarial_optimizer = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.999, epsilon=10e-8)

    """
    Build and compile networks
    """
    # Build and compile the discriminator network
    discriminator = build_discriminator()
    discriminator.compile(loss=['binary_crossentropy'], optimizer=dis_optimizer)

    # Build and compile the generator network
    generator = build_generator()
    generator.compile(loss=['binary_crossentropy'], optimizer=gen_optimizer)

    # Build and compile the adversarial model
    discriminator.trainable = False
    input_z_noise = Input(shape=(100,))
    input_label = Input(shape=(6,))
    recons_images = generator([input_z_noise, input_label])
    valid = discriminator([recons_images, input_label])
    adversarial_model = Model(inputs=[input_z_noise, input_label], outputs=[valid])
    adversarial_model.compile(loss=['binary_crossentropy'], optimizer=gen_optimizer)

    tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()))
    tensorboard.set_model(generator)
    tensorboard.set_model(discriminator)

    """
    Load the dataset
    """
    images, age_list = load_data(wiki_dir=wiki_dir, dataset="wiki")
    age_cat = age_to_category(age_list)
    final_age_cat = np.reshape(np.array(age_cat), [len(age_cat), 1])
    classes = len(set(age_cat))
    y = to_categorical(final_age_cat, num_classes=len(set(age_cat)))
    loaded_images = load_images(wiki_dir, images, (image_shape[0], image_shape[1]))

    # Implement label smoothing
    real_labels = np.ones((batch_size, 1), dtype=np.float32) * 0.9
    fake_labels = np.zeros((batch_size, 1), dtype=np.float32) * 0.1

    """
    Train the generator and the discriminator network
    """
    if TRAIN_GAN:
        print(f'\n\n #################### TRAINING GAN ####################\n\n')
        
        # Call function to load data from previous training run
        epoch_start = 0
        gen_losses = []
        dis_losses = []
        epoch_start, gen_losses, dis_losses = load_data_when_training_gan()

        for epoch in range(epoch_start, epochs):
            print(f"\nEpoch: {epoch + 1} out of {len(range(epochs))}")

            if epoch > epoch_start:
                gen_losses = []
                dis_losses = []

            number_of_batches = range(int(len(loaded_images) / batch_size))
            pbar = tqdm(total=len(number_of_batches)) # Init pbar

            for index in number_of_batches:
                images_batch = loaded_images[index * batch_size:(index + 1) * batch_size]
                images_batch = images_batch / 127.5 - 1.0
                images_batch = images_batch.astype(np.float32)

                y_batch = y[index * batch_size:(index + 1) * batch_size]
                z_noise = np.random.normal(0, 1, size=(batch_size, z_shape))

                """
                Train the discriminator network
                """

                # Generate fake images
                initial_recon_images = generator.predict_on_batch([z_noise, y_batch])

                d_loss_real = discriminator.train_on_batch([images_batch, y_batch], real_labels)
                d_loss_fake = discriminator.train_on_batch([initial_recon_images, y_batch], fake_labels)

                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                """
                Train the generator network
                """

                z_noise2 = np.random.normal(0, 1, size=(batch_size, z_shape))
                random_labels = np.random.randint(0, 6, batch_size).reshape(-1, 1)
                random_labels = to_categorical(random_labels, 6)

                g_loss = adversarial_model.train_on_batch([z_noise2, random_labels], np.asarray([1] * batch_size))

                gen_losses.append(g_loss)
                dis_losses.append(d_loss)
                
                pbar.update(n=1) # Increments counter

            # Write losses to Tensorboard
            write_log(tensorboard, 'g_loss', np.mean(gen_losses), epoch)
            write_log(tensorboard, 'd_loss', np.mean(dis_losses), epoch)

            # Save to pickle to future use
            save_data_when_training_gan(epoch, gen_losses, dis_losses)

            """
            Generate images after every 10th epoch
            """
            if epoch % 10 == 0:
                images_batch = loaded_images[0:batch_size]
                images_batch = images_batch / 127.5 - 1.0
                images_batch = images_batch.astype(np.float32)

                y_batch = y[0:batch_size]
                z_noise = np.random.normal(0, 1, size=(batch_size, z_shape))

                gen_images = generator.predict_on_batch([z_noise, y_batch])

                for i, img in enumerate(gen_images[:5]):
                    save_rgb_img(img, path="results/img_{}_{}.png".format(epoch, i))

                # Save networks every 10th epoch
                try:
                    generator.save_weights("generator.h5")
                    discriminator.save_weights("discriminator.h5")
                except Exception as e:
                    print(f"Error: {e}")

        # Save networks after all simulations are complete
        try:
            generator.save_weights("generator.h5")
            discriminator.save_weights("discriminator.h5")
        except Exception as e:
            print(f"Error: {e}")

    """
    Train encoder
    """

    if TRAIN_ENCODER:
        # Build and compile encoder
        encoder = build_encoder()
        encoder.compile(loss=euclidean_distance_loss, optimizer='adam')

        # Load the generator network's weights
        try:
            generator.load_weights("generator.h5")
        except Exception as e:
            print(f"Error: {e}")

        z_i = np.random.normal(0, 1, size=(5000, z_shape))

        y = np.random.randint(low=0, high=6, size=(5000,), dtype=np.int64)
        num_classes = len(set(y))
        y = np.reshape(np.array(y), [len(y), 1])
        y = to_categorical(y, num_classes=num_classes)

        # Call function to load data from previous training run
        epoch_start = 0
        encoder_losses = []
        epoch_start, encoder_losses = load_data_when_training_encoder()

        print(f'\n\n #################### TRAINING ENCODER ####################\n\n')
        for epoch in range(epoch_start, epochs):
            print(f"\nEpoch: {epoch + 1} out of {len(range(epochs))}")

            if epoch > epoch_start:
                encoder_losses = []

            number_of_batches = range(int(z_i.shape[0] / batch_size))
            pbar = tqdm(total=len(number_of_batches)) # Init pbar

            for index in number_of_batches:
                z_batch = z_i[index * batch_size:(index + 1) * batch_size]
                y_batch = y[index * batch_size:(index + 1) * batch_size]

                generated_images = generator.predict_on_batch([z_batch, y_batch])

                # Train the encoder model
                encoder_loss = encoder.train_on_batch(generated_images, z_batch)

                encoder_losses.append(encoder_loss)
                pbar.update(n=1) # Increments counter

            # Write the encoder loss to Tensorboard
            write_log(tensorboard, "encoder_loss", np.mean(encoder_losses), epoch)

            # Save to pickle to future use
            save_data_when_training_encoder(epoch, encoder_losses)


            # Save encoder weights every 10th epoch
            if epoch % 10 == 0:
                # Save the encoder model
                encoder.save_weights("encoder.h5")

        # Save the encoder model
        encoder.save_weights("encoder.h5")

    """
    Optimize the encoder and the generator network
    """
    if TRAIN_GAN_WITH_FR:

        # Load the encoder network
        encoder = build_encoder()
        encoder.load_weights("encoder.h5")

        # Load the generator network
        generator.load_weights("generator.h5")

        image_resizer = build_image_resizer()
        image_resizer.compile(loss=['binary_crossentropy'], optimizer='adam')

        # Face recognition model
        fr_model = build_fr_model(input_shape=fr_image_shape)
        fr_model.compile(loss=['binary_crossentropy'], optimizer="adam")

        # Make the face recognition network as non-trainable
        fr_model.trainable = False

        # Input layers
        input_image = Input(shape=(64, 64, 3))
        input_label = Input(shape=(6,))

        # Use the encoder and the generator network
        latent0 = encoder(input_image)
        gen_images = generator([latent0, input_label])

        # Resize images to the desired shape
        resized_images = Lambda(resized_images_layer, name='Lambda_Layer')(gen_images)
        embeddings = fr_model(resized_images)

        # Create a Keras model and specify the inputs and outputs for the network
        fr_adversarial_model = Model(inputs=[input_image, input_label], outputs=[embeddings])

        # Compile the model
        fr_adversarial_model.compile(loss=euclidean_distance_loss, optimizer=adversarial_optimizer)

        # Call function to load data from previous training run
        epoch_start = 0
        epoch_start, reconstruction_losses = load_data_when_training_gan_with_fr()

        print(f'\n\n #################### TRAINING GAN WITH FACE RECOGNITION ####################\n\n')
        for epoch in range(epoch_start, epochs):
            print(f"\nEpoch: {epoch + 1} out of {len(range(epochs))}")
            reconstruction_losses = []

            number_of_batches = range(int(len(loaded_images) / batch_size))
            pbar = tqdm(total=len(number_of_batches)) # Init pbar

            for index in number_of_batches:
                images_batch = loaded_images[index * batch_size:(index + 1) * batch_size]
                images_batch = images_batch / 127.5 - 1.0
                images_batch = images_batch.astype(np.float32)

                y_batch = y[index * batch_size:(index + 1) * batch_size]
                images_batch_resized = image_resizer.predict_on_batch(images_batch)
                real_embeddings = fr_model.predict_on_batch(images_batch_resized)
                reconstruction_loss = fr_adversarial_model.train_on_batch([images_batch, y_batch], real_embeddings)
                reconstruction_losses.append(reconstruction_loss)
                pbar.update(n=1) # Increments counter

            # Write the reconstruction loss to Tensorboard
            write_log(tensorboard, "reconstruction_loss", np.mean(reconstruction_losses), epoch)

            save_data_when_training_gan_with_fr(epoch, index, reconstruction_losses)

            """
            Generate images
            """
            if epoch % 10 == 0:
                images_batch = loaded_images[0:batch_size]
                images_batch = images_batch / 127.5 - 1.0
                images_batch = images_batch.astype(np.float32)

                y_batch = y[0:batch_size]
                z_noise = np.random.normal(0, 1, size=(batch_size, z_shape))

                gen_images = generator.predict_on_batch([z_noise, y_batch])

                for i, img in enumerate(gen_images[:5]):
                    save_rgb_img(img, path="results/img_opt_{}_{}.png".format(epoch, i))

                # Save improved weights for both of the networks every 10th epoch
                generator.save_weights("generator_optimized.h5")
                encoder.save_weights("encoder_optimized.h5")

        # Save improved weights for both of the networks
        generator.save_weights("generator_optimized.h5")
        encoder.save_weights("encoder_optimized.h5")
