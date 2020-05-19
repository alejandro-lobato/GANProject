import numpy as np
import os

from tqdm import tqdm

import time
from keras import Input
from keras.layers import Dense, Reshape, LeakyReLU, Conv2D, Conv2DTranspose, Flatten, Dropout
from keras.models import Model
from keras.optimizers import RMSprop
from matplotlib import pyplot as plt
from PIL import Image

# ----------------------- Constants + Initialization ----------------------- 

# Number of Images to use from dataset
IMAGES_COUNT = 10000

# Dimensions of Images from CelebA
ORIG_WIDTH = 178
ORIG_HEIGHT = 208
diff = (ORIG_HEIGHT - ORIG_WIDTH) // 2

# Dimension of Output Images
WIDTH = 128
HEIGHT = 128

# Number of iterations for training
iters = 50000
# Number of images to use for each training step
batch_size = 10

# Number of images to calculate
CONTROL_SIZE_SQRT = 3

# Location of dataset
PIC_DIR = f'celeba-dataset/img_align_celeba/img_align_celeba/'

# Directory to save results of training
RES_DIR = f'res_{batch_size}'
FILE_PATH = '%s/generated_%d.png'

crop_rect = (0, diff, ORIG_WIDTH, ORIG_HEIGHT - diff)

# Load dataset in memory
images = []
for pic_file in tqdm(os.listdir(PIC_DIR)[:IMAGES_COUNT]):
    pic = Image.open(PIC_DIR + pic_file).crop(crop_rect)
    pic.thumbnail((WIDTH, HEIGHT), Image.ANTIALIAS)
    images.append(np.uint8(pic))

images = np.array(images) / 255
print(images.shape)

# Show some loaded images
plt.figure(1, figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(images[i])
    plt.axis('off')
# plt.show()

LATENT_DIM = 32
CHANNELS = 3

# ----------------------- Modeling of GAN ----------------------- 

# Create stacked convolutional network for Generator
def create_generator():
    gen_input = Input(shape=(LATENT_DIM, ))

    x = Dense(128 * 16 * 16)(gen_input)
    x = LeakyReLU()(x)
    x = Reshape((16, 16, 128))(x)

    x = Conv2D(256, 5, padding='same')(x)
    x = LeakyReLU()(x)

    x = Conv2DTranspose(256, 4, strides=2, padding='same')(x)
    x = LeakyReLU()(x)

    x = Conv2DTranspose(256, 4, strides=2, padding='same')(x)
    x = LeakyReLU()(x)

    x = Conv2DTranspose(256, 4, strides=2, padding='same')(x)
    x = LeakyReLU()(x)

    x = Conv2D(512, 5, padding='same')(x)
    x = LeakyReLU()(x)
    x = Conv2D(512, 5, padding='same')(x)
    x = LeakyReLU()(x)
    x = Conv2D(CHANNELS, 7, activation='tanh', padding='same')(x)

    generator = Model(gen_input, x)
    return generator

# Create stacked convolutional network for Discriminator
def create_discriminator():
    disc_input = Input(shape=(HEIGHT, WIDTH, CHANNELS))

    x = Conv2D(256, 3)(disc_input)
    x = LeakyReLU()(x)

    x = Conv2D(256, 4, strides=2)(x)
    x = LeakyReLU()(x)

    x = Conv2D(256, 4, strides=2)(x)
    x = LeakyReLU()(x)

    x = Conv2D(256, 4, strides=2)(x)
    x = LeakyReLU()(x)

    x = Conv2D(256, 4, strides=2)(x)
    x = LeakyReLU()(x)

    x = Flatten()(x)
    x = Dropout(0.4)(x)

    x = Dense(1, activation='sigmoid')(x)
    discriminator = Model(disc_input, x)

    optimizer = RMSprop(
        lr=.0001,
        clipvalue=1.0,
        decay=1e-8
    )

    discriminator.compile(
        optimizer=optimizer,
        loss='binary_crossentropy'
    )

    return discriminator


# Create both networks to use in GAN
generator = create_generator()
discriminator = create_discriminator()
discriminator.trainable = False

# Input is the latent space of the dataset (images)
gan_input = Input(shape=(LATENT_DIM, ))
# Output is the result of Discriminator => Generator
# (created images from Generator obtained from modifying weights based on the Discriminator output)
gan_output = discriminator(generator(gan_input))
# Model GAN
gan = Model(gan_input, gan_output)
# Optimizar for adapting learning
optimizer = RMSprop(lr=.0001, clipvalue=1.0, decay=1e-8)
gan.compile(optimizer=optimizer, loss='binary_crossentropy')

if not os.path.isdir(RES_DIR):
    os.mkdir(RES_DIR)

control_vectors = np.random.normal(size=(CONTROL_SIZE_SQRT**2, LATENT_DIM)) / 2

# ----------------------- Training process ----------------------- 

start = 0
d_losses = []
a_losses = []
images_saved = 0

start_iterations = 0

# If there are iterations saved, use them as the starting point
if os.path.isfile('gan.h5'): 
    print('Loading previous weights')
    gan.load_weights('gan.h5')
    print('Loading previous iterations count...')
    with open('iterations.txt') as file:
        start_iterations = int(file.readlines()[0])
        print('Read iterations: ' + str(start_iterations))


for step in range(start_iterations, iters):

    start_time = time.time()
    latent_vectors = np.random.normal(size=(batch_size, LATENT_DIM))
    # Create fake samples
    generated = generator.predict(latent_vectors)
    # Obtain some real samples
    real = images[start:start + batch_size]
    # Combine fake and real samples for Discriminator to classify
    combined_images = np.concatenate([generated, real])
    # Combine samples with their labels
    # 1 if comes from dataset
    # 0 if is fake (comes from Generator)
    labels = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])
    labels += .05 * np.random.random(labels.shape)

    # Train Discriminator (Supervised Learning)
    d_loss = discriminator.train_on_batch(combined_images, labels)
    d_losses.append(d_loss)

    # Create random normal noise for the Generator
    latent_vectors = np.random.normal(size=(batch_size, LATENT_DIM))
    misleading_targets = np.zeros((batch_size, 1))

    # Train Generator
    a_loss = gan.train_on_batch(latent_vectors, misleading_targets)
    a_losses.append(a_loss)

    start += batch_size
    if start > images.shape[0] - batch_size:
        start = 0

    # Save results every 50 steps
    if step % 50 == 0:
        print("Step " + str(step))
        print("="*50)
        print('%d/%d: d_loss: %.4f,  a_loss: %.4f.  (%.1f sec)' % (step + 1, iters, d_loss, a_loss, time.time() - start_time))
        print('Saving weights')
        gan.save_weights('gan.h5')
        with open('iterations.txt', 'w') as file:
            file.write(str(step))
        # Placeholder for Output image (0 filled)
        control_image = np.zeros((WIDTH * CONTROL_SIZE_SQRT, HEIGHT * CONTROL_SIZE_SQRT, CHANNELS))
        # Prediction from Generator
        control_generated = generator.predict(control_vectors)
        # Copy Generator => Placeholder
        for i in range(CONTROL_SIZE_SQRT ** 2):
            # Get x, y of current pizel (moving from array to matrix)
            x_off = i % CONTROL_SIZE_SQRT
            y_off = i // CONTROL_SIZE_SQRT
            # Copying generator's output to placeholder
            control_image[x_off * WIDTH:(x_off + 1) * WIDTH, y_off * HEIGHT:(y_off + 1) * HEIGHT, :] = control_generated[i, :, :, :]
        # Generate image from Generator 
        im = Image.fromarray(np.uint8(control_image * 255))
        # Save image
        im.save(FILE_PATH % (RES_DIR, step))


# Plot losses
plt.figure(1, figsize=(12, 8))
plt.subplot(121)
plt.plot(d_losses)
plt.xlabel('epochs')
plt.ylabel('discriminant losses')
plt.subplot(122)
plt.plot(a_losses)
plt.xlabel('epochs')
plt.ylabel('adversary losses')
plt.show()