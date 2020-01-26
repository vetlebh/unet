import os
import random
import pickle

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from tensorflow.keras.layers import Lambda, RepeatVector, Reshape
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D, GlobalMaxPool2D
from tensorflow.keras.layers import concatenate, add
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split

from utils import iou

from google.colab import files

if 'COLAB_TPU_ADDR' not in os.environ:
  print('Not connected to TPU')
else:
  print("Connected to TPU")

from google.colab import drive
drive.mount('/mntDrive')



# Hyperparameters

im_height = 256
im_width = 256

checkpoint_path = 'weights_5_2.h5'
model_path = 'model_5_2.h5'

epochs = 60
batch_size = 128

training_path = '/mntDrive/My Drive/Prosjektoppgave/training.nosync'
img_name = 'image.npy'
gt_name = 'gt.npy'

save_history_path = '/content/trainHistoryDict_5_2'
save_metrics_path = '/content/metricsDict_5_2'

test_size = 0.1
val_size = 0.1
random_state = 1

max_slices = 15



# Data Generator

walk = next(os.walk(training_path))[1]

X = np.zeros((len(walk)*max_slices, im_height, im_width, 1))
y = np.zeros((len(walk)*max_slices, im_height, im_width, 1))

img_nr = 0
sum_slices = 0
patients_not_found = 0
for ids in walk:

    try:
        img = np.load(os.path.join(training_path, ids, img_name))
        gt = np.load(os.path.join(training_path, ids, gt_name))
        slices = img.shape[2]

        for slice_nr in range(slices):

            img_slice, gt_slice = img[:, :, slice_nr], gt[:, :, slice_nr]
            img_resized = resize(img_slice, (im_height, im_width, 1), mode = 'edge', preserve_range = True, anti_aliasing=True)
            gt_resized = resize(gt_slice, (im_height, im_width, 1), mode = 'edge', preserve_range = True, anti_aliasing=True)

            # We are only interested in the classes 'heart' and 'background' for this experiment
            gt_resized = (gt_resized > 0.5).astype(np.uint8)

            X[sum_slices, :, :, :] = img_resized/255.0
            y[sum_slices, :, :, :] = gt_resized

            sum_slices +=1

    except:
        print(f'{ids} not found')
        patients_not_found += 1
        continue

    if(img_nr%10 == 0):
        print(f'{img_nr} images and {sum_slices} slices loaded to array')
    img_nr += 1
print(f'Image load complete. {img_nr} images and {sum_slices} slices loaded successfully. ')

X, y = X[:sum_slices, :, :, :], y[:sum_slices, :, :, :]
print(X.shape, y.shape)
print(np.unique(y))

# Split train data and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

# Split train data into train and valid
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=val_size, random_state=random_state)

print(f'Training size: {X_train.shape[0]}, Validation size: {X_valid.shape[0]}, Test size: {X_test.shape[0]}')




# Model

def conv2d_block(input_tensor, n_filters, kernel_size = 3, batchnorm = True):
    """Function to add 2 convolutional layers with the parameters passed to it"""
    # first layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # second layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x


def get_unet(input_img, n_filters = 16, dropout = 0.1, batchnorm = True):
    # Contracting Path
    c1 = conv2d_block(input_img, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)

    c2 = conv2d_block(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)

    c5 = conv2d_block(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm)

    # Expansive Path
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)

    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)

    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)

    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8)
    u9 = concatenate([u9, c1])
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model

model = load_model('/mntDrive/My Drive/Prosjektoppgave/model_5_2.h5')
model.load_weights('/mntDrive/My Drive/Prosjektoppgave/weights_5_2.h5')

input_img = Input((im_height, im_width, 1), name='img')
model = get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)
model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])



# Evaluate the model pre training
loss, acc = model.evaluate(X_test,  y_test, verbose=2)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc))



# Train

callbacks = [
    EarlyStopping(patience=10, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1),
    ModelCheckpoint(checkpoint_path, verbose=1, save_best_only=True, save_weights_only=True)
]

results = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=callbacks, validation_data=(X_valid, y_valid))




model.load_weights(checkpoint_path)
files.download(checkpoint_path)
model.save(model_path)
files.download(model_path)

with open(save_history_path, 'wb') as file_pi:
        pickle.dump(results.history, file_pi)

files.download(save_history_path)

area = np.sum(y_test, axis=(1,2,3))
nonempty = area > 0

model.compile(
            optimizer=model.optimizer,
            loss=model.loss,
            metrics=model.metrics + [
                "binary_accuracy",
                "FalseNegatives",
                "FalsePositives",
                "Precision",
                "Recall",
                iou
            ],)

evaluation = model.evaluate(X_test[nonempty], y_test[nonempty])
metrics = {
    name: value
    for name, value
    in zip(model.metrics_names, evaluation)
                }

metrics

with open(save_metrics_path, 'wb') as file_pi:
        pickle.dump(metrics, file_pi, protocol=2)

files.download(save_metrics_path)

#model.summary()

"""Evaluate"""

model.load_weights(checkpoint_path)
# Evaluate the model post training
loss, acc = model.evaluate(X_test,  y_test, verbose=2)
print("Trained model, accuracy: {:5.2f}%".format(100*acc))

model.save(model_path)

with open(save_history_path, 'wb') as file_pi:
        pickle.dump(results.history, file_pi)
