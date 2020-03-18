
# Ref  https://www.tensorflow.org/model_optimization/guide/pruning/pruning_with_keras

## Setup 

#%load_ext tensorboard
import tensorboard

import tensorflow as tf
#tf.enable_eager_execution()

#from keras.applications.resnet50 import ResNet50

import tempfile
import zipfile
import os
import time
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

from keras.models import Model
#from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
#from keras.applications.resnet50 import preprocess_input, decode_predictions
#from keras.applications.vgg16 import VGG16
#from keras.applications.inception_resnet_v2 import InceptionResNetV2

from tensorflow.python.framework.ops import disable_eager_execution
#from inception_blocks_v2 import *

#from utils import get_available_gpus, get_available_cpus, ensure_folder, triplet_loss, get_smallest_loss, get_best_model

from keras.backend.tensorflow_backend import set_session
from keras import optimizers

embedding_size = 32
img_size = 96
channel = 3

#config.gpu_options.allow_growth = True

gpus = tf.config.experimental.list_physical_devices('GPU')

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.gpu_options.per_process_gpu_memory_fraction = 0.7
#config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
set_session(sess)


if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


def build_model():
    base_model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=(img_size, img_size, channel),pooling='avg')
    image_input = base_model.input
    x = base_model.layers[-1].output
    out = Dense(embedding_size)(x)
    image_embedder = Model(image_input, out)

    input_a = Input((img_size, img_size, channel), name='anchor')
    input_p = Input((img_size, img_size, channel), name='positive')
    input_n = Input((img_size, img_size, channel), name='negative')

    normalize = Lambda(lambda x: K.l2_normalize(x, axis=-1), name='normalize')

    x = image_embedder(input_a)
    output_a = normalize(x)
    x = image_embedder(input_p)
    output_p = normalize(x)
    x = image_embedder(input_n)
    output_n = normalize(x)

    merged_vector = concatenate([output_a, output_p, output_n], axis=-1)

    model = Model(inputs=[input_a, input_p, input_n],
                  outputs=merged_vector)
    return model


def triplet_loss(y_true, y_pred, alpha = 0.3):
    """
    Implementation of the triplet loss as defined by formula (3)
    
    Arguments:
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor images, of shape (None, 128)
            positive -- the encodings for the positive images, of shape (None, 128)
            negative -- the encodings for the negative images, of shape (None, 128)
    
    Returns:
    loss -- real number, value of the loss
    """
    
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    
    # Step 1: Compute the (encoding) distance between the anchor and the positive, you will need to sum over axis=-1
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    # Step 2: Compute the (encoding) distance between the anchor and the negative, you will need to sum over axis=-1
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    # Step 3: subtract the two previous distances and add alpha.
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
    
    return loss

disable_eager_execution()

start = time.time()
 

## Prepare the training data

batch_size = 8
num_classes = 7
#num_classes = 5750
epochs = 1
num_of_train_samples = 13171
num_of_train_samples = 255
#num_of_test_samples = 255

### Train a MNIST model without pruning

## Build the MNIST model

l = tf.keras.layers

input_shape = (224,224,3)
#input_shape = (3,96,96)

# AlexNet

model = tf.keras.Sequential()

# 1st Convolutional Layer
model.add(tf.keras.layers.Conv2D(filters=96, input_shape=input_shape, kernel_size=(11,11),\
 strides=(4,4), padding='valid'))
model.add(tf.keras.layers.Activation('relu'))
# Pooling 
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# Batch Normalisation before passing it to the next layer
model.add(tf.keras.layers.BatchNormalization())

# 2nd Convolutional Layer
model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
model.add(tf.keras.layers.Activation('relu'))
# Pooling
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# Batch Normalisation
model.add(tf.keras.layers.BatchNormalization())

# 3rd Convolutional Layer
model.add(tf.keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(tf.keras.layers.Activation('relu'))
# Batch Normalisation
model.add(tf.keras.layers.BatchNormalization())

# 4th Convolutional Layer
model.add(tf.keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(tf.keras.layers.Activation('relu'))
# Batch Normalisation
model.add(tf.keras.layers.BatchNormalization())

# 5th Convolutional Layer
model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(tf.keras.layers.Activation('relu'))
# Pooling
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# Batch Normalisation
model.add(tf.keras.layers.BatchNormalization())

# Passing it to a dense layer
model.add(tf.keras.layers.Flatten())
# 1st Dense Layer
model.add(tf.keras.layers.Dense(4096, input_shape=(224*224*3,)))
model.add(tf.keras.layers.Activation('relu'))
# Add Dropout to prevent overfitting
model.add(tf.keras.layers.Dropout(0.4))
# Batch Normalisation
model.add(tf.keras.layers.BatchNormalization())

# 2nd Dense Layer
model.add(tf.keras.layers.Dense(4096))
model.add(tf.keras.layers.Activation('relu'))
# Add Dropout
model.add(tf.keras.layers.Dropout(0.4))
# Batch Normalisation
model.add(tf.keras.layers.BatchNormalization())

# 3rd Dense Layer
model.add(tf.keras.layers.Dense(1000))
model.add(tf.keras.layers.Activation('relu'))
# Add Dropout
model.add(tf.keras.layers.Dropout(0.4))
# Batch Normalisation
model.add(tf.keras.layers.BatchNormalization())

# Output Layer
model.add(tf.keras.layers.Dense(num_classes))
model.add(tf.keras.layers.Activation('softmax'))

#  https://forums.fast.ai/t/vgg16-fine-tuning-low-accuracy/3123/19

#base_model = VGG16(include_top=False, weights=None,
#                       input_tensor=None, input_shape=input_shape)

#base_model = ResNet50(include_top=False, weights=None,
#                       input_tensor=None, input_shape=input_shape)

# Add final layers
#x = base_model.output
#x = Flatten()(x)
#predictions = Dense(num_classes, activation='softmax', name='fc1000')(x)

#x = Flatten(name='flatten')(x)
#x = Dense(1024, activation='relu')(x)
#x = Dropout(0.4)(x)
#x = Dense(1024, activation='relu')(x)
#x = Dropout(0.4)(x)

#predictions = Dense(num_classes, activation='softmax')(x)


# This is the model we will train
#model = Model(input=base_model.input, output=predictions)

#model = build_model()

#model = faceRecoModel(input_shape)

#model = tf.keras.Sequential([
#    l.Conv2D(
#        32, 5, padding='valid', activation='relu', input_shape=input_shape),
#    l.MaxPooling2D((2, 2), (2, 2), padding='same'),
#    l.BatchNormalization(),
#    l.Conv2D(64, 5, padding='same', activation='relu'),
#    l.MaxPooling2D((2, 2), (2, 2), padding='same'),
#    l.Flatten(),
#    l.Dense(1024, activation='relu'),
#    l.Dropout(0.4),
#   l.Dense(num_classes, activation='softmax')
#])


model.summary()

## Train the model to reach an accuracy >99%

logdir = tempfile.mkdtemp()
print('Writing training logs to ' + logdir)

#%tensorboard --logdir={logdir}


callbacks = [tf.keras.callbacks.TensorBoard(log_dir=logdir, profile_batch=0)]

#rmsprop = optimizers.RMSprop

model.compile(
    loss=tf.keras.losses.categorical_crossentropy,
    #loss=triplet_loss,
    #optimizer=adam,
    optimizer='rmsprop',
    #optimizer='sgd',
    metrics=['accuracy'])


train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(
        #'/home/scott/lfw_align',
        '/home/scott/HotGirl_dataset_align',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical')

#x_train, y_train = train_datagen.flow_from_directory(
#        '/home/scott/HotGirl_dataset_align',
#        target_size=(110, 110),
#        batch_size=32,
#        class_mode='binary')


test_generator = test_datagen.flow_from_directory(
        #'/home/scott/lfw_align',
        '/home/scott/HotGirl_dataset_align',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical')

model.fit_generator(
        train_generator,
        steps_per_epoch=num_of_train_samples/batch_size,
        epochs=epochs,validation_data=train_generator,validation_steps=num_of_train_samples/batch_size)

#model.fit_generator(
#        train_generator,
#        steps_per_epoch=num_of_train_samples/batch_size,
#        epochs=epochs,validation_data=train_generator,validation_steps=num_of_train_samples/batch_size,
#        use_multiprocessing=True,workers=get_available_cpus())

#model.fit(x_train, y_train,
#          batch_size=batch_size,
#          epochs=epochs,
#          verbose=1,
#          callbacks=callbacks,
#          validation_data=(x_test, y_test))

#model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
#                    steps_per_epoch=len(x_train) / batch_size, epochs=epochs)

#score = model.evaluate(x_test, y_test, verbose=0)
#score = model.evaluate(test_generator, verbose=0)

#print('Test loss:', score[0])
#print('Test accuracy:', score[1])

#Y_pred = model.predict_generator(test_generator, num_of_train_samples / batch_size)
#y_pred = np.argmax(Y_pred, axis=1)

#print('Confusion Matrix')
#print(test_generator.classes)
#print(y_pred)

#print(confusion_matrix(test_generator.classes, y_pred))


#print('Classification Report')

target_names = ['IliCheng', 'KashiwagiYuki', 'MarinaNagasawa', 'MatsuiRena', 'NanaAsakawa', 'SandyWu', 'Shinozaki']
#print(classification_report(test_generator.classes, y_pred))


## Save the original model for size comparison later

# Backend agnostic way to save/restore models
_, keras_file = tempfile.mkstemp('.h5')
print('Saving model to: ', keras_file)
tf.keras.models.save_model(model, keras_file, include_optimizer=False)
#tf.keras.models.save_model(model, "/tmp/model", save_format="tf")

# VGG16 /tmp/tmpcvy4ze1j.h5  Time to excute is : 1795.7865588665009    accuracy  0.04
# ResNet50 /tmp/tmpcndeg04n.h5  loss: 12.6315 - accuracy: 0.0113 - val_loss: 9.2542 - val_accuracy: 0.0243

end = time.time()
print("\nTime to excute is :",end - start)

# loss: 8.3237 - accuracy: 0.0367 - val_loss: 7.7447 - val_accuracy: 0.0402
# Saving model to:  /tmp/tmp61ml91wf.h5



