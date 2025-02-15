'''Train a simple deep CNN on the CIFAR10 small images dataset.

It gets to 75% validation accuracy in 25 epochs, and 79% after 50 epochs.
(it's still underfitting at that point, though).
'''

from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
import os
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import keras.backend as K

def do_stuff():
    batch_size = 32
    num_classes = 2
    epochs = 50
    data_augmentation = True
    num_predictions = 20
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = 'keras_cifar10_trained_model.h5'

    # The data, split between train and test sets:
    (x_train1, y_train1), (x_test1, y_test1) = cifar10.load_data()
    print("SIZE X TRAIN",x_train1.shape,"SIZE Y TRAIN",y_train1.shape)
    y_train=y_train1[np.logical_or(y_train1==3, y_train1==5)]
    x_train=x_train1[np.logical_or(y_train1==3, y_train1==5).flatten()]
    y_test=y_test1[np.logical_or(y_test1==3, y_test1==5)]
    x_test=x_test1[np.logical_or(y_test1==3, y_test1==5).flatten()]
    for i in range(y_train.shape[0]):
        if y_train[i]==3:
            y_train[i] = 0
        else:
            y_train[i] = 1
    for i in range(y_test.shape[0]):
        if y_test[i]==3:
            y_test[i] = 0
        else:
            y_test[i] = 1
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    print("SIZE X TRAIN",x_train.shape,"SIZE Y TRAIN",y_train.shape)
    # Convert class vectors to binary class matrices.
    print("Y_TRAIN",y_train)
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    inputs=Input(shape=(32,32,3))

    conv1=keras.layers.Conv2D(32, (3,3), padding='same', activation='relu')(inputs)
    conv2=keras.layers.Conv2D(32,(3,3),activation='relu')(conv1)
    pool1=MaxPooling2D(pool_size=(2,2))(conv2)
    drop1=keras.layers.Dropout(0.25)(pool1, training=True)

    conv3=keras.layers.Conv2D(64, (3,3), padding='same', activation='relu')(drop1)
    conv4=keras.layers.Conv2D(64,(3,3),activation='relu')(conv3)
    pool2=MaxPooling2D(pool_size=(2,2))(conv4)
    drop2=keras.layers.Dropout(0.25)(pool2, training=True)

    flat=Flatten()(drop2)
    dense1=keras.layers.Dense(512,activation='relu')(flat)
    drop3=keras.layers.Dropout(0.25)(dense1, training=True)
    output=keras.layers.Dense(num_classes,activation='softmax')(drop3)

    model=Model(inputs=inputs,outputs=output)


    # initiate RMSprop optimizer
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test, y_test),
                  shuffle=True)
    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images

        # Compute quantities required for feature-wise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)

        # Fit the model on the batches generated by datagen.flow().
        model.fit_generator(datagen.flow(x_train, y_train,
                                         batch_size=batch_size),
                            epochs=epochs,
                            validation_data=(x_test, y_test),
                            workers=4)

    # Save model and weights
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)

    result=np.zeros((200,2000,num_classes))
    for i in tqdm(range(200)):
        result[i,:,:]=model.predict(x_test, verbose=0)
    predictionDOGS = result[:,0,1].mean(axis=0)
    uncertaintyDOGS = result[:,0,1].std(axis=0)
    predictionCATS = result[:,0,0].mean(axis=0)
    uncertaintyCATS = result[:,0,0].std(axis=0)
    un
    print('DOGS PREDICTION',predictionDOGS)
    print('DOGS UNCERTAINTY',uncertaintyDOGS)

    print('CATS PREDICTION',predictionCATS)
    print('CATS UNCERTAINTY',uncertaintyCATS)
    res2=result[:,0,1]
    res1=result[:,0,0]

    plt.close('all')
    plt.hist(res2,30, density=True, facecolor='g', alpha=0.75)
    plt.xlabel('Score')
    plt.title('Histogram of dogs')
    plt.grid(True)
    plt.show()

    plt.hist(res1,30, density=True, facecolor='g', alpha=0.75)
    plt.xlabel('Score')
    plt.title('Histogram of cats')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    gpu_device = "/gpu:1"
    if keras.backend.backend() == 'tensorflow':
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_device.rsplit(':', 1)[-1]
        session_config = K.tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        session_config.gpu_options.allow_growth = True
        session = K.tf.Session(config=session_config)
        with K.tf.device(gpu_device):
            do_stuff()