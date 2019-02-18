"""在TPU运行时下测试有没有分配TPU计算资源"""
import os
import pprint
import tensorflow as tf
from keras.callbacks import *
from keras.layers import *
from keras.models import *
from keras.preprocessing import image
from keras.models import Model

datagen = image.ImageDataGenerator(featurewise_center=False,
                                   samplewise_center=False,
                                   featurewise_std_normalization=False,
                                   samplewise_std_normalization=False,
                                   zca_whitening=False,
                                   rotation_range=0.3,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.,
                                   zoom_range=0.2,
                                   channel_shift_range=0.,
                                   fill_mode='nearest',
                                   cval=0.0,
                                   horizontal_flip=False,
                                   vertical_flip=False,
                                   rescale=1. / 255,
                                   preprocessing_function=None,
                                   # data_format=K.image_data_format(),
                                   )

train_generator = datagen.flow_from_directory(
    # '/Users/imperatore/tmp/num_ocr',  # this is the target directory
    r'number_ok1',  # this is the target directory
    target_size=(48, 48),  # all images will be resized to 48*40
    batch_size=256,
    class_mode='categorical',
    color_mode='grayscale')

validation_generator = datagen.flow_from_directory(
    # '/Users/imperatore/tmp/nums_classed',
    r'number_ok1',
    target_size=(48, 48),
    batch_size=128,
    class_mode='categorical',
    color_mode='grayscale')

num_class = 10
drop = drop

# define structure
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

model = tf.keras.models.Sequential()

tf.keras.losses.categorical_crossentropy
model.add(tf.keras.layers.Input((48, 48, 1)))
# 以下为第一个卷积层
model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), padding='same',
                                 strides=(1, 1), name=None))
model.add(tf.keras.layers.Activation(activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(rate=drop))

model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same',
                                 strides=(1, 1), name=None))
model.add(tf.keras.layers.Activation(activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(rate=drop))

model.add(tf.keras.layers.Conv2D(512, kernel_size=(3, 3), padding='same',
                                 strides=(1, 1), name=None))
model.add(tf.keras.layers.Activation(activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(rate=drop))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(10, kernel_initializer='he_normal'))
model.add(tf.keras.layers.Activation(activation='relu'))
model.add(tf.keras.layers.Dropout(rate=drop))

drop = drop
input_tensor = Input((48, 48, 1))
x = input_tensor

# x = resnet(x)

x = Flatten()(x)
x = Dense(1000, kernel_initializer='he_normal')(x)
# x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(drop)(x)

x = Dense(10, kernel_initializer='he_normal')(x)

# model = Model(inputs=input_tensor, outputs=x)
model = tf.keras.models.Model(inputs=input_tensor, outputs=x)
print(model.layers)
# sys.exit(0)

# model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

# # convert model
# model_dir = './'
# estimator_model = tf.keras.estimator.model_to_estimator(keras_model=model,
#                                                         model_dir=model_dir)

import os

tpu_model = tf.contrib.tpu.keras_to_tpu_model(
    model,
    strategy=tf.contrib.tpu.TPUDistributionStrategy(
        tf.contrib.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
    )
)
tpu_model.compile(
    optimizer=tf.train.AdamOptimizer(learning_rate=1e-3, ),
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    metrics=['sparse_categorical_accuracy']
)

# model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
# # model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
# print('\n'.join([str(tmp) for tmp in model.layers]))
# print('model length: %s' % len(model.layers))

early_stopping = EarlyStopping(monitor='val_loss', patience=30)
tpu_model.fit_generator(
    train_generator,
    steps_per_epoch=256,
    epochs=20,
    validation_data=validation_generator,
    nb_val_samples=256,
    verbose=True,
    callbacks=[early_stopping])

tpu_model.save('cnn3_gen_tpu_1.4.h5')  # always save your weights after training or during training
