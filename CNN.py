


############################################################################################################
import os

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, model_from_json
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import matplotlib.pyplot as plt


# dimensions of our images.
img_width, img_height = 100, 100

train_data_dir = '/Users/amruthaa/IdeaProjects/CNN/tcss555/training_image'
validation_data_dir = '/Users/amruthaa/IdeaProjects/CNN/tcss555/test'
nb_train_samples = 2000
nb_validation_samples = 800
epochs = 50
batch_size = 32

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

fit = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

save_dir = "/Users/amruthaa/IdeaProjects/CNN"
model_name = 'CNN_model.h5'

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir,model_name)
model.save(model_path)
print('saved trained model at %s' %model_path)

accuracy = fit.history['acc']
val_accuracy = fit.history['val_acc']

loss = fit.history['loss']
val_loss = fit.history['val_loss']

# plt.plot(fit.history['acc'])
# plt.plot(fit.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend()
# plt.savefig('/Users/amruthaa/IdeaProjects/CNN/Accuracyplot.png', format='png')
#plt.show()
# summarize history for loss
plt.plot(fit.history['loss'], label= 'Loss')
plt.plot(fit.history['val_loss'], label = 'Validation Loss')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.savefig('/Users/amruthaa/IdeaProjects/CNN/Lossplot.png', format='png')
#plt.show()

#############################################################################################

# from keras.models import load_model
# model = load_model("/Users/amruthaa/IdeaProjects/CNN/CNN_model.h5")
#
#
#
# # Part 3 - Making new predictions
# import numpy as np
# import pandas as pd
# from keras.preprocessing import image
# import csv
#
# test_data_dir = "/Users/amruthaa/IdeaProjects/CNN/tcss555/public-test-data/image"
# profile = "/Users/amruthaa/IdeaProjects/CNN/tcss555/public-test-data/profile/profile.csv"
#
#
# with open(profile, 'a') as newFile:
#     newFileCSV = csv.reader(newFile, delimiter=',')
#     for row in newFileCSV:
#         print(row)
#         # test_image = image.load_img(test_data_dir + "/" + row[1] + ".jpg", target_size = (100, 100))
#         # test_image = image.img_to_array(test_image)
#         # test_image = np.expand_dims(test_image, axis = 0)
#         # result = model.predict_classes(test_image)
#         # if result[0][0] == 1:
#         #     prediction = 'male'
#         # else:
#         #     prediction = 'female'
#         #
#         # row[3]=prediction
#
# #########backup#######
# # with open(profile, 'a') as newFile:
# #     newFileWriter = csv.writer(newFile)
# #     for row in newFileWriter:
# #         test_image = image.load_img(test_data_dir + "/" + row[1] + ".jpg", target_size = (100, 100))
# #         test_image = image.img_to_array(test_image)
# #         test_image = np.expand_dims(test_image, axis = 0)
# #         result = model.predict_classes(test_image)
# #         if result[0][0] == 1:
# #             prediction = 'male'
# #         else:
# #             prediction = 'female'
# #
# #         row[3]=prediction