import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv3D, AveragePooling3D, ZeroPadding3D
from tensorflow.keras.optimizers import SGD

HEIGHT = 128
WIDTH = 171
CROP_SIZE = 112
learning_late = 0.001
drop_out = 0.9
weight_decay = 0.0005
momentum = 0.9
drop_epoch = 5000
training_epoch = 120000
Total_filter = 32


model = Sequential()
model.add(Conv3D(64, kernel_size=(7, 7, 3), strides=(2, 1, 1), input_shape=(3, 32, 112, 112), activation='relu'))
model.add(Conv3D(64, kernel_size=(1, 1, 1), activation='relu'))
model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu'))
model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu'))
# residual block
model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu'))
model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu'))
# residual block
model.add(Conv3D(128, kernel_size=(1, 1, 1), strides=(2, 2), activation='relu'))
model.add(Conv3D(128, kernel_size=(3, 3, 3), activation='relu'))
model.add(Conv3D(128, kernel_size=(3, 3, 3), activation='relu'))
# residual block
model.add(Conv3D(128, kernel_size=(3, 3, 3), activation='relu'))
model.add(Conv3D(128, kernel_size=(3, 3, 3), activation='relu'))
# residual block
model.add(Conv3D(256, kernel_size=(1, 1, 1), strides=(2, 2), activation='relu'))
model.add(Conv3D(256, kernel_size=(3, 3, 3), activation='relu'))
model.add(Conv3D(256, kernel_size=(3, 3, 3), activation='relu'))
# residual block
model.add(Conv3D(256, kernel_size=(3, 3, 3), activation='relu'))
model.add(Conv3D(256, kernel_size=(3, 3, 3), activation='relu'))
# 4까지 완료

model.add(Conv3D(512, kernel_size=(1, 1, 1), strides=(2, 2), activation='relu'))
model.add(Conv3D(512, kernel_size=(3, 3, 3), activation='relu'))
model.add(Conv3D(512, kernel_size=(3, 3, 3), activation='relu'))
# residual block

model.add(Conv3D(512, kernel_size=(3, 3, 3), activation='relu'))
model.add(Conv3D(512, kernel_size=(3, 3, 3), activation='relu'))
# residual block

model.add(Conv3D(1024, kernel_size=(1, 1, 1), activation='relu'))
model.add(AveragePooling3D(7, 7, 1))
model.add(Flatten())
model.add(Dense(249, activation='softmax'))