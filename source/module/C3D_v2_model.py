import os
from abc import ABC

import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
# from source.module.ResC3D import *
from tensorflow.keras.optimizers import SGD
from source.module_image_util.image_util import *
from source.common_util import *  # *을 이용해서 가져오게되면 그 파일내에 전역변수와 함수를 다 가져오게 된다.
# from source.module.C3D_v2_model import *
import tensorflow.keras.layers as layers


class ResC3D(tf.keras.Model, ABC):
    def __init__(self):
        super(ResC3D, self).__init__()
        self.reshape = tf.keras.layers.Reshape((3, 32, 112, 112))
        self.conv1 = layers.Conv3D(64, (7, 7, 3), strides=(2, 1, 1), padding='same', activation='relu')
        self.conv2a = layers.Conv3D(64, (1, 1, 1), strides=(1, 1, 1), padding='same', activation='relu')
        self.conv2 = layers.Conv3D(64, (3, 3, 3), padding='same', activation='relu')
        self.residual = layers.Add()

        self.conv3a = layers.Conv3D(128, (1, 1, 1), strides=(2, 2, 1), padding='same', activation='relu')
        self.conv3 = layers.Conv3D(128, (3, 3, 3), padding='same', activation='relu')

        self.conv4a = layers.Conv3D(256, (1, 1, 1), strides=(1, 2, 2), padding='same', activation='relu')
        self.conv4 = layers.Conv3D(256, (3, 3, 3), padding='same', activation='relu')

        self.conv5a = layers.Conv3D(512, (1, 1, 1), strides=(1, 2, 2), padding='same', activation='relu')
        self.conv5 = layers.Conv3D(512, (3, 3, 3), padding='same', activation='relu')

        self.conv6 = layers.Conv3D(1024, (1, 1, 1), padding='same', activation='relu')
        self.pool = layers.AveragePooling3D((7, 7, 1), padding='same')
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(249, activation='softmax')

    def call(self, inputs):
        x = self.reshape(inputs)
        x = self.conv1(x)
        # Res2a Unit
        x = self.conv2a(x)
        # shortcut = x
        x = self.conv2(x)
        x = self.conv2(x)
        # shortcut = self.residual(x, shortcut)
        # Res2b Unit
        x = self.conv2(x)
        x = self.conv2(x)
        # shortcut = self.residual(x, shortcut)
        # Res3a Unit
        x = self.conv3a(x)
        # shortcut = x
        x = self.conv3(x)
        x = self.conv3(x)
        # shortcut = self.residual(x, shortcut)
        # Res3b Unit
        x = self.conv3(x)
        x = self.conv3(x)
        # shortcut = self.residual(x, shortcut)
        # Res4a Unit
        x = self.conv4a(x)
        # shortcut = x
        x = self.conv4(x)
        x = self.conv4(x)
        # shortcut = self.residual(x, shortcut)
        # Res4b Unit
        x = self.conv4(x)
        x = self.conv4(x)
        # shortcut = self.residual(x, shortcut)
        # Res5a Unit
        x = self.conv5a(x)
        # shortcut = x
        x = self.conv5(x)
        x = self.conv5(x)
        # shortcut = self.residual(x, shortcut)
        # Res5b Unit
        x = self.conv5(x)
        x = self.conv5(x)
        # shortcut = self.residual(x, shortcut)
        x = self.conv6(x)
        x = self.pool(x)
        x = self.flatten()
        return self.dense(x)
