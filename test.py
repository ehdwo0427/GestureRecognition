import os
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from source.module_image_util.image_util import *
from source.common_util import *  # *을 이용해서 가져오게되면 그 파일내에 전역변수와 함수를 다 가져오게 된다.
from source.module.C3D_v2_model import *

train_RGB = []
train_optical = []
train_depth = []


def model_load_data():
    # 35878개 트레이닝용
    for num in range(2):
        RGB_DATA, OPTICAL_FLOW_DATA = video_enhancement_RGB(gesture_folder_path + RGB_VIDEO_LIST[num][0])
        DEPTH_DATA = video_enhancement_depth(gesture_folder_path + RGB_VIDEO_LIST[num][1])
        train_RGB.append(RGB_DATA)
        train_optical.append(OPTICAL_FLOW_DATA)
        train_depth.append(DEPTH_DATA)
        print(num)

    return train_RGB, train_optical, train_depth


if __name__ == '__main__':
    gesture_folder_path = DATA_PATH + 'gesture_training_data/'
    TRAINING_TEXT_DATA = gesture_folder_path + 'train_list.txt'
    ALL_VIDEO_LIST = np.loadtxt(TRAINING_TEXT_DATA, delimiter=' ', dtype='str')
    RGB_VIDEO_LIST = ALL_VIDEO_LIST[:, 0:-1]
    LABEL_LIST = ALL_VIDEO_LIST[:, [-1]]
    LABEL_LIST = np.array(LABEL_LIST, dtype=np.int)
    Label_List = LABEL_LIST[:2]
    epochs = 120000
    num_classes = 249
    lr = 0.001

    # pickle로 training data load
    retinex = open("train_retinex_data.pkl", "rb")
    optical = open("train_optical_data.pkl", "rb")
    depth = open("train_depth_data.pkl", "rb")
    # train_RGB, train_optical, train_depth = model_load_data()
    # print(train_RGB)
    # print('Save file start')
    # pickle.dump(train_RGB, retinex)
    # print('retinex_done')
    # pickle.dump(train_optical, optical)
    # print('optical_done')
    # pickle.dump(train_depth, depth)
    temp = pickle.load(retinex)
    a = pickle.load(optical)
    b = pickle.load(depth)
    print('Finish')
    temp = np.array(temp).astype('float')
    retinex.close()
    optical.close()
    depth.close()


    # train_RGB = np.array(train_RGB).astype('float')
    # train_optical = np.array(train_optical).astype('float')
    # train_depth = np.array(train_depth).astype('float')

    # np.save('./save_train_RGB', train_RGB)
    # np.save('./save_train_optical', train_optical)
    # np.save('./save_train_depth', train_depth)

    # train_RGB = np.load('./save_train.npy')
    # train_optical = np.load('./save_train_optical')
    # train_depth = np.load('./save_train_depth')

    model = ResC3D()


    # model = tf.keras.models.Sequential([
    #     tf.keras.layers.Reshape((3, 32, 112, 112)),
    #     tf.keras.layers.Conv3D(64, (7, 7, 3), strides=(2, 1, 1), padding='same', activation='relu'),
    #     tf.keras.layers.Conv3D(64, (1, 1, 1), strides=(1, 1, 1), padding='same', activation='relu'),
    #     tf.keras.layers.Conv3D(64, (3, 3, 3), padding='same', activation='relu'),
    #     tf.keras.layers.Conv3D(64, (3, 3, 3), padding='same', activation='relu'),
    #
    #     tf.keras.layers.Conv3D(64, (3, 3, 3), padding='same', activation='relu'),
    #     tf.keras.layers.Conv3D(64, (3, 3, 3), padding='same', activation='relu'),
    #
    #     tf.keras.layers.Conv3D(128, (1, 1, 1), strides=(2, 2, 1), padding='same', activation='relu'),
    #     tf.keras.layers.Conv3D(128, (3, 3, 3), padding='same', activation='relu'),
    #     tf.keras.layers.Conv3D(128, (3, 3, 3), padding='same', activation='relu'),
    #
    #     tf.keras.layers.Conv3D(128, (3, 3, 3), padding='same', activation='relu'),
    #     tf.keras.layers.Conv3D(128, (3, 3, 3), padding='same', activation='relu'),
    #
    #     tf.keras.layers.Conv3D(256, (1, 1, 1), padding='same', activation='relu'),
    #     tf.keras.layers.Conv3D(256, (3, 3, 3), padding='same', activation='relu'),
    #     tf.keras.layers.Conv3D(256, (3, 3, 3), padding='same', activation='relu'),
    #
    #     tf.keras.layers.Conv3D(256, (3, 3, 3), padding='same', activation='relu'),
    #     tf.keras.layers.Conv3D(256, (3, 3, 3), padding='same', activation='relu'),
    #
    #     tf.keras.layers.Conv3D(512, (1, 1, 1), strides=(2, 2, 1), padding='same', activation='relu'),
    #     tf.keras.layers.Conv3D(512, (3, 3, 3), padding='same', activation='relu'),
    #     tf.keras.layers.Conv3D(512, (3, 3, 3), padding='same', activation='relu'),
    #
    #     tf.keras.layers.Conv3D(512, (3, 3, 3), padding='same', activation='relu'),
    #     tf.keras.layers.Conv3D(512, (3, 3, 3), padding='same', activation='relu'),
    #
    #     tf.keras.layers.Conv3D(1024, (1, 1, 1), padding='same', activation='relu'),
    #     tf.keras.layers.AveragePooling3D((7, 7, 1), padding='same'),
    #     tf.keras.layers.Flatten(),
    #     tf.keras.layers.Dense(249, activation='softmax')
    # ])

    sgd = SGD(learning_rate=0.001, momentum=0.9)
    model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(temp, Label_List, epochs=epochs)
    model.summary()

