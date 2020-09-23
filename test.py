import os
import pandas as pd
import numpy as np
from source.module_image_util.image_util import *
from source.common_util import *        # *을 이용해서 가져오게되면 그 파일내에 전역변수와 함수를 다 가져오게 된다.

if __name__ == '__main__':
    gesture_folder_path = DATA_PATH + 'gesture_training_data/'
    TRAINING_TEXT_DATA = gesture_folder_path + 'train_list.txt'
    ALL_VIDEO_LIST = np.loadtxt(TRAINING_TEXT_DATA, delimiter=' ', dtype='str')
    RGB_VIDEO_LIST = ALL_VIDEO_LIST[:, 0:-1]
    LABEL_LIST = ALL_VIDEO_LIST[:, [-1]]

    b = video_enhancement_RGB(gesture_folder_path + RGB_VIDEO_LIST[3][0])
    # g = video_dense_optical_flow(gesture_folder_path + RGB_VIDEO_LIST[100][0])
    # c = read_depth_video_file(gesture_folder_path + RGB_VIDEO_LIST[3][1])




'''
for g_num in range(NUM_GESTURE):
    gesture_data_path = gesture_folder_path + str(g_num + 1).zfill(3) + '/'
    file_list = os.listdir(gesture_data_path)
    data = []
    for file_num in range(int(len(file_list)/2)):
        tmp_data = []
        for tag in tag_list:
            tmp_data.append(tag+'_'+str(file_num+1).zfill(5)+'.avi')
        data.append(tmp_data)

        # read_image_file()

print(data)
data = []
label =[]

for g_num in range(NUM_GESTURE):
    gesture_data_path = gesture_folder_path + str(g_num + 1).zfill(3) + '/'
    file_list = os.listdir(gesture_data_path)

    for file_name in file_list:
        tmp_data = []
        if 'K' in file_name:
            tmp_data.append(file_name)
            tmp_data.append('M'+file_name[1:])
            tmp_label = file_name[2:-4]
        data.append(tmp_data)
        label.append(int(tmp_label))

print(data)
print(label)
'''
'''
#frame을 32로 고정시켜서 하라고 하면 넣어주는 것
video = [i for i in range(30*3)]
output = []
target_num = 32
for index in range(target_num):
    output.append(video[int(index*len(video)/32)])
print(len(output))
print(output)
'''