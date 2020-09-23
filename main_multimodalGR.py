from source.common_util import log_init
from source.module_image_util.image_util import *

#python main문 시작
if __name__ == '__main__':
    logger = log_init(module_name='multimodalGR_main')

    data_list = get_gesture_data(logger)

    flag = False
    for data_index, data in enumerate(data_list):
        if data_index == 5:
            print("test")
        if data_index == 2:
            flag = True
        if flag:
            break

    print('hello')