import sys
import os
import logging
import time

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

LOG_PATH = './log/'  #log파일의 경로


def log_init(module_name='', module_path=''):
    '''
    :param module_name: 돌려본 모듈이 이름
    :param module_path: log파일 안에 따로 저장할 것이라면 경로를 추가로 넣어준다.
    :return:
    '''
    module_path = LOG_PATH + module_path
    curr_time = time.localtime() #log함수가 돌아가는 시간
    file_name = "%04d-%02d-%02d-%02d-%02d-%02d" % (curr_time.tm_year, curr_time.tm_mon, curr_time.tm_mday,
                                                   curr_time.tm_hour, curr_time.tm_min, curr_time.tm_sec)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  #Level에 따라 허용되는 구간이 있다. DEBUG가 가장 낮고 ERROR가 가장 높다. Level을 높게 잡으면 다른 상황을 표시 못한다.
    log_format = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s: %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_format)
    logger.addHandler(stream_handler)
    file_handler = logging.FileHandler(module_path + file_name + '_' + module_name + '.log') #파일의 저장경로와 이름
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger

