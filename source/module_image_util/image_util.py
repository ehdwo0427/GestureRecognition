import os
import sys
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import tensorflow as tf
from source.common_util import *

# 가장 상위 폴더로 인식하게 해주는 코드
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))

DATA_PATH = './data/'  # data의 초반 경로
NUM_GESTURE = 32
SPLIT_SECTION = 16
Sigma_List = np.uint8([15, 80, 250])


def video_enhancement_depth(file_path):
    """
    depth video median filter 씌우는 함
    :param file_path: 파일 명이 들어가야함.
    :return: 비디오파일이 리턴됨
    """
    cap = cv2.VideoCapture(file_path)

    image_input_data = []
    # print(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    gesture_section = cap.get(cv2.CAP_PROP_FRAME_COUNT) / NUM_GESTURE
    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break

        img = cv2.resize(img, (128, 171))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        median = cv2.medianBlur(gray, 3)

        if int(cap.get(cv2.CAP_PROP_POS_FRAMES) % gesture_section) == 0:
            # print(cap.get(cv2.CAP_PROP_POS_FRAMES))
            median = tf.image.random_crop(median, [112, 112])
            image_input_data.append(median)

        # cv2.imshow('video', gray)
        # cv2.imshow('median', median)
        cv2.waitKey(10)
    while len(image_input_data) != NUM_GESTURE:
        if len(image_input_data) < NUM_GESTURE:
            image_input_data.insert(int(len(image_input_data) / 2), image_input_data[int(len(image_input_data) / 2)])
        elif len(image_input_data) > NUM_GESTURE:
            image_input_data.pop()

    cap.release()

    return image_input_data


def video_enhancement_RGB(file_path):
    '''
    RGB 데이터를 받아서 multi scale retinex video 데이터와 optical flow video 데이터를 튜플형태로 반환시켜준다.
    :param file_path: RGB input video
    :return: multi_scale_retinex_input_data, optical_flow_input_data
    '''
    cap = cv2.VideoCapture(file_path)

    delay = 10  # waitKey 시간지연
    image_input_data = []  # retinex input data
    optical_flow_list = []  # optical flow all list
    optical_flow_input_data = []  # optical flow input data
    frame_average = []
    sum_average_of_frame = 0
    get_section_frame = []
    gesture_section = cap.get(cv2.CAP_PROP_FRAME_COUNT) / NUM_GESTURE
    optical_flow_section = cap.get(cv2.CAP_PROP_FRAME_COUNT) / SPLIT_SECTION
    section_sum = 0

    ret, frame1 = cap.read()
    frame1 = cv2.resize(frame1, (128, 171))
    prev = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255

    while cap.isOpened():
        success, frame2 = cap.read()
        if not success:
            break
        frame2 = cv2.resize(frame2, (128, 171))
        next_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prev, next_frame, None, 0.5, 3, 13, 10, 5, 1.1, 0)  # dense optical flow 계산

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])  # 크기, 방향 성분으로 나눈다.
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)  # 색으로 표현한 HSV영상을 BGR로 변경
        section_sum += mag.mean()
        bgr = tf.image.random_crop(bgr, [112, 112, 3])
        optical_flow_list.append(bgr)

        if int(cap.get(cv2.CAP_PROP_POS_FRAMES) % gesture_section) == 0:  # multi_scale_retinex 가우시안 필터를 적용할 32개의 프레임 저장
            gaussian = automatedMSRCR(img=frame2, sigma_list=Sigma_List)  # multi_scale_retinex 삽입
            gaussian = tf.image.random_crop(gaussian, [112, 112, 3])
            image_input_data.append(gaussian)  # gaussian 저장

        if int(cap.get(cv2.CAP_PROP_POS_FRAMES) % optical_flow_section) == 0:  # section별 계산을 위한 조건
            # optical_flow_input_data.append(bgr)  # section별로 한장씩 입력
            section_sum /= ((cap.get(cv2.CAP_PROP_FRAME_COUNT)) / SPLIT_SECTION)
            sum_average_of_frame += section_sum
            # cv2.imwrite('sample_hsv.PNG', hsv)
            # cv2.imwrite('smaple_bgr.PNG', bgr)
            frame_average.append(section_sum)
            get_section_frame.append(int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
            section_sum = 0

        # cv2.imshow('bgr', bgr)
        # cv2.imshow('video', frame2)
        # cv2.imshow('rgb', gaussian)
        # print(np.shape(optical_flow_list))  # optical_flow_list array shape : (count, height, weight, rgb)
        cv2.waitKey(delay)

    # 만약에 32개보다 작으면 더 채워주고 더 많으면 빼준다.
    while len(image_input_data) != NUM_GESTURE:
        if len(image_input_data) < NUM_GESTURE:
            image_input_data.insert(int(len(image_input_data) / 2), image_input_data[int(len(image_input_data) / 2)])
        elif len(image_input_data) > NUM_GESTURE:
            image_input_data.pop()
    count = 0
    for i in range(int(len(frame_average))):
        select_frame = frame_average[i] / sum_average_of_frame * NUM_GESTURE

        for cnt in range(int(round(select_frame))):
            if len(get_section_frame) == count:
                count -= 1
            if len(get_section_frame) - 1 == count:
                optical_flow_input_data.append(optical_flow_list[count])
                continue
            if get_section_frame[count + 1] <= count + cnt:
                if len(get_section_frame) >= count + 1:
                    optical_flow_input_data.append(optical_flow_list[-1])
                    continue
                optical_flow_input_data.append(optical_flow_list[get_section_frame[count + 1] - 1])
                continue
            if count + cnt >= len(optical_flow_list):
                optical_flow_input_data.append(optical_flow_list[-1])
                continue
            optical_flow_input_data.append(optical_flow_list[count + cnt])

        count += 1

    while len(optical_flow_input_data) != NUM_GESTURE:
        if len(optical_flow_input_data) < NUM_GESTURE:
            optical_flow_input_data.insert(int(len(optical_flow_input_data) / 2),
                                           optical_flow_input_data[int(len(optical_flow_input_data) / 2)])
        elif len(optical_flow_input_data) > NUM_GESTURE:
            optical_flow_input_data.pop()

    # for i in range(16):
    #     cv2.imshow('bgr', optical_flow_input_data[i])   # 튜블방식으로 저장되기 때문에 인덱스만 불러와도 가능
    #     cv2.waitKey(delay)
    # while len(optical_flow_input_data) != NUM_GESTURE:
    #     if len(optical_flow_input_data) < NUM_GESTURE:

    # frame_num = np.arange(1, len(frame_average) + 1)
    # print(frame_average)
    # plt.plot(frame_num, frame_average, marker='o')
    # plt.show()
    # cap.release()
    # print(len(image_input_data))
    # print(len(optical_flow_input_data))
    # for i in range(32):
    #     cv2.imshow('retinex', image_input_data[i])
    #     cv2.imshow('optical_flow', optical_flow_input_data[i])
    #     cv2.waitKey(delay)
    # for i in range(32):

    return image_input_data, optical_flow_input_data  # retinex와 optical flow 데이터 두개 return


def automatedMSRCR(img, sigma_list):
    img = np.float64(img) + 1.0

    img_retinex = multi_scale_retinex(img, sigma_list)

    for i in range(img_retinex.shape[2]):
        unique, count = np.unique(np.int32(img_retinex[:, :, i] * 100), return_counts=True)
        for u, c in zip(unique, count):
            if u == 0:
                zero_count = c
                break

        low_val = unique[0] / 100.0
        high_val = unique[-1] / 100.0
        for u, c in zip(unique, count):
            if u < 0 and c < zero_count * 0.1:
                low_val = u / 100.0
            if u > 0 and c < zero_count * 0.1:
                high_val = u / 100.0
                break

        img_retinex[:, :, i] = np.maximum(np.minimum(img_retinex[:, :, i], high_val), low_val)

        img_retinex[:, :, i] = (img_retinex[:, :, i] - np.min(img_retinex[:, :, i])) / \
                               (np.max(img_retinex[:, :, i]) - np.min(img_retinex[:, :, i])) \
                               * 255

    img_retinex = np.uint8(img_retinex)

    return img_retinex


def single_scale_retinex(img, sigma):
    retinex = np.log10(img) - np.log10(cv2.GaussianBlur(img, (0, 0), sigma))
    # print(retinex.shape)
    # print(type(retinex[0][0][0]))
    return retinex


def multi_scale_retinex(img, sigma_list):
    '''
    single scale retinex를 여러개 사용하여서 multi scale retinex로 적용시켜주는 함수
    :param img: img file
    :param sigma_list: multi scale 개수겸 시그마 리스트 입력
    :return: multi scale retinex
    '''

    retinex = np.zeros_like(img, dtype=np.float)
    # print(retinex.shape)
    # print(type(retinex[0][0][0]))
    for sigma in sigma_list:
        retinex += single_scale_retinex(img, sigma)
    retinex = retinex / len(sigma_list)

    return retinex
