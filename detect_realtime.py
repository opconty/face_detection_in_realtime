#-*- coding: utf-8 -*-
#'''
# Created on 2018/8/7 14:25
#
# @Author: Greg Gao (laygin)
#'''
import cv2
import os
from keras.models import load_model
from config import *
from utils import get_yolo_boxes, draw_boxes
import argparse


ap = argparse.ArgumentParser('face detection in realtime.')
ap.add_argument('-m', '--model', default=weights_name,
                help='path to pre-trained weights.')
ap.add_argument('-v', '--video', default=None, help='path to video.')
args = vars(ap.parse_args())

os.environ['CUDA_VISIBLE_DEVICES'] = ''
model = load_model(args['model'])


if os.path.exists(args['video']):
    camera = cv2.VideoCapture(args['video'])
else:
    camera = cv2.VideoCapture(0)

while camera.isOpened():
    ret, frame = camera.read()
    h, w = frame.shape[:2]
    boxes = get_yolo_boxes(model, [frame], net_h, net_w, anchors, obj_thresh, nms_thresh)[0]

    print('[INFO] length of boxes{}'.format(len(boxes)))
    frame = draw_boxes(frame, boxes, obj_thresh, rect=False)

    cv2.imshow('Face Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()

if __name__ == '__main__':

    pass
