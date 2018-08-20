#-*- coding:utf-8 -*-
#'''
# Created on 18-8-20 下午6:58
#
# @Author: Greg Gao(laygin)
#'''

net_h = 320  # fixed
net_w = 320  # fixed
anchors = [26,42, 51,85, 67,138, 93,107, 93,193, 128,149, 142,289, 192,212, 272,306]
obj_thresh = 0.9
nms_thresh = 0.3
weights_name = r'./weights/shufflenetv2.h5'
