#-*- coding:utf-8 -*-
#'''
# Created on 18-8-20 下午6:48
#
# @Author: Greg Gao(laygin)
#'''
'''
mostly,refers to https://github.com/experiencor/keras-yolo3
'''
import numpy as np
import cv2
from scipy.special import expit


class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, c=None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.c = c

    def get_score(self):
        return self.c


def normalize(image):
    return image / 255.


def preprocess_input(image, net_h, net_w,normalized=True):
    new_h, new_w, _ = image.shape

    # determine the new size of the image
    if (float(net_w) / new_w) < (float(net_h) / new_h):
        new_h = (new_h * net_w) // new_w
        new_w = net_w
    else:
        new_w = (new_w * net_h) // new_h
        new_h = net_h

    if normalized:
        image = normalize(image[:, :, ::-1])
        resized = cv2.resize(image, (new_w, new_h))
    # resize the image to the new size
    else:
        resized = cv2.resize(image[:, :, ::-1] / 255., (new_w, new_h))

    # embed the image into the standard letter box
    new_image = np.ones((net_h, net_w, 3)) * 0.5
    new_image[(net_h - new_h) // 2:(net_h + new_h) // 2, (net_w - new_w) // 2:(net_w + new_w) // 2, :] = resized
    new_image = np.expand_dims(new_image, 0)

    return new_image


def decode_netout(netout, anchors, obj_thresh, net_h, net_w):
    grid_h, grid_w = netout.shape[:2]
    nb_box = 3
    netout = netout.reshape((grid_h, grid_w, nb_box, -1))
    # nb_class = netout.shape[-1] - 5

    boxes = []

    netout[..., :2] = _sigmoid(netout[..., :2])
    netout[..., 4] = _sigmoid(netout[..., 4])

    for i in range(grid_h * grid_w):
        row = i // grid_w
        col = i % grid_w

        for b in range(nb_box):
            # 4th element is objectness score
            objectness = netout[row, col, b, 4]

            if (objectness <= obj_thresh): continue

            # first 4 elements are x, y, w, and h
            x, y, w, h = netout[row, col, b, :4]

            x = (col + x) / grid_w  # center position, unit: image width
            y = (row + y) / grid_h  # center position, unit: image height
            w = anchors[2 * b + 0] * np.exp(w) / net_w  # unit: image width
            h = anchors[2 * b + 1] * np.exp(h) / net_h  # unit: image height

            box = BoundBox(x - w / 2, y - h / 2, x + w / 2, y + h / 2, objectness)

            boxes.append(box)

    return boxes


def draw_boxes(image, boxes, obj_thresh, rect=True, quiet=True):
    image_c = image.copy()
    for box in boxes:
        label_str = ''
        label = -1

        if box.c > obj_thresh:
            if label_str != '': label_str += ', '
            label_str += (str(round(box.c * 100, 2)) + '%')
            label = 0
        if not quiet: print(label_str)

        if label >= 0:
            text_size = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 1.1e-3 * image.shape[0], 2)

            if rect:
                cv2.rectangle(img=image_c, pt1=(box.xmin, box.ymin), pt2=(box.xmax, box.ymax), color=[255, 0, 0],
                              thickness=2)
            else:
                ra, rb, theta = int((box.xmax - box.xmin)/2), int((box.ymax - box.ymin)/2), 0
                cx, cy = box.xmin + ra, box.ymin + rb
                # print(ra, rb, cx, cy)
                cv2.ellipse(image_c, (cx,cy), (ra,rb), theta,0,360,(0,0,255),2)

            cv2.putText(img=image_c,
                        text=label_str,
                        org=(box.xmin + 13, box.ymax + 13),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1e-3 * image_c.shape[0],
                        color=(13, 200, 13),
                        thickness=2)

    return image_c



def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
    if (float(net_w) / image_w) < (float(net_h) / image_h):
        new_w = net_w
        new_h = (image_h * net_w) / image_w
    else:
        new_h = net_w
        new_w = (image_w * net_h) / image_h

    for i in range(len(boxes)):
        x_offset, x_scale = (net_w - new_w) / 2. / net_w, float(new_w) / net_w
        y_offset, y_scale = (net_h - new_h) / 2. / net_h, float(new_h) / net_h

        boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
        boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
        boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
        boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)


def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2, x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2, x4) - x3


def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])

    intersect = intersect_w * intersect_h

    w1, h1 = box1.xmax - box1.xmin, box1.ymax - box1.ymin
    w2, h2 = box2.xmax - box2.xmin, box2.ymax - box2.ymin

    union = w1 * h1 + w2 * h2 - intersect

    return float(intersect) / union


def do_nms(boxes, nms_thresh):
    # haah, if there is no boxes,break
    if len(boxes) > 0:
        sorted_indices = np.argsort([box.c for box in boxes])[::-1]
        for i in range(len(boxes)):
            index_i = sorted_indices[i]  # get the largest score box index

            # if boxes[index_i].classes[c] == 0: continue
            if boxes[index_i].c == 0: continue

            for j in range(i + 1, len(boxes)):
                index_j = sorted_indices[j]  # get the second large score box index

                iou = bbox_iou(boxes[index_i], boxes[index_j])

                if iou >= nms_thresh:
                    # boxes[index_j].c = 0
                    boxes[index_j].c = boxes[index_j].c * (1-iou)  # 18-7-18 apply soft nms

    else:
        return


def get_yolo_boxes(model, images, net_h, net_w, anchors, obj_thresh, nms_thresh):
    image_h, image_w, _ = images[0].shape
    nb_images = len(images)
    batch_input = np.zeros((nb_images, net_h, net_w, 3))

    # preprocess the input
    for i in range(nb_images):
        batch_input[i] = preprocess_input(images[i], net_h, net_w)

        # run the prediction
    batch_output = model.predict_on_batch(batch_input)
    batch_boxes = [None] * nb_images

    for i in range(nb_images):
        yolos = [batch_output[0][i], batch_output[1][i], batch_output[2][i]]
        boxes = []

        # decode the output of the network
        for j in range(len(yolos)):
            yolo_anchors = anchors[(2 - j) * 6:(3 - j) * 6]  # config['model']['anchors']
            boxes += decode_netout(yolos[j], yolo_anchors, obj_thresh, net_h, net_w)

        # correct the sizes of the bounding boxes
        correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w)

        # suppress non-maximal boxes
        do_nms(boxes, nms_thresh)

        batch_boxes[i] = boxes

    return batch_boxes


def _sigmoid(x):
    return expit(x)