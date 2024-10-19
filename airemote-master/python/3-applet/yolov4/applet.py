#!/usr/bin/env python3
"""
Inferemote: a Remote Inference Toolkit for Ascend 310

Copyright (c) 2021 Jiasheng Hao <haojiash@qq.com>
(University of Electronic Science and Technology of China, UESTC)

Permission  is  hereby  granted,  free  of  charge,  to  any  person
obtaining a copy of this software and associated documentation files
(the  "Software"),  to deal  in  the  Software without  restriction,
including without limitation the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of the Software,
and to  permit persons to whom  the Software is furnished  to do so,
subject to the following conditions:

The  above copyright  notice  and this  permission  notice shall  be
included in all copies or substantial portions of the Software.

THE  SOFTWARE IS  PROVIDED "AS  IS", WITHOUT  WARRANTY OF  ANY KIND,
EXPRESS OR IMPLIED,  INCLUDING BUT NOT LIMITED TO  THE WARRANTIES OF
MERCHANTABILITY,    FITNESS   FOR    A   PARTICULAR    PURPOSE   AND
NONINFRINGEMENT. IN NO EVENT SHALL  THE AUTHORS OR COPYRIGHT HOLDERS
BE LIABLE FOR  ANY CLAIM, DAMAGES OR OTHER LIABILITY,  WHETHER IN AN
ACTION OF  CONTRACT, TORT OR OTHERWISE,  ARISING FROM, OUT OF  OR IN
CONNECTION WITH  THE SOFTWARE OR  THE USE  OR OTHER DEALINGS  IN THE
SOFTWARE.

"""
import os, sys
import pickle, zstd
import numpy as np

from inferemote.atlas_applet import AtlasApplet
from inferemote.image_encoder import ImageEncoder

class Applet(AtlasApplet):

    MODEL_WIDTH  = 608
    MODEL_HEIGHT = 608
    ''' This shape should be changed due to specific input images/video '''

    def __init__(self, orig_shape):
        self.orig_shape = orig_shape
        super().__init__(port=5560)
  
    def helpinfo(self):
        return (''' \n 
This is a demo applet for YoloV4, please **CARE** about the ``orig_shape'' of images from your application.\n
        ''')

    def pre_process(self, blob):
        image = ImageEncoder.bytes_to_image(blob)
        ''' This floating operations shoud be done outside of cv2 encodings '''
        image = image.astype(np.float32)
        image = image / 255
        '''Transposing must go here, after jpeg decoding by cv2'''
        image = image.transpose(2, 0, 1).tobytes()

        return image

    def post_process(self, result):
        output0 = np.frombuffer(result[0], np.float32)
        output0 = output0.reshape((1,255,19,19,))
        output1 = np.frombuffer(result[1], np.float32)
        output1 = output1.reshape((1,255,38,38,))
        output2 = np.frombuffer(result[2], np.float32)
        output2 = output2.reshape((1,255,76,76,))
        infer_output = [output0, output1, output2]

        ''' Attension here! order of dims ''' 
        img_h = self.orig_shape[0]
        img_w = self.orig_shape[1]

        scale = min(float(self.MODEL_WIDTH) / float(img_w), float(self.MODEL_HEIGHT) / float(img_h))
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        shift_x_ratio = (self.MODEL_WIDTH - new_w) / 2.0 / self.MODEL_WIDTH
        shift_y_ratio = (self.MODEL_HEIGHT - new_h) / 2.0 / self.MODEL_HEIGHT
        class_number = len(labels)
        num_channel = 3 * (class_number + 5)
        x_scale = self.MODEL_WIDTH / float(new_w)
        y_scale = self.MODEL_HEIGHT / float(new_h)
        all_boxes = [[] for ix in range(class_number)]
        for ix in range(3):    
            pred = infer_output[ix]
            #print('pred.shape', pred.shape)
            anchors = anchor_list[ix]
            boxes = decode_bbox(pred, anchors, img_w, img_h, x_scale, y_scale, shift_x_ratio, shift_y_ratio)
            all_boxes = [all_boxes[iy] + boxes[iy] for iy in range(class_number)]

        res = apply_nms(all_boxes, iou_threshold)
        result_list = {}
        if not res:
            result_list['detection_classes'] = []
            result_list['detection_boxes'] = []
            result_list['detection_scores'] = []
        else:
            new_res = np.array(res)
            picked_boxes = new_res[:, 0:4]
            picked_boxes = picked_boxes[:, [1, 0, 3, 2]]
            picked_classes = convert_labels(new_res[:, 4])
            picked_score = new_res[:, 5]
            result_list['detection_classes'] = picked_classes
            result_list['detection_boxes'] = picked_boxes.tolist()
            result_list['detection_scores'] = picked_score.tolist()
        
        blob = pickle.dumps(result_list)
        #blob = zstd.compress(blob, 1)
        return [blob]

    def _post_process(self, result):
        blob = pickle.dumps(result)
        blob = zstd.compress(blob, 1)
        return [blob]


conf_threshold = 0.8
iou_threshold = 0.3

''' internal functions supporting the above Class '''
labels = ["person",
        "bicycle", "car", "motorbike", "aeroplane",
        "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
        "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
        "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
        "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
        "pizza", "donut", "cake", "chair", "sofa", "potted plant", "bed", "dining table",
        "toilet", "TV monitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush"]

class_num = 80
stride_list = [32, 16, 8]
anchors_3 = np.array([[12, 16], [19, 36], [40, 28]]) / stride_list[2]
anchors_2 = np.array([[36, 75], [76, 55], [72, 146]]) / stride_list[1]
anchors_1 = np.array([[142, 110], [192, 243], [459, 401]]) / stride_list[0]
anchor_list = [anchors_1, anchors_2, anchors_3]

conf_threshold = 0.8
iou_threshold = 0.3

def overlap(x1, x2, x3, x4):
    left = max(x1, x3)
    right = min(x2, x4)
    return right - left

def cal_iou(box, truth):
    w = overlap(box[0], box[2], truth[0], truth[2])
    h = overlap(box[1], box[3], truth[1], truth[3])
    if w <= 0 or h <= 0:
        return 0
    inter_area = w * h
    union_area = (box[2] - box[0]) * (box[3] - box[1]) + (truth[2] - truth[0]) * (truth[3] - truth[1]) - inter_area
    return inter_area * 1.0 / union_area

def apply_nms(all_boxes, thres):
    res = []

    for cls in range(class_num):
        cls_bboxes = all_boxes[cls]
        sorted_boxes = sorted(cls_bboxes, key=lambda d: d[5])[::-1]

        p = dict()
        for i in range(len(sorted_boxes)):
            if i in p:
                continue

            truth = sorted_boxes[i]
            for j in range(i + 1, len(sorted_boxes)):
                if j in p:
                    continue
                box = sorted_boxes[j]
                iou = cal_iou(box, truth)
                if iou >= thres:
                    p[j] = 1

        for i in range(len(sorted_boxes)):
            if i not in p:
                res.append(sorted_boxes[i])
    return res

def _sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def decode_bbox(conv_output, anchors, img_w, img_h, x_scale, y_scale, shift_x_ratio, shift_y_ratio):
    #print('conv_output.shape', conv_output.shape)
    _, _, h, w = conv_output.shape 
    conv_output = conv_output.transpose(0, 2, 3, 1)
    pred = conv_output.reshape((h * w, 3, 5 + class_num))

    '''FIXED: ValueError: assignment destination is read-only  '''
    pred = pred.copy()

    pred[..., 4:] = _sigmoid(pred[..., 4:])
    pred[..., 0] = (_sigmoid(pred[..., 0]) + np.tile(range(w), (3, h)).transpose((1, 0))) / w
    pred[..., 1] = (_sigmoid(pred[..., 1]) + np.tile(np.repeat(range(h), w), (3, 1)).transpose((1, 0))) / h
    pred[..., 2] = np.exp(pred[..., 2]) * anchors[:, 0:1].transpose((1, 0)) / w
    pred[..., 3] = np.exp(pred[..., 3]) * anchors[:, 1:2].transpose((1, 0)) / h

    bbox = np.zeros((h * w, 3, 4))
    bbox[..., 0] = np.maximum((pred[..., 0] - pred[..., 2] / 2.0 - shift_x_ratio) * x_scale * img_w, 0)  # x_min
    bbox[..., 1] = np.maximum((pred[..., 1] - pred[..., 3] / 2.0 - shift_y_ratio) * y_scale * img_h, 0)  # y_min
    bbox[..., 2] = np.minimum((pred[..., 0] + pred[..., 2] / 2.0 - shift_x_ratio) * x_scale * img_w, img_w)  # x_max
    bbox[..., 3] = np.minimum((pred[..., 1] + pred[..., 3] / 2.0 - shift_y_ratio) * y_scale * img_h, img_h)  # y_max
    pred[..., :4] = bbox
    pred = pred.reshape((-1, 5 + class_num))
    pred[:, 4] = pred[:, 4] * pred[:, 5:].max(1)
    pred[:, 5] = np.argmax(pred[:, 5:], axis=-1)    
    pred = pred[pred[:, 4] >= 0.2]
    #print('pred[:, 5]', pred[:, 5])
    #print('pred[:, 5] shape', pred[:, 5].shape)

    all_boxes = [[] for ix in range(class_num)]
    for ix in range(pred.shape[0]):
        box = [int(pred[ix, iy]) for iy in range(4)]
        box.append(int(pred[ix, 5]))
        box.append(pred[ix, 4])
        all_boxes[box[4] - 1].append(box)
    return all_boxes

def convert_labels(label_list):
    if isinstance(label_list, np.ndarray):
        label_list = label_list.tolist()
        label_names = [labels[int(index)] for index in label_list]
    return label_names


if __name__ == '__main__':
    '''Set the original shape of images from you applications here.'''
    orig_shape = (720, 1280, 3)

    Applet(orig_shape).run()

# Ends.
