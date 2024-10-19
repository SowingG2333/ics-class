"""
Inferemote: a Remote Inference Toolkit for Ascend 310

Base on code from https://gitee.com/ascend/samples.git
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
import sys, os, time
import numpy as np
import cv2 as cv
import pickle, json, zstd
sys.path.append(os.path.dirname(__file__))
import contrib.LaneFinder as LaneFinder

def mark_objects(frame, result_list):
    
    frame_with_lane = find_lane(frame)

    distance = np.zeros(shape=(len(result_list['detection_classes']), 1))
    for i in range(len(result_list['detection_classes'])):
        box = result_list['detection_boxes'][i]
        class_name = result_list['detection_classes'][i]
        confidence = result_list['detection_scores'][i]
        distance[i] = calculate_position(bbox=box, transform_matrix=perspective_transform,
                    warped_size=WARPED_SIZE, pix_per_meter=pixels_per_meter)
        label_dis = '{} {:.2f}m'.format('dis:', distance[i][0])
        cv.putText(frame_with_lane, label_dis, (int(box[1]) + 10, int(box[2]) + 15),
                    cv.FONT_ITALIC, 0.6, colors[i % 6], 1)

        cv.rectangle(frame_with_lane, (int(box[1]), int(box[0])), (int(box[3]), int(box[2])), colors[i % 6])
        p3 = (max(int(box[1]), 15), max(int(box[0]), 15))
        out_label = class_name
        cv.putText(frame_with_lane, out_label, p3, cv.FONT_ITALIC, 0.6, colors[i % 6], 1)

    return frame_with_lane

# Class definition ends.

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

path = os.path.dirname(__file__)+'/contrib/configure.json'
config_file = open(path, "rb")
fileJson = json.load(config_file)
cam_matrix = fileJson[0]["cam_matrix"]
dist_coeffs = fileJson[0]["dist_coeffs"]
perspective_transform = fileJson[0]["perspective_transform"]
pixels_per_meter = fileJson[0]["pixels_per_meter"]
WARPED_SIZE = fileJson[0]["WARPED_SIZE"]
ORIGINAL_SIZE = fileJson[0]["ORIGINAL_SIZE"]

cam_matrix = np.array(cam_matrix)
dist_coeffs = np.array(dist_coeffs)
perspective_transform = np.array(perspective_transform)
pixels_per_meter = tuple(pixels_per_meter)
WARPED_SIZE = tuple(WARPED_SIZE)
ORIGINAL_SIZE = tuple(ORIGINAL_SIZE)

lf = LaneFinder.LaneFinder(ORIGINAL_SIZE, WARPED_SIZE, cam_matrix, dist_coeffs,
                           perspective_transform, pixels_per_meter)

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

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0)]

def find_lane(bgr_img):
    bgr_img = bgr_img[:, :, ::-1]
    image = bgr_img
    image = LaneFinder.Image.fromarray(image.astype('uint8'), 'RGB')

    Width = bgr_img.shape[1]
    Height = bgr_img.shape[0]
    lf.set_img_size((Width, Height))

    fframe = np.array(image)
    fframe = lf.process_image(fframe, False)
    frame = LaneFinder.Image.fromarray(fframe)
    framecv = cv.cvtColor(np.asarray(frame), cv.COLOR_RGB2BGR)
    return framecv

def calculate_position(bbox, transform_matrix, warped_size, pix_per_meter):
    if len(bbox) == 0:
        print('Nothing')
    else:
        point = np.array((bbox[1] / 2 + bbox[3] / 2, bbox[2])).reshape(1, 1, -1)
        pos = cv.perspectiveTransform(point, transform_matrix).reshape(-1, 1)
        return np.array((warped_size[1] - pos[1]) / pix_per_meter[1])


''' Ends. '''
