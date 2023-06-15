#!/usr/bin/env python
# coding: utf-8


import optparse
import torch
import pandas as pd

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

from pathlib import Path
import os
import platform
import sys

import pandas as pd

FILE = Path('.').resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

ROOT = Path(os.path.relpath('.', Path.cwd()))
ROOT


parser = optparse.OptionParser()

parser.add_option('--img', help='Pass the image\'s path ', default='.')
parser.add_option('--name', help='Pass the image name', default='index')

(opts, args) = parser.parse_args()  # instantiate parser

img = opts.img
filename = opts.name


# param
weights = f'{ROOT}/pretrain_model/yolov5x6[exp]_on_all_yolo_dataset/weights/last.pt'
#
data = f'{ROOT}/data/coco128.yaml'  # dataset.yaml path
imgsz = (640, 640)  # inference size (height, width)
conf_thres = 0.50  # confidence threshold
iou_thres = 0.45  # NMS IOU threshold
max_det = 50  # maximum detections per image
device = 'cpu'  # cuda device, i.e. 0 or 0,1,2,3 or cpu
view_img = False  # show results
save_txt = False  # save results to *.txt
save_conf = False  # save confidences in --save-txt labels
save_crop = False  # save cropped prediction boxes
nosave = False  # do not save images/videos
classes = None  # filter by class: --class 0, or --class 0 2 3
agnostic_nms = False  # class-agnostic NMS
augment = False  # augmented inference
visualize = False  # visualize features
update = False  # update all models
project = f'{ROOT}/runs/detect'  # save results to project/name
name = 'exp',  # save results to project/name
exist_ok = False  # existing project/name ok, do not increment
line_thickness = 3  # bounding box thickness (pixels)
hide_labels = False  # hide labels
hide_conf = False  # hide confidences
half = False  # use FP16 half-precision inference
dnn = False  # use OpenCV DNN for ONNX inference
vid_stride = 1


model = DetectMultiBackend(weights, device=device,
                           dnn=dnn, data=data, fp16=half)


# img ='/home/uwu/Desktop/object_detection/preview/sketch_web_ui_dataset/all_yolo/images/16483b6e-47fe-4e9c-b78c-7f0ba4fef80c.png'


# img = '/home/uwu/Desktop/object_detection/preview/sketch_web_ui_dataset/new_dataset/80ce0ae6-9f89-429b-b2da-ae1b25e45acb.png'


source = str(img)
save_img = not nosave and not source.endswith('.txt')  # save inference images
is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))

if is_url and is_file:
    source = check_file(source)  # download

# Directories
save_dir = '.'  # make dir

# Load model
device = select_device(device)
model = DetectMultiBackend(weights, device=device,
                           dnn=dnn, data=data, fp16=half)
stride, names, pt = model.stride, model.names, model.pt
imgsz = check_img_size(imgsz, s=stride)  # check image size

# Dataloader
bs = 1
stride, names, pt = model.stride, model.names, model.pt


dataset = LoadImages(source, img_size=imgsz, stride=stride,
                     auto=pt, vid_stride=vid_stride)


model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup


seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

for path, im, im0s, vid_cap, s in dataset:
    with dt[0]:
        im = torch.from_numpy(im).to(model.device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

    # Inference
    with dt[1]:
        visualize = increment_path(
            save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)

    # NMS
    with dt[2]:
        pred = non_max_suppression(
            pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)


# process pred to df
det = pred[0]
pred_df = pd.DataFrame(
    columns=['class', 'xmin', 'ymin', 'xmax', 'ymax', 'conf'])

p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
p = Path(p)  # to Path
gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
imc = im0.copy() if save_crop else im0  # for save_crop
annotator = Annotator(im0, line_width=line_thickness, example=str(names))
if len(det):
    # Rescale boxes from img_size to im0 size
    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()


for *xyxy, conf, cls in reversed(det):
    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) /
            gn).view(-1).tolist()  # normalized xywh

    # Add bbox to image
    clsInt = int(cls)  # integer class
    label = None if hide_labels else (
        names[c] if hide_conf else f'{names[clsInt]} {conf:.2f}')

    annotator.box_label(xyxy, label, color=colors(clsInt, True))

    row = [
        names[clsInt],
        int(xyxy[0]),
        int(xyxy[1]),
        int(xyxy[2]),
        int(xyxy[3]),
        float(f'{conf:.2f}')
    ]
    pred_df.loc[len(pred_df)+1] = row
#     print("xywh:",xywh)
#     print("xyxy:",int(xyxy[0]))
#     print("xyxy:",xyxy)
#     print('cls:',cls)
#     print('conf:',conf)
#     print('label:',label)
#     print('row:',row)
#     print('\n')


pred_img = annotator.result()

cv2.imwrite('../predicted/1.jpg', pred_img)

# cv2.imshow('result', pred_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# print(pred_img)
# print(im0)


# pred_df = pred_df.sort_values(by=['xmin'], ascending=[True])
pred_df = pred_df.sort_values(by=['ymin'], ascending=[True])
pred_df.to_csv('../predicted/DSL.csv', index=None)
column = pred_df['class']
my_list = pred_df["class"].tolist()
result = ''
with open(f"../predicted/{filename}.txt", "w") as File_object:
    # File_object.write("{\n")
    for item in my_list:
        item += ','+'\n'
        File_object.write(item)

    # File_object.write("}")


File_object.close()
