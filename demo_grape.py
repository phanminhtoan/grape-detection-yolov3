import argparse
import time
from pathlib import Path
import os
import cv2
import torch
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from numpy import random
import numpy as np
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
os.environ['KMP_DUPLICATE_LIB_OK']='True'

model = torch.load('yolov3_grape.pt', map_location='cpu')['model'].float().fuse().eval()
labels = ['Grape Chardonnay','Grape Cabernet Franc', 'Grape Cabernet Sauvignon', 'Grape Sauvignon Blanc', 'Grape Syrah']
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def predict(model, img):
    #img = cv2.imread(os.path.join(image_path, img_name))
    img_org = img.copy()
    h,w,s = img_org.shape
    img = letterbox(img, new_shape = 640)[0]

    img = img[:, :, ::-1].transpose(2, 0, 1) 
    image = img.astype(np.float32)
    image /= 255.0
    image = np.expand_dims(image, 0)

    image = torch.from_numpy(image)

    #print("shape tensor image:", image.shape)

    pred = model(image)[0]
    # print("pred shape:", pred.shape)
    temp_img = None
    pred = non_max_suppression(pred, 0.5, 0.5,None)
    #print(pred[0])
    num_boxes  = 0 
    for i, det in enumerate(pred):
            im0 = img_org
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(image.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                
                # Write results
            
                for *xyxy, conf, cls in reversed(det):
                    check = True 
                    bbox = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    bbox_new = bbox.clone() if isinstance(bbox, torch.Tensor) else np.copy(bbox)
                    #line = (cls, *xywh, conf)  # label format
                    bbox_new[0] = bbox[0] - bbox[2] / 2  # top left x
                    bbox_new[1] = bbox[1] - bbox[3] / 2  # top left y
                    bbox_new[2] = bbox[0] + bbox[2] / 2  # bottom right x
                    bbox_new[3] = bbox[1] + bbox[3] / 2  # bottom right y

                    bbox_new[0] = bbox_new[0] * w
                    bbox_new[2] = bbox_new[2] * w
                    bbox_new[1] = bbox_new[1] * h
                    bbox_new[3] = bbox_new[3] * h
                    #print("class: ", labels[int(cls)])
                    #print("conf: ", float(conf))
                    num_boxes = num_boxes + 1
                    cv2.rectangle(img_org,(int(bbox_new[0]), int(bbox_new[1])), (int(bbox_new[2]), int(bbox_new[3])), (0,255,0), 3)
    return img_org, num_boxes    

frame_rate = 1
prev = 0

cap = cv2.VideoCapture("grape.mp4")
cap.set(cv2.CAP_PROP_FPS, 1)
frame_temp = None
check = None
while(True):
    # Capture frame-by-frame
    # start_time = time.time()
    # print(start_time)
    time_elapsed = time.time() - prev
    #print(time_elapsed)
    ret, frame = cap.read()
    #ß
    #time.sleep(1.0)
    # Our operations on the frame come here
    if time_elapsed > 3: #1./frame_rate:
        prev = time.time()
        frame, num_boxes = predict(model, frame)
        #print("time:", time.time()- prev)
        frame_temp = frame.copy()
        check = True
    # Display the resulting ßframe
    if check:
        frame_t = cv2.putText(frame_temp, "So luong chum nho: "+ str(num_boxes), (00, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA, False)
        cv2.imshow('He thong dem nho',frame_t)
    else :
        cv2.imshow('He thong dem nho',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()