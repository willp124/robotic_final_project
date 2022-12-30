#!/usr/bin/env python
import rclpy

from rclpy.node import Node

import sys
sys.path.append('/home/robot/colcon_ws/install/tm_msgs/lib/python3.6/site-packages')
from tm_msgs.msg import *
from tm_msgs.srv import *
import cv2
from sensor_msgs.msg import Image
import numpy as np
import math
import time
import os
import csv

# from yolov5.models.experimental import attempt_load
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device
from models.common import DetectMultiBackend
from models.experimental import attempt_load
import torch

class ImageSub(Node):
    def __init__(self, nodeName):
        super().__init__(nodeName)
        self.subscription = self.create_subscription(Image, 
        'techman_image', self.image_callback, 10)
        self.subscription
    
    def image_callback(self, data):
        self.get_logger().info('Received image')
        img = np.array(data.data)
        img = img.reshape((data.height,data.width,3))
         # h=960 w=1280
        cv2.imwrite('./img1.jpg', img)

        weights = './yolov5/runs/train/exp/weights/best.pt'
        w = str(weights[0] if isinstance(weights, list) else weights)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        model = torch.jit.load(w) if 'torchscript' in w else attempt_load(weights)   # load the model
        names= model.names


        self.get_logger().info('Received image')
        img0 = img
        # h=960 w=1280

        # h,w=img.shape 
        # height, width = 640, 640                           # image size
        img = img / 255.
        img = img[:, :, ::-1].transpose((2, 0, 1))         # transfer from HWC to CHW
        img = np.expand_dims(img, axis=0)                  # extend the dimension to [1,3,640,640]
        img = torch.from_numpy(img.copy())                 # transfer from numpy to tensor
        img = img.to(torch.float32).to(device)             # transfer from float64 to float32
        model = model.to(device)
        pred = model(img, augment='store_true', visualize='store_true')[0]
        pred.clone().detach()
        pred = non_max_suppression(pred, 0.1, 0.45, None, False, max_det=1000)  # NMS: Non Max Suppression
        
        for i, det in enumerate(pred):
            if len(det):
                det = det.cpu()
                # det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()
                # xyxy: coords, conf: conference, cls: classification results
                for *xyxy, conf, cls in reversed(det):
                    
                    c_name = ['10c', '10d', '10h', '10s', '2c', '2d', '2h', '2s',
                        '3c', '3d', '3h', '3s', '4c', '4d', '4h', '4s',
                        '5c', '5d', '5h', '5s', '6c', '6d', '6h', '6s',
                        '7c', '7d', '7h', '7s', '8c', '8d', '8h', '8s',
                        '9c', '9d', '9h', '9s', 'Ac', 'Ad', 'Ah', 'As',
                        'Jc', 'Jd', 'Jh', 'Js', 'Kc', 'Kd', 'Kh', 'Ks',
                        'Qc', 'Qd', 'Qh', 'Qs']

                    c = int(cls)
                    # c[cls]
                    label = f'{names[c]} {conf:.2f}'
                    # print the prediction information
                    print('{},{},{}'.format(xyxy, conf.numpy(), cls))

                    x1 = int(xyxy[0].item())
                    y1 = int(xyxy[1].item())
                    x2 = int(xyxy[2].item())
                    y2 = int(xyxy[3].item())

                    # class_idx = cls
                    # card_name = names[int(cls)]
                    print('Bounding box detected: ', x1, y1, x2, y2)
                    xc1 = int((x1 + x2)) / 2
                    yc1 = ((y1 + y2)) / 2
                    print(f'Card: {label}, predicted center: ({xc1}, {yc1})')

                    data = [[str(label), str(xc1), str(yc1)]]
                    with open('detection_result.csv', 'w', encoding='utf-8', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerows(data)

                    # draw the bounding box
                    img0 = cv2.rectangle(img0, (int(xyxy[0].numpy()), int(xyxy[1].numpy())), (int(xyxy[2].numpy()), int(xyxy[3].numpy())), (0, 255, 0), 2)
                    # draw the class results
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    color = (np.random.randint(255), np.random.randint(255), np.random.randint(255))
                    cv2.putText(img0,text=str(c_name[c]) + ": " + str(format(conf.numpy(), '.2f')),org=(int(xyxy[0].numpy()), int(xyxy[1].numpy()) - 10),
                                fontFace=1,fontScale=1.5,thickness=2, color=(color[1], color[2], color[0]))
        cv2.imwrite('out.jpg', img0)
        # cv2.imshow('detect_result', img0)
        # cv2.imshow('image',img)

        cv2.waitKey(1)

        # TODO (write your code here)

def send_script(script):
    arm_node = rclpy.create_node('arm')
    arm_cli = arm_node.create_client(SendScript, 'send_script')

    while not arm_cli.wait_for_service(timeout_sec=1.0):
        arm_node.get_logger().info('service not availabe, waiting again...')

    move_cmd = SendScript.Request()
    move_cmd.script = script
    arm_cli.call_async(move_cmd)
    arm_node.destroy_node()

def set_io(state):
    gripper_node = rclpy.create_node('gripper')
    gripper_cli = gripper_node.create_client(SetIO, 'set_io')

    while not gripper_cli.wait_for_service(timeout_sec=1.0):
        node.get_logger().info('service not availabe, waiting again...')
    
    io_cmd = SetIO.Request()
    io_cmd.module = 1
    io_cmd.type = 1
    io_cmd.pin = 0
    io_cmd.state = state
    gripper_cli.call_async(io_cmd)
    gripper_node.destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = ImageSub('image_sub')
    
    rclpy.spin(node)
    

    

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
