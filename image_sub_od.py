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
import sys
import csv
import torch

# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
# from utils.torch_utils import select_device, smart_inference_mode
# from utils.plots import Annotator, colors, save_one_box
# from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
#                            increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
# from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
# from models.common import DetectMultiBackend
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device
 
'''
ref: https://blog.csdn.net/ericdiii/article/details/125632853
'''
class ImageSub(Node):
    def __init__(self, nodeName):
        super().__init__(nodeName)
        self.subscription = self.create_subscription(Image, 
        'techman_image', self.image_callback, 10)
        self.subscription
    
    def image_callback(self, data):
        # Load model
        weights = './runs/train/exp/weights/best.pt'
        w = str(weights[0] if isinstance(weights, list) else weights)
        device = select_device(0)
        model = torch.jit.load(w) if 'torchscript' in w else attempt_load(weights, map_location=device)   # load the model


        self.get_logger().info('Received image')
        img = np.array(data.data)
        img0 = img
        img = img.reshape((data.height,data.width,3))
        # h=960 w=1280
        img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        h,w=img.shape 
        # height, width = 640, 640                           # image size
        img = img / 255.
        img = img[:, :, ::-1].transpose((2, 0, 1))         # transfer from HWC to CHW
        img = np.expand_dims(img, axis=0)                  # extend the dimension to [1,3,640,640]
        img = torch.from_numpy(img.copy())                 # transfer from numpy to tensor
        img = img.to(torch.float32).to(device)             # transfer from float64 to float32
        pred = model(img, augment='store_true', visualize='store_true')[0]
        pred.clone().detach()
        pred = non_max_suppression(pred, 0.1, 0.45, None, False, max_det=1000)  # NMS: Non Max Suppression
        
        for i, det in enumerate(pred):
            if len(det):
                det = det.cpu()
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                # xyxy: coords, conf: conference, cls: classification results
                for *xyxy, conf, cls in reversed(det):

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
                    cv2.putText(img0,text=cls + ": " + str(format(conf.numpy(), '.2f')),org=(int(xyxy[0].numpy()), int(xyxy[1].numpy()) - 10),
                                fontFace=1,fontScale=1.5,thickness=2, color=(color[1], color[2], color[0]))

        print('fwscascaascegw')
        print(img.shape)
        print('fwegw')
        blur = cv2.GaussianBlur(img, (5, 5), cv2.BORDER_DEFAULT)
        print('gqegwg5w4')
        cv2.imwrite('out.jpg', img0)
        
        ret, thresh = cv2.threshold(blur, 210, 255, cv2.THRESH_BINARY_INV)
      
        contours, hierarchies = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        print(contours)
        blank = np.zeros(thresh.shape[:2], dtype='uint8')
        cv2.drawContours(blank, contours, -1, (255, 0, 0), 1)
        #cv2.imshow("blank",blank)

        px_list = []
        py_list = []
        angle_list = []
        for i, c in enumerate(contours):
            M = cv2.moments(c)
            if M['m00'] != 0:
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
     
                    print(cx, cy)

            # Calculate the area of each contour
            area = cv2.contourArea(c)
            # Ignore contours that are too small
            if area < 1000:
                continue
            
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
        # Retrieve the key parameters of the rotated bounding box
            center = (int(rect[0][0]), int(rect[0][1])) 
            width = float(rect[1][0])
            height = float(rect[1][1])
            angle = float(rect[2])
            print(angle)
                
            if width < height:
                angle = 90 - angle
            else:
                angle = -angle
            # angle = 0.5*math.atan2(2*u11, u20-u02)

            # angle_list.append(round(angle*4068/71, 4))
            if cx != 639 and cy != 479: # drop the center of the whole image
                cv2.drawContours(img, [c], -1, (0, 255, 0), 2)
                cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(img, f"center: ({cx}, {cy})", (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                # cv2.putText(img, f"center: ({cx_float}, {cy_float})", (cx - 80, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                label = "Principal angle: " + str(round(angle,4)) + " degrees"
                cv2.putText(img, label, (center[0]-200, center[1]+100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                #px = cx*0.237 - cy*0.236 + 332.148
                #px = -0.23688*cx+0.23688*cy+331.432
                #px = -0.23866*cx+0.23980*cy+409.243722
                px = 300-(cy-745)*0.338
                #py = -cx*0.246 - cy*0.239 + 81.574
                #py = 0.24024*cx+0.237552*cy+82.294
                #py = 0.23923*cx+0.23923*cy+63.8755981
                py = 300-(cx-630)*0.338
                px_list.append(px)
                py_list.append(py)
                angle2 = 90.00 + angle
                angle_list.append(angle2)

        for i in range(len(px_list)):
            h = 25
            print("px = ", px_list[i])
            print("py = ", py_list[i])
            targetP1 = f"{float(px_list[i])}, {float(py_list[i])}, 150, -180.00, 0.0, {float(angle_list[i])}"
            script1 = "PTP(\"CPP\","+targetP1+",100,200,0,false)"
            send_script(script1)
            targetP2 = f"{float(px_list[i])}, {float(py_list[i])}, 115, -180.00, 0.0, {float(angle_list[i])}"
            script2 = "PTP(\"CPP\","+targetP2+",100,200,0,false)"
            send_script(script2)
            time.sleep(8)
            set_io(1.0)
            time.sleep(3)
            script1 = "PTP(\"CPP\","+targetP1+",100,200,0,false)"
            send_script(script1)
            targetP2 = "300, 300, 500, -180.00, 0.0, 90.00"
            script2 = "PTP(\"CPP\","+targetP2+",100,200,0,false)"
            send_script(script2)
            put = h*i + 115
            targetP2 = f"300, 300, {put}, -180.00, 0.0, 90.00"
            script2 = "PTP(\"CPP\","+targetP2+",100,200,0,false)"
            send_script(script2)
            time.sleep(8)
            set_io(0.0)
            targetP2 = "300, 300, 500, -180.00, 0.0, 90.00"
            script2 = "PTP(\"CPP\","+targetP2+",100,200,0,false)"
            send_script(script2)
        cv2.imshow('eye on hand',img)
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