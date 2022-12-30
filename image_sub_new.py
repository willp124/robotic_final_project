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
from matplotlib import pyplot as plt

# from yolov5.models.experimental import attempt_load
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device
from models.common import DetectMultiBackend
from models.experimental import attempt_load
import torch

from cards_handler import Hands_divider

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
        names = model.names


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
        pred = non_max_suppression(pred, 0.01, 0.45, None, False, max_det=1000)  # NMS: Non Max Suppression
        cards = []
        cards_coordinate = []
        c_name = ['10c', '10d', '10h', '10s', '2c', '2d', '2h', '2s',
            '3c', '3d', '3h', '3s', '4c', '4d', '4h', '4s',
            '5c', '5d', '5h', '5s', '6c', '6d', '6h', '6s',
            '7c', '7d', '7h', '7s', '8c', '8d', '8h', '8s',
            '9c', '9d', '9h', '9s', 'Ac', 'Ad', 'Ah', 'As',
            'Jc', 'Jd', 'Jh', 'Js', 'Kc', 'Kd', 'Kh', 'Ks',
            'Qc', 'Qd', 'Qh', 'Qs']
        cls_name = ['C10', 'D10', 'H10', 'S10', 'C2', 'D2', 'H2', 'S2',
            'C3', 'D3', 'H3', 'S3', 'C4', 'D4', 'H4', 'S4',
            'C5', 'D5', 'H5', 'S5', 'C6', 'D6', 'H6', 'S6',
            'C7', 'D7', 'H7', 'S7', 'C8', 'D8', 'H8', 'S8',
            'C9', 'D9', 'H9', 'S9', 'C1', 'D1', 'H1', 'S1',
            'C11', 'D11', 'H11', 'S11', 'C13', 'D13', 'H13', 'S13',
            'C12', 'D12', 'H12', 'S12']

        bounding_box_dict = {}

        for i, det in enumerate(pred):
            if len(det):
                det = det.cpu()
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()
                # xyxy: coords, conf: conference, cls: classification results
                for *xyxy, conf, cls in reversed(det):

                    c = int(cls)
                    # if cls_name[c] not in cards:
                    #     cards += [cls_name[c]]
                    # c[cls]
                    label = f'{c_name[c]} {conf:.2f}'
                    # print the prediction information
                    print('{},{},{}'.format(xyxy, conf.numpy(), cls))

                    x1 = int(xyxy[0].item())
                    y1 = int(xyxy[1].item())
                    x2 = int(xyxy[2].item())
                    y2 = int(xyxy[3].item())

                    # class_idx = cls
                    # card_name = names[int(cls)]
                    print('Bounding box detected: ', x1, y1, x2, y2)
                    xc1 = int((x1 + x2) / 2)
                    yc1 = int((y1 + y2) / 2)
                    coordinate = (xc1, yc1)
                    print(f'Card: {label}, predicted center: ({xc1}, {yc1})')

                    bounding_box_dict[(c, coordinate)] = float(conf)

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

        # print(bounding_box_dict)
        while len(bounding_box_dict) != 0:
            delete = []
            items = list(bounding_box_dict.items())
            # values = list(bounding_box_dict.values())
            c, coord = items[0][0]
            confi = items[0][1]
            for key, value in items[1:]:
                if (key[1][0] - coord[0])**2 + (key[1][1] - coord[1])**2 < 100:
                    if bounding_box_dict[key] <= confi:
                        delete += [key]
                    else:
                        delete += [(c, coord)]
                        c, coord = key
                        confi = bounding_box_dict[key]
            if cls_name[c] not in cards and ((coord[1]>=250 and coord[1]<=300) or (coord[1]>=450 and coord[1]<=500)):
                cards += [cls_name[c]]
                cards_coordinate += [coord]
            delete += [(c, coord)]
            print(delete)
            for key in delete:
                del bounding_box_dict[key]
                # bounding_box_dict.remove(key)

        print("cards: ", cards)
        print("cards_coordinate: ", cards_coordinate)
        new_dict_low = {}
        new_dict_high = {}
        for k in range(len(cards)):
            if cards_coordinate[k][1] >= 250 and cards_coordinate[k][1] <= 300:
                new_dict_low[cards[k]] = cards_coordinate[k]
            if cards_coordinate[k][1] >= 450 and cards_coordinate[k][1] <= 500:
                new_dict_high[cards[k]] = cards_coordinate[k]
        new_dict_low = dict(sorted(new_dict_low.items(), key=lambda item: item[1]))
        new_dict_high = dict(sorted(new_dict_high.items(), key=lambda item: item[1]))
        print('new_dict_low = ', new_dict_low)
        print('new_dict_high = ', new_dict_high)
        cards_low = list(new_dict_low.keys())
        cards_high = list(new_dict_high.keys())
        cards_coordinate_low = list(new_dict_low.values())
        cards_coordinate_high = list(new_dict_high.values())
        cards = cards_high + cards_low
        cards_coordinate = cards_coordinate_high + cards_coordinate_low
        print("sorted cards: ", cards)
        print("sorted cards_coordinate: ", cards_coordinate)
        if len(cards) == 13:
            cards_divider = Hands_divider(cards)
            cards_divider.display_cards()
            print(cards_divider.divide())

            cv2.imwrite('out.jpg', img0)
            #cv2.imshow('detect_result', img0)
            #cv2.waitKey(1)
            # cv2.imshow('image',img)

            # Show the playing cards
            # create figure
            fig = plt.figure(figsize=(10, 7))

            # setting values to rows and column variables
            rows = 3
            columns = 5

            for i in range(13):
                image = cv2.imread(f'./cardsphoto/{cards_divider.hands[i]}.png')[...,::-1]
                # cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # Adds a subplot at the 1st position
                if i < 3:
                    fig.add_subplot(rows, columns, i+2)
                else:
                    fig.add_subplot(rows, columns, i+3)
                plt.imshow(image)
                plt.axis('off')
            plt.show()
            plt.savefig('playCards.png')

            targetP1 = "300, 300, 500, -180.00, 0.0, 90.00"
            script1 = "PTP(\"CPP\","+targetP1+",100,200,0,false)"
            send_script(script1)

            for i in range(13):
                index = cards.index(cards_divider.hands[i])
                time.sleep(3)
                set_io(1.0)  #grip
                time.sleep(3)

                if index<6: #first row
                    # posx = 360
                    posx = 366-index*43
                    # posy = 550
                    posy = 540-index*43
                    targetP2 = f"{posx-50},{posy+50}, 95, 110, 0, 45"  #pos of card rack related to index
                    script2 = "PTP(\"CPP\","+targetP2+",100,200,0,false)"
                    send_script(script2)
                    time.sleep(5)
                    targetP2 = f"{posx},{posy}, 95, 110, 0, 45"  #pos of card rack related to index
                    script2 = "PTP(\"CPP\","+targetP2+",100,200,0,false)"
                    send_script(script2)
                    time.sleep(5)
                    set_io(0.0)
                    time.sleep(5)
                    targetP2 = f"{posx},{posy}, 300, 110, 0, 45"  #pos of card rack related to index
                    script2 = "PTP(\"CPP\","+targetP2+",100,200,0,false)"
                    send_script(script2)

                else: #second row
                    # posx = 495
                    posx = 497-(index-6)*43
                    # posy = 435
                    posy = 433-(index-6)*43
                    targetP2 = f"{posx-20},{posy+20}, 180, 60, 180, 45"  #pos of card rack related to index
                    script2 = "PTP(\"CPP\","+targetP2+",100,200,0,false)"
                    send_script(script2)
                    targetP2 = f"{posx-15},{posy+15}, 150, 60, 180, 45"  #pos of card rack related to index
                    script2 = "PTP(\"CPP\","+targetP2+",100,200,0,false)"
                    send_script(script2)
                    time.sleep(5)
                    targetP2 = f"{posx},{posy}, 120, 60, 180, 45"  #pos of card rack related to index
                    script2 = "PTP(\"CPP\","+targetP2+",100,200,0,false)"
                    send_script(script2)
                    time.sleep(5)
                    set_io(0.0)
                    time.sleep(5)
                    targetP2 = f"{posx},{posy}, 300, 60, 180, 45"  #pos of card rack related to index
                    script2 = "PTP(\"CPP\","+targetP2+",100,200,0,false)"
                    send_script(script2)


                #release pos
                if i in range(3):
                    targetP2 = "700, 200, 400, -180, 0, 45"
                    script2 = "PTP(\"CPP\","+targetP2+",100,200,0,false)"
                    send_script(script2)

                    relposx = 895 - i*43
                    relposy = 145 - i*43
                    targetP2 = f"{relposx},{relposy}, 190, -180, 0, 45"
                    script2 = "PTP(\"CPP\","+targetP2+",100,200,0,false)"
                    send_script(script2)
                    time.sleep(12)
                    set_io(1.0)
                    time.sleep(8)
                    set_io(0.0)

                if i in range(3,8):
                    targetP2 = "700, 200, 400, -180, 0, 45"
                    script2 = "PTP(\"CPP\","+targetP2+",100,200,0,false)"
                    send_script(script2)

                    relposx = 835 - (i-3)*43
                    relposy = 265 - (i-3)*43
                    targetP2 = f"{relposx},{relposy}, 190, -180, 0, 45"
                    script2 = "PTP(\"CPP\","+targetP2+",100,200,0,false)"
                    send_script(script2)
                    time.sleep(12)
                    set_io(1.0)
                    time.sleep(8)
                    set_io(0.0)

                if i in range(8,13):
                    targetP2 = "700, 200, 400, -180, 0, 45"
                    script2 = "PTP(\"CPP\","+targetP2+",100,200,0,false)"
                    send_script(script2)

                    relposx = 745 - (i-8)*43
                    relposy = 355 - (i-8)*43
                    targetP2 = f"{relposx},{relposy}, 190, -180, 0, 45"
                    script2 = "PTP(\"CPP\","+targetP2+",100,200,0,false)"
                    send_script(script2)
                    time.sleep(12)
                    set_io(1.0)
                    time.sleep(8)
                    set_io(0.0)

                send_script(script1)
        else:
            send_script("Vision_DoJob(job1)")


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
