#!/usr/bin/env python

import rclpy
import cv2
import sys
sys.path.append('/home/robot/colcon_ws/install/tm_msgs/lib/python3.6/site-packages')
from tm_msgs.msg import *
from tm_msgs.srv import *
import time
import pandas as pd
import csv

# arm client
def send_script(script):
    arm_node = rclpy.create_node('arm')
    arm_cli = arm_node.create_client(SendScript, 'send_script')

    while not arm_cli.wait_for_service(timeout_sec=1.0):
        arm_node.get_logger().info('service not availabe, waiting again...')

    move_cmd = SendScript.Request()
    move_cmd.script = script
    arm_cli.call_async(move_cmd)
    arm_node.destroy_node()

# gripper client
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
    #--- move command by joint angle ---#
    # script = 'PTP(\"JPP\",45,0,90,0,90,0,35,200,0,false)'

    #--- move command by end effector's pose (x,y,z,a,b,c) ---#
    # targetP1 = "398.97, -122.27, 748.26, -179.62, 0.25, 90.12"s
    # set_io(1.0) # 1.0: close gripper, 0.0: open gripper
    # set_io(0.0)
    # Initial camera position for taking image (Please do not change the values)
    # For right arm: targetP1 = "230.00, 230, 730, -180.00, 0.0, 135.00"
    # For left  arm: targetP1 = "350.00, 350, 730, -180.00, 0.0, 135.00"

    '''
    targetP1 = "495, 435, 120, 60, 180, 45" #right end
    #targetP1 = "690,640 , 120, 110, 0, 45" #right limit
    #targetP1 = "470,-60 , 100, -95, 0, 45" #left end
    script1 = "PTP(\"CPP\","+targetP1+",100,200,0,false)"

    send_script(script1)
    #send_script("Vision_DoJob(job1)")
    #cv2.waitKey(1)
    time.sleep(3)

    '''

    targetP1 = "300, 300, 500, -180.00, 0.0, 90.00"
    script1 = "PTP(\"CPP\","+targetP1+",100,200,0,false)"
    send_script(script1)


    for i in range(10,13):
        targetP2 = "460, -70, 500, -180, 0.0, 45"  #original pos of cards (high)
        script2 = "PTP(\"CPP\","+targetP2+",100,200,0,false)"
        send_script(script2)
        time.sleep(3)
        set_io(1.0)  #grip
        time.sleep(3)
        targetP2 = "460, -70, 250, -180, 0.0, 45"  #original pos of cards (high)
        script2 = "PTP(\"CPP\","+targetP2+",100,200,0,false)"
        send_script(script2)
        h = 175
        h = h-i/2
        targetP2 = f"460, -70, {h}, -180, 0.0, 45"  #original pos of cards (down)
        script2 = "PTP(\"CPP\","+targetP2+",100,200,0,false)"
        send_script(script2)
        time.sleep(5)
        set_io(0.0)  #suck


        for j in range(2):
            targetP2 = "465, -75, 190, -180, 0.0, 45"  #original pos of cards (down)
            script2 = "PTP(\"CPP\","+targetP2+",100,200,0,false)"
            send_script(script2)
            time.sleep(1)
            targetP2 = "465, -75, 195, -180, 0.0, 45"  #original pos of cards (down)
            script2 = "PTP(\"CPP\","+targetP2+",100,200,0,false)"
            send_script(script2)
            time.sleep(1)

        time.sleep(3)

        targetP2 = "465, -75, 200, -180, 0, 45"
        script2 = "PTP(\"CPP\","+targetP2+",100,200,0,false)"
        send_script(script2)
        targetP2 = "465, -75, 300, -180, 0, 45"
        script2 = "PTP(\"CPP\","+targetP2+",100,200,0,false)"
        send_script(script2)

        if i<6:
            # posx = 610
            posx = 607-i*43
            # posy = 320
            posy = 295-i*43
            targetP2 = f"{posx},{posy}, 200, -110, 0, 45"  #pos of card rack related to i
            script2 = "PTP(\"CPP\","+targetP2+",100,200,0,false)"
            send_script(script2)
            targetP2 = f"{posx},{posy}, 137, -110, 0, 45"  #pos of card rack related to i
            script2 = "PTP(\"CPP\","+targetP2+",100,200,0,false)"
            send_script(script2)
            time.sleep(5)

        else:
            # posx = 735
            posx = 739-(i-6)*43
            # posy = 205
            posy = 185-(i-6)*43
            targetP2 = f"{posx},{posy}, 200, -100, 0, 45"  #pos of card rack related to i
            script2 = "PTP(\"CPP\","+targetP2+",100,200,0,false)"
            send_script(script2)
            targetP2 = f"{posx},{posy}, 110, -100, 0, 45"  #pos of card rack related to i
            script2 = "PTP(\"CPP\","+targetP2+",100,200,0,false)"
            send_script(script2)
            time.sleep(5)

        time.sleep(5)
        set_io(1.0)  #release card
        time.sleep(3)
        set_io(0.0)

    targetP2 = "480,-72, 300, -100, 0, 45"  #pos of card rack related to i
    script2 = "PTP(\"CPP\","+targetP2+",100,200,0,false)"
    send_script(script2)

    send_script(script1)

    targetP1 = "100, 610, 250, 125, 0, 45" #move to photo pos
    script1 = "PTP(\"CPP\","+targetP1+",100,200,0,false)"
    send_script(script1)

    send_script("Vision_DoJob(job1)")
    cv2.waitKey(1)

    rclpy.shutdown()

if __name__ == '__main__':
    main()




