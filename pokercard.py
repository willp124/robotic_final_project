import os
import torch
from yolov5 import utils
# display = utils.notebook_init()  # checks

python3 train.py - -img 416 - -batch 16 - -workers 4 - -epochs 120 - -data .. / data.yaml - -weights yolov5s.pt - -cache
os.system("")
