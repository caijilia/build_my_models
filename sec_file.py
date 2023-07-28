import torch
import numpy as np
import sklearn
import cv2

print(np.__version__)
print(torch.__version__)

class unet:
    def __init__(self, chnnels=6):
        self.ch = channels
    def forward(self, x):
        x1 = self.ch + 256
        return x1

class res_unet:
    def __init__(self, chnnels=6, weight = 0.5):
        self.ch = chnnels
        self.weight = weight
    def forward(self, x):
        x1 = self.ch + 256 + (self.ch * weight)
        return x1
