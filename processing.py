import cv2
import numpy as np


'''
    Processing class reserved for image analysis tasks:
    - segmentation, morphology, restauration, etc.
'''
class Processing:
    def __init__(self):
        pass
    
    @staticmethod
    def cannyEdgeDetection(img, lowThreshold=100, highThreshold=200, apertureSize=3, L2gradient=False):
        return cv2.Canny(img, lowThreshold, highThreshold, apertureSize=apertureSize, L2gradient=L2gradient)