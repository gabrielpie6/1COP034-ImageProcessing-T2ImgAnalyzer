import cv2
import numpy as np

class Transformation:
    def __init__(self):
        pass

    @staticmethod
    def BWConversion(img):
        return cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY), cv2.COLOR_GRAY2RGBA)