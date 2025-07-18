import cv2
import numpy as np


'''
    Transformation class for image transformation operations.
'''
class Transformation:
    def __init__(self):
        pass

    @staticmethod
    def BWConversion(img):
        return cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY), cv2.COLOR_GRAY2RGBA)
    @staticmethod
    def RGBAtoGray(img):
        return cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    @staticmethod
    def flipHorizontal(img):
        return cv2.flip(img, 1)
    @staticmethod
    def flipVertical(img):
        return cv2.flip(img, 0)    


    #
    # BW BASED TRANSFORMATIONS
    #
    @staticmethod
    def histogramEqualization(img):
        return cv2.equalizeHist(img)
    
    @staticmethod
    def meanBlur(img, sx, sy):
        return cv2.blur(img, (sx, sy))
    @staticmethod
    def gaussianBlur(img, sx, sy, sigmaX = 0, sigmaY = 0):
        return cv2.GaussianBlur(img, (sx, sy), sigmaX=sigmaX, sigmaY=sigmaY)
    @staticmethod
    def medianBlur(img, ksize):
        return cv2.medianBlur(img, ksize)