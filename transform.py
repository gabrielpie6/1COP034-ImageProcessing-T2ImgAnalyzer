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
    

    @staticmethod
    def laplacianFilter(img, ksize):
        return cv2.convertScaleAbs(cv2.Laplacian(img, cv2.CV_64F, ksize=ksize))

    @staticmethod
    def sobelFilter(img, dx, dy, ksize):
        return cv2.convertScaleAbs(cv2.Sobel(img, cv2.CV_64F, dx=dx, dy=dy, ksize=ksize))

    @staticmethod
    def frequencyDomain(img):
        imgDFT = np.fft.fftshift(np.fft.fft2(img))
        imgFreqNorm = np.log(1 + np.abs(imgDFT))
        imgFreqNorm = cv2.normalize(imgFreqNorm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return imgFreqNorm


    @staticmethod
    def bandPassMask(width, height, split, inner, outer, fadeIn, fadeOut, bandMethod='Pass'):
        d = np.sqrt((width/2)**2 + (height/2)**2)
        m = split * d

        a = (1 - outer) * m + outer * d
        b = inner * m

        # f = lambda t, k: 1.0 / (1 + np.exp(k * t))
        # g = lambda t: f(- t + b, fadeIn)
        # h = lambda t: f(  t - a, fadeOut)

        g = lambda t: sigmoid(  t - b, fadeIn)
        h = lambda t: sigmoid(- t + a, fadeOut)

        q = lambda t: g(t)*h(t)
        c = lambda x, y: q(np.sqrt(x**2 + y**2))
        if bandMethod == 'Pass':
            normalized = lambda x, y: c(x - width / 2, y - height / 2)
        elif bandMethod == 'Reject':
            normalized = lambda x, y: 1 - c(x - width / 2, y - height / 2)
        else:
            raise ValueError("bandMethod must be either 'Pass' or 'Reject'")

        ys, xs = np.indices((height, width))
        mask = normalized(xs, ys)

        return mask

    @staticmethod
    def bandPass(img, split, inner, outer, fadeIn, fadeOut, bandMethod='Pass'):
        width,  height  = img.shape[1], img.shape[0]

        mask = Transformation.bandPassMask(width, height, split, inner, outer, fadeIn, fadeOut, bandMethod)

        imgDFT = np.fft.fftshift(np.fft.fft2(img))
        res = np.abs(np.fft.ifft2(imgDFT * mask)).astype(np.uint8)

        return res
    



    # Binarization
    @staticmethod
    def binarizationOtsu(img):
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    @staticmethod
    def binarizationThreshold(img, threshold):
        _, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
        return binary

    @staticmethod
    def binarizationAdaptive(img, block_size, c):
        return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c)





def sigmoid(t, k):
    z = k*t
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    neg = ~pos
    # For z >= 0: sig = 1/(1 + exp(-z))
    out[pos] = 1.0/(1.0 + np.exp(-z[pos]))
    # For z <  0: sig = exp(z)/(1 + exp(z))
    exp_z = np.exp(z[neg])
    out[neg] = exp_z/(1.0 + exp_z)
    return out