import numpy as np
from scipy import ndimage
from PIL import Image, ImageOps
import cv2
from scipy import signal

def enhance(data):
    # apply laplacian blur
    laplacian = cv2.Laplacian(data, cv2.CV_64F)

    # sobel x filter where dx=1 and dy=0
    sobelx = cv2.Sobel(data, cv2.CV_64F, 1, 0, ksize=7)

    # sobel y filter where dx=0 and dy=1
    sobely = cv2.Sobel(data, cv2.CV_64F, 0, 1, ksize=7)

    sobel = np.sqrt(np.square(sobelx) + np.square(sobely))

    return sobel

def weird(data):
    img_gray = data.astype(np.uint8)
    kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])
    image_sharp = cv2.filter2D(src=data, ddepth=-1, kernel=kernel)

    return image_sharp

def GaussianHPF(img):
    hpf = img - cv2.GaussianBlur(img, (21, 21), 3) + 63

    return hpf

