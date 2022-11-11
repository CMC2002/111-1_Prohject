import numpy as np
from PIL import Image
import cv2 as cv
from skimage import exposure
from skimage.exposure import match_histograms

def his_equal(img):
    img_gray = img.astype(np.uint8)
    eq_img_array = cv.equalizeHist(img_gray)
    
    return eq_img_array

def his_mat(ref, img):
    img = img.astype(np.uint8)
    ref = ref.astype(np.uint8)

    matched = match_histograms(img, ref)

    return matched

