import numpy as np
from scipy import signal
from PIL import Image, ImageOps

def NOT(a):
    if a:
        return 0
    else:
        return 255

def reverse(data):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data[i][j] = NOT(data[i][j])
    
    return data

def Roberts(data, Background = False):
    mask_v = np.array([[-1, 0], [0, 1]], dtype= int)
    mask_h = np.array([[0, -1], [1, 0]], dtype= int)

    v_img = signal.convolve2d(data, mask_v)
    h_img = signal.convolve2d(data, mask_h)
    edged_img = np.sqrt(np.square(h_img) + np.square(v_img))

    if Background:
        edged_img = reverse(edged_img)

    return edged_img

def Sobel(data, Background = False):
    mask_v = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype= int)
    mask_h = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype= int)

    v_img = signal.convolve2d(data, mask_v)
    h_img = signal.convolve2d(data, mask_h)
    edged_img = np.sqrt(np.square(h_img) + np.square(v_img))

    if Background:
        edged_img = reverse(edged_img)

    return edged_img

def Prewitt(data, Background = False):
    mask_v = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype= int)
    mask_h = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype= int)

    v_img = signal.convolve2d(data, mask_v)
    h_img = signal.convolve2d(data, mask_h)
    edged_img = np.sqrt(np.square(h_img) + np.square(v_img))

    if Background:
        edged_img = reverse(edged_img)

    return edged_img
