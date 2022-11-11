from perlin_noise import PerlinNoise
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def perlin():
    noise = PerlinNoise(octaves= 2)
    mask = [[noise([i*0.01, j*0.01]) for j in range(192)] for i in range(192)]
    mask = np.array(mask)

    return mask

