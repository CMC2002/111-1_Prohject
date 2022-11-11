from perlin_noise import PerlinNoise
import matplotlib.pyplot as plt

def perlin(img):
    noise = PerlinNoise(octaves= 2)
    mask = [[noise([i*0.01, j*0.01]) for j in range(img.shape[0])] for i in range(img.shape[1])]

    return mask

