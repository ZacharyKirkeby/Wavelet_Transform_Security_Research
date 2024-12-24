import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

f_path = "<File_Path_Here>"  # put your file path here

dim1 = 512
dim2 = 512


def preprocess_image(image_path, dim1, dim2):
    cover_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_resized = cv2.resize(cover_image, (dim1, dim2))
    return image_resized


f = preprocess_image(f_path, dim1, dim2)

plt.imshow(f, cmap='gray')
plt.title("Base Image")
plt.axis('off')
plt.show()


def image_to_matrix(f_path):
    img = Image.open(f_path)
    width, height = img.size
    pixels = list(img.getdata())
    matrix = [pixels[i * width:(i + 1) * width] for i in range(height)]
    return matrix


matrix = image_to_matrix(f_path)


def plot_matrix(matrix):
    img_array = np.array(matrix)
    white_mask = np.all(img_array == (255, 255, 255), axis=-1)
    fig, ax = plt.subplots()
    ax.imshow(white_mask, cmap='gray', interpolation='none')
    ax.set_aspect('equal')
    plt.show()


plot_matrix(matrix)


def plot_matrix_with_overlay(matrix, block_size=8):
    img_array = np.array(matrix)
    white_mask = np.all(img_array == (255, 255, 255), axis=-1)
    fig, ax = plt.subplots()
    ax.imshow(white_mask, cmap='gray', interpolation='none')
    for i in range(0, img_array.shape[0], block_size):
        ax.axhline(i, color='red', linewidth=0.5)
    for j in range(0, img_array.shape[1], block_size):
        ax.axvline(j, color='red', linewidth=0.5)
    ax.set_aspect('equal')
    plt.show()


plot_matrix_with_overlay(matrix)
