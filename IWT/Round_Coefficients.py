import numpy as np
import pywt
import cv2
import matplotlib.pyplot as plt

f_path = "C:\\Users\zakir\Downloads\\lena.png" #put your file path here

counter = 0
wavelet = 'haar'
mode = 'symmetric'
level = 3

dim1 = 512
dim2 = 512

def preprocess_image(image_path, dim1, dim2 ):
    cover_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_resized = cv2.resize(cover_image, (dim1, dim2))

    return image_resized


f = preprocess_image(f_path, dim1, dim2)

plt.imshow(f, cmap='gray')
plt.title("Base Image")
plt.axis('off')
plt.show()
def apply_wavelet_transform(image, wavelet, level):
    # Apply 2D discrete wavelet transform
    coefficients = pywt.wavedec2(image, wavelet, level=level)

    return coefficients

# talk to dylan
cover_coeffs = apply_wavelet_transform(f, wavelet, level)

def round_coefficients(coeffs, threshold=15):
    # Round coefficients to zero if their absolute value is below the threshold
    rounded = 0
    rounded_coeffs = [
        tuple(np.round(np.where(np.abs(coeff) < threshold, 0, coeff)))
        for coeff in coeffs
    ]
    rounded = sum(np.count_nonzero(np.abs(coeff) < threshold) for coeff in coeffs)
    print(rounded)

    return rounded_coeffs


cover_coeffs = round_coefficients(cover_coeffs)


def reconstruct_image(coeffs, wavelet):
    # Reconstruct the image from wavelet coefficients
    reconstructed_image = pywt.waverec2(coeffs, wavelet)
    return reconstructed_image

cover = reconstruct_image(cover_coeffs, wavelet)

plt.imshow(cover, cmap='gray')
plt.title("Rounded")
plt.axis('off')
plt.show()
