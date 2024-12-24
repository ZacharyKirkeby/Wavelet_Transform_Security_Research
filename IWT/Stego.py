import numpy as np
from PIL import Image
import pywt
import cv2
import matplotlib.pyplot as plt

f_path = "<Image_1>"
m_path = "<Image_2>"

wavelet = 'haar'
mode = 'symmetric'
level = 3
embed_strength = 0.0001

dim1 = 512
dim2 = 512

def preprocess_image(image_path, dim1, dim2 ):
    cover_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_resized = cv2.resize(cover_image, (dim1, dim2))

    return image_resized


f = preprocess_image(f_path, dim1, dim2)
m = preprocess_image(m_path, dim1, dim2)

plt.imshow(f, cmap='gray')
plt.title("Cover Image")
plt.axis('off')
plt.show()
plt.imshow(m,cmap='gray')
plt.title("Secret")
plt.axis('off')
plt.show()

def apply_wavelet_transform(image, wavelet, level):
    # Apply 2D discrete wavelet transform
    coefficients = pywt.wavedec2(image, wavelet, level=level)

    return coefficients

# talk to dylan
cover_coeffs = apply_wavelet_transform(f, wavelet, level)
#secret_coeffs = apply_wavelet_transform(m, wavelet, level)
def round_coefficients(coeffs, threshold=4):
    # Round coefficients to zero if their absolute value is below the threshold
    rounded_coeffs = [
        tuple(np.round(np.where(np.abs(coeff) < threshold, 0, coeff)))
        for coeff in coeffs
    ]

    return rounded_coeffs
cover_coeffs = round_coefficients(cover_coeffs)

def embed_secret_information(cover_coeffs, secret_coeffs, embedding_strength):
    stego_coeffs = list(cover_coeffs)
    for i in range(1, len(cover_coeffs)):
        stego_coeffs[i] = tuple(
            np.array(cover_coeff) + embedding_strength * np.array(secret_coeff)
            for cover_coeff, secret_coeff in zip(cover_coeffs[i], secret_coeffs[i])
        )

    return stego_coeffs


steg_coeffs = embed_secret_information(cover_coeffs, secret_coeffs, embed_strength)

def reconstruct_image(coeffs, wavelet):
    # Reconstruct the image from wavelet coefficients
    reconstructed_image = pywt.waverec2(coeffs, wavelet)

    return reconstructed_image

cover = reconstruct_image(cover_coeffs, wavelet)
steggo_image = reconstruct_image(steg_coeffs, wavelet)

plt.imshow(cover, cmap='gray')
plt.title("Cover Image")
plt.axis('off')
plt.show()
