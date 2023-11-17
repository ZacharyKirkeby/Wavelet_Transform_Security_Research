import numpy as np
import pywt
import cv2
import matplotlib.pyplot as plt

f_path = "C:\\Users\zakir\Downloads\\lena.png" #put your file path here
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
    #print(sum(np.count_nonzero(np.abs(coeff) == 0) for coeff in coefficients))

    return coefficients


cover_coeffs = apply_wavelet_transform(f, wavelet, level)

LL, LH, HH = cover_coeffs[0], cover_coeffs[1][0], cover_coeffs[1][1]
LL_zeros = np.size(LL) - np.count_nonzero(LL)
LH_zeros = np.size(LH) - np.count_nonzero(LH)
HH_zeros = np.size(HH) - np.count_nonzero(HH)
print("baseline total zeros")
print(LL_zeros, LH_zeros, HH_zeros)

def round_coefficients(coeffs, threshold=15):
    # Round coefficients to zero if their absolute value is below the threshold
    rounded = 0
    rounded_coeffs = [
        tuple(np.round(np.where(np.abs(coeff) < threshold, 0, coeff)))
        for coeff in coeffs
    ]
    return rounded_coeffs


cover_coeffs = round_coefficients(cover_coeffs)
RoundedLL, RoundedLH, RoundedHH = np.array(cover_coeffs[0]), np.array(cover_coeffs[1][0]), np.array(cover_coeffs[1][1])
LL_Roundedzeros = np.size(RoundedLL) - np.count_nonzero(RoundedLL)
LH_Roundedzeros = np.size(RoundedLH) - np.count_nonzero(RoundedLH)
HH_Roundedzeros = np.size(RoundedHH) - np.count_nonzero(RoundedHH)

print("Rounded total zeros")
print(LL_Roundedzeros, LH_Roundedzeros, HH_Roundedzeros)

print("New zeros created")
print((LL_zeros-LL_Roundedzeros), (LH_Roundedzeros - LH_zeros), (HH_Roundedzeros - HH_zeros))

block_size = 8
rows, cols = f.shape

num_zeros_LL_blocks = []
num_zeros_LH_blocks = []
num_zeros_HH_blocks = []
average_per_cube = 0

for r in range(0, rows, block_size):
    for c in range(0, cols, block_size):
        block_LL = LL[r:r+block_size, c:c+block_size]
        block_LH = LH[r:r+block_size, c:c+block_size]
        block_HH = HH[r:r+block_size, c:c+block_size]

        num_zeros_LL = np.size(block_LL) - np.count_nonzero(block_LL)
        num_zeros_LH = np.size(block_LH) - np.count_nonzero(block_LH)
        num_zeros_HH = np.size(block_HH) - np.count_nonzero(block_HH)

        num_zeros_LL_blocks.append(num_zeros_LL)
        num_zeros_LH_blocks.append(num_zeros_LH)
        num_zeros_HH_blocks.append(num_zeros_HH)

plt.scatter(range(len(num_zeros_LL_blocks)), num_zeros_LL_blocks, label='LL', alpha=.5)
plt.scatter(range(len(num_zeros_LH_blocks)), num_zeros_LH_blocks, label='LH', alpha=0.5)
plt.scatter(range(len(num_zeros_HH_blocks)), num_zeros_HH_blocks, label='HH', alpha=0.5)

plt.xlabel('Block Index')
plt.ylabel('Number of Zeros')
plt.title('Occurrences of Zeros')
plt.legend()
plt.xlim(0, 500)

plt.show()




def reconstruct_image(coeffs, wavelet):
    # Reconstruct the image from wavelet coefficients
    reconstructed_image = pywt.waverec2(coeffs, wavelet)
    return reconstructed_image

cover = reconstruct_image(cover_coeffs, wavelet)

plt.imshow(cover, cmap='gray')
plt.title("Rounded")
plt.axis('off')
plt.show()
