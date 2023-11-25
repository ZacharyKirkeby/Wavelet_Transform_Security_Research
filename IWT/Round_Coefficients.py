import numpy as np
import pywt
import cv2
import matplotlib.pyplot as plt

f_path = "C:\\Users\zakir\Downloads\\Peppers.png"  # put your file path here
wavelet = 'haar'
mode = 'symmetric'
level = 1
dim1 = 512
dim2 = 512
# original was 512 x 512
print("Mode:", mode, "Wavelet:", wavelet, "Level:", level, "Dimensions:", dim1, "x", dim2, "Greyscale")


# TODO -> make more space
# make more 8x8 blocks with lots of zeros
# find floats


def preprocess_image(image_path, dim1, dim2):
    cover_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_resized = cv2.resize(cover_image, (dim1, dim2))

    return image_resized


f = preprocess_image(f_path, dim1, dim2)

plt.imshow(f, cmap='gray')
plt.title("Base Image")
plt.axis('off')
plt.show()

print("Pixel Count: ", dim1 * dim2)
print("Original Number of Zeros: ", np.count_nonzero(f == 0))


def apply_wavelet_transform(image, wavelet, level):
    # Apply 2D discrete wavelet transform
    coefficients = pywt.wavedec2(image, wavelet, level=level)

    return coefficients


newDim = 512
for x in range(0, level):
    newDim = newDim / 2

print("IWT band size: ", newDim)
print("After IWT")

cover_coeffs = apply_wavelet_transform(f, wavelet, level)
LL, LH, HH = cover_coeffs[0], cover_coeffs[1][0], cover_coeffs[1][1]

block_size = 8
rows, cols = 256, 256

PreRoundnum_zeros_LL_blocks = []
PreRoundnum_zeros_LH_blocks = []
PreRoundnum_zeros_HH_blocks = []
PreRoundaverage_per_cube = 0
PreRoundtotal_zeros_per_block = []

for r in range(0, rows, block_size):
    for c in range(0, cols, block_size):
        block_LL = LL[r:r + block_size, c:c + block_size]
        block_LH = LH[r:r + block_size, c:c + block_size]
        block_HH = HH[r:r + block_size, c:c + block_size]

        num_zeros_LL = np.count_nonzero(block_LL == 0)
        num_zeros_LH = np.count_nonzero(block_LH == 0)
        num_zeros_HH = np.count_nonzero(block_HH == 0)

        PreRoundnum_zeros_LL_blocks.append(num_zeros_LL)
        PreRoundnum_zeros_LH_blocks.append(num_zeros_LH)
        PreRoundnum_zeros_HH_blocks.append(num_zeros_HH)

        total_zeros = num_zeros_LL + num_zeros_LH + num_zeros_HH
        PreRoundtotal_zeros_per_block.append(total_zeros)

print("Average Zeros per 8x8", np.mean(PreRoundtotal_zeros_per_block))

plt.scatter(range(len(PreRoundnum_zeros_LL_blocks)), PreRoundnum_zeros_LL_blocks, label='LL', alpha=.5)
plt.scatter(range(len(PreRoundnum_zeros_LH_blocks)), PreRoundnum_zeros_LH_blocks, label='LH', alpha=0.5)
plt.scatter(range(len(PreRoundnum_zeros_HH_blocks)), PreRoundnum_zeros_HH_blocks, label='HH', alpha=0.5)

plt.xlabel('Block Index')
plt.ylabel('Number of Zeros')
plt.title('Occurrences of Zeros')
plt.legend()
plt.xlim(0, 500)

plt.show()

plt.scatter(range(len(PreRoundtotal_zeros_per_block)), PreRoundtotal_zeros_per_block)

plt.xlabel('Block Index')
plt.ylabel('Total Number of Zeros')
plt.title('Total Zeros per 8x8 Block')
plt.xlim(0, 500)
plt.show()

LL_zeros = np.count_nonzero(LL == 0)
LH_zeros = np.count_nonzero(LH == 0)
HH_zeros = np.count_nonzero(HH == 0)
print("baseline total zeros: ", LL_zeros, LH_zeros, HH_zeros)

min_zeros = min(LL_zeros, LH_zeros, HH_zeros)
max_zeros = max(LL_zeros, LH_zeros, HH_zeros)

print("Minimum number of zeros before rounding:", min_zeros)
print("Maximum number of zeros before rounding:", max_zeros)


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
LL_Roundedzeros = np.count_nonzero(RoundedLL == 0)
LH_Roundedzeros = np.count_nonzero(RoundedLH == 0)
HH_Roundedzeros = np.count_nonzero(RoundedHH == 0)

print("Rounded total zeros")
print(LL_Roundedzeros, LH_Roundedzeros, HH_Roundedzeros)

print("New zeros created")
print((LL_Roundedzeros - LL_zeros), (LH_Roundedzeros - LH_zeros), (HH_Roundedzeros - HH_zeros))

num_zeros_LL_blocks = []
num_zeros_LH_blocks = []
num_zeros_HH_blocks = []
average_per_cube = 0
total_zeros_per_block = []

for r in range(0, rows, block_size):
    for c in range(0, cols, block_size):
        block_LL = LL[r:r + block_size, c:c + block_size]
        block_LH = LH[r:r + block_size, c:c + block_size]
        block_HH = HH[r:r + block_size, c:c + block_size]

        num_zeros_LL = np.count_nonzero(block_LL == 0)
        num_zeros_LH = np.count_nonzero(block_LH == 0)
        num_zeros_HH = np.count_nonzero(block_HH == 0)

        num_zeros_LL_blocks.append(num_zeros_LL)
        num_zeros_LH_blocks.append(num_zeros_LH)
        num_zeros_HH_blocks.append(num_zeros_HH)

        total_zeros = num_zeros_LL + num_zeros_LH + num_zeros_HH
        total_zeros_per_block.append(total_zeros)

average_zeros_LL = np.mean(num_zeros_LL_blocks)
average_zeros_LH = np.mean(num_zeros_LH_blocks)
average_zeros_HH = np.mean(num_zeros_HH_blocks)

print("Average number of zeros in LL blocks:", average_zeros_LL)
print("Average number of zeros in LH blocks:", average_zeros_LH)
print("Average number of zeros in HH blocks:", average_zeros_HH)


plt.hist(total_zeros_per_block, bins=32, color="blue", edgecolor='black')
plt.xlabel("Number of zeros")
plt.ylabel("frequency")
plt.title("Histogram of Zeros per 8x8 Block")
plt.show()

total_minus_zero = []
for val in total_zeros_per_block:
    if val != 0:
        total_minus_zero.append(val)


print(type(total_minus_zero))
plt.hist(total_minus_zero, bins=32, color="blue", edgecolor='black')
plt.xlabel("Number of zeros")
plt.ylabel("frequency")
plt.title("Histogram of Zeros per 8x8 Block")
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

flattened = []
for val in range(len(cover_coeffs)):
    for vals in range(len(cover_coeffs[0])):
        for tup in cover_coeffs[val]:
            flattened.extend(tup)

a = np.ravel(RoundedLH)
b = np.ravel(RoundedHH)
c = np.ravel(RoundedLL)
# square root rule says 64 bins
plt.hist(a, bins=64, color='blue', label="LH", alpha=0.5)
plt.hist(b, bins=64, color='red', label="HH", alpha=0.5)
plt.hist(c, bins=64, color='green', label="LL", alpha=0.5)
plt.legend()
plt.xlabel("Coefficient")
plt.ylabel("Coefficient frequency")
plt.title("coefficients")
plt.xlim()
plt.show()
