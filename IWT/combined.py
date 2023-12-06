import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import pylab as py
from scipy.special import erfinv, erf
from tkinter import *
import random
from time import perf_counter
import pickle
from PIL import Image
import cv2
import pywt
import statistics as sta


def get_run_length_encoding(image):
    i = 0
    skip = 0
    stream = []
    image = image.astype(int)
    while i < image.shape[0]:
        if image[i] != 0:
            stream.append(image[i])
            stream.append(skip)
            skip = 0
        else:
            skip = skip + 1
        i = i + 1
    return stream


def inverse_zigzag(input, vmax, hmax):
    # initializing the variables
    # ----------------------------------
    h = 0
    v = 0

    vmin = 0
    hmin = 0

    output = np.zeros((vmax, hmax))

    i = 0
    # ----------------------------------

    while ((v < vmax) and (h < hmax)):

        if ((h + v) % 2) == 0:  # going up

            if (v == vmin):

                output[v, h] = input[i]  # if we got to the first line

                if (h == hmax):
                    v = v + 1
                else:
                    h = h + 1

                i = i + 1

            elif ((h == hmax - 1) and (v < vmax)):  # if we got to the last column

                output[v, h] = input[i]
                v = v + 1
                i = i + 1

            elif ((v > vmin) and (h < hmax - 1)):  # all other cases

                output[v, h] = input[i]
                v = v - 1
                h = h + 1
                i = i + 1


        else:  # going down

            if ((v == vmax - 1) and (h <= hmax - 1)):  # if we got to the last line

                output[v, h] = input[i]
                h = h + 1
                i = i + 1

            elif (h == hmin):  # if we got to the first column

                output[v, h] = input[i]
                if (v == vmax - 1):
                    h = h + 1
                else:
                    v = v + 1
                i = i + 1

            elif ((v < vmax - 1) and (h > hmin)):  # all other cases
                output[v, h] = input[i]
                v = v + 1
                h = h - 1
                i = i + 1

        if ((v == vmax - 1) and (h == hmax - 1)):  # bottom right element

            output[v, h] = input[i]
            break

    return output


def zigzag(input):
    # initializing the variables
    # ----------------------------------
    h = 0
    v = 0

    vmin = 0
    hmin = 0

    vmax = input.shape[0]
    hmax = input.shape[1]

    i = 0

    output = np.zeros((vmax * hmax))
    # ----------------------------------

    while ((v < vmax) and (h < hmax)):

        if ((h + v) % 2) == 0:  # going up

            if (v == vmin):

                output[i] = input[v, h]  # if we got to the first line

                if (h == hmax):
                    v = v + 1
                else:
                    h = h + 1

                i = i + 1

            elif ((h == hmax - 1) and (v < vmax)):  # if we got to the last column

                output[i] = input[v, h]
                v = v + 1
                i = i + 1

            elif ((v > vmin) and (h < hmax - 1)):  # all other cases

                output[i] = input[v, h]
                v = v - 1
                h = h + 1
                i = i + 1


        else:  # going down

            if ((v == vmax - 1) and (h <= hmax - 1)):  # if we got to the last line

                output[i] = input[v, h]
                h = h + 1
                i = i + 1

            elif (h == hmin):  # if we got to the first column

                output[i] = input[v, h]

                if (v == vmax - 1):
                    h = h + 1
                else:
                    v = v + 1

                i = i + 1

            elif ((v < vmax - 1) and (h > hmin)):  # all other cases

                output[i] = input[v, h]
                v = v + 1
                h = h - 1
                i = i + 1

        if ((v == vmax - 1) and (h == hmax - 1)):  # bottom right element

            output[i] = input[v, h]
            break

    return output


def scale(val, scale):
    new_values = []

    for i in range(len(val) + 1):
        test = 0
        if i == 0:
            test = len(val)
        else:
            test = val[i - 1]
        temporary_rem = []
        while abs(test) > k - 2:
            remainder = (test % (k - 1))
            test = test // (k - 1)
            temporary_rem.append(test)
            temporary_rem.append(remainder)
        if len(temporary_rem) > 0:
            new_values.append((k - 2 + (len(temporary_rem) / 2)))
            new_values.append(temporary_rem[len(temporary_rem) - 2])
            for o in range(int(len(temporary_rem) / 2)):
                new_values.append(temporary_rem[len(temporary_rem) - 2 * (o + 1) + 1])
        else:
            new_values.append(test)

    return np.array(new_values)


def dct(nbh, nbw, height_pad, width_pad, img, blockS, seed, oldSeed):
    pixelBlocks = []
    for i in range(nbh):
        rowInd = i * blockS + height_pad * blockS
        rowInd2 = rowInd + blockS
        for j in range(nbw):
            colInd = j * blockS + width_pad * blockS
            colInd2 = colInd + blockS
            block = img[rowInd: rowInd2, colInd: colInd2]
            DCT = cv2.dct(block)
            DCT_normalized = np.divide(DCT, QUANTIZATION_MAT).astype(int)

            # selecting DCT values using compression
            reordered2 = zigzag(DCT)
            reordered = zigzag(DCT_normalized)
            reordered2 = np.array([0 if abs(reordered[i]) < 1 else reordered2[i] for i in range(len(reordered2))])

            stream = get_run_length_encoding(reordered2)

            # before stats
            # coeff.append(max(abs(np.array(stream))))
            # string.append(np.copy(stream))
            # length.append(len(stream))

            # scaling the values
            stream = scale(stream, 0)
            stream = XOR(stream, seed, oldSeed, 0)

            # stats
            coeff.append(max(abs(np.array(stream))))
            string.append(stream)
            length.append(len(stream))

            pixelBlocks.append(stream)
    return pixelBlocks


def unscale(val, scale):
    answer = []
    index = 0
    value = 0
    counter = 0
    if val[index] > (k - 2):
        revert_count = val[index] - (k - 2)
        index += 1
        value = val[index]
        while (counter < revert_count):
            index += 1
            value = (value * (k - 1)) + val[index]
            counter += 1
        index += 1
    else:
        value = val[index]
        index += 1
    for i in range(int(value)):
        counter = 0
        if val[index] > (k - 2):
            revert_count = val[index] - (k - 2)
            index += 1
            value = val[index]
            while (counter < revert_count):
                index += 1
                value = (value * (k - 1)) + val[index]
                counter += 1
            answer.append(value)
        else:
            answer.append(val[index])
        index += 1
    return answer


def idct(nbh, nbw, pixels, blockS, seed, oldSeed):
    img = np.zeros([nbh * blockS, nbw * blockS])
    for o in range(len(pixels)):
        stream = unXOR(pixels[o], seed, oldSeed, o)
        pixels[o] = stream
        stream = unscale(pixels[o], o)
        pixels[o] = stream

        # stats
        # coeffI.append(max(abs(np.array(pixels[o]))))
        # stringI.append(np.copy(pixels[o]))
        # lengthI.append(len(pixels[o]))

        rowInd = int(o / nbw)
        colInd = (o % nbw)
        array = np.zeros(blockS * blockS).astype(int)
        r = 0
        i = 0
        j = 0
        while r < array.shape[0]:
            array[r] = pixels[o][i]
            if (i + 3 < len(pixels[o])):
                j = int(abs(pixels[o][i + 3]))
            if j == 0:
                r = r + 1
            else:
                r = r + j + 1
            i = i + 2
            if i >= len(pixels[o]):
                break
        array = np.reshape(array, (blockS, blockS))
        i = 0
        j = 0
        r = 0
        padded_img = np.zeros((blockS, blockS))
        while i < blockS:
            j = 0
            while j < blockS:
                temp_stream = array[i:i + 8, j:j + 8]
                block = inverse_zigzag(temp_stream.flatten(), blockS, blockS)
                padded_img[i:i + 8, j:j + 8] = cv2.idct(block)
                j = j + 8
            i = i + 8
        img[(rowInd * blockS):(rowInd + 1) * blockS, colInd * blockS:(colInd + 1) * blockS] = np.rint(padded_img)
    return img


def XOR(val, seed, oldSeed, scale):
    XORMatrix = np.zeros(len(val))
    random.seed(seed)
    XORMatrix = np.array([elem + random.uniform(0, 1) * (2 * k - 2) for elem in
                          XORMatrix]).astype(int)
    val = [elem + (k - 2) for elem in val]

    val = val + XORMatrix

    val = val % (2 * k - 1)

    val = [elem - k for elem in val]

    random.seed(oldSeed)
    return val


def unXOR(val, seed, oldSeed, scale):
    XORMatrix = np.zeros(len(val))
    random.seed(seed)
    XORMatrix = np.array([elem + random.uniform(0, 1) * (2 * k - 2) for elem in
                          XORMatrix]).astype(int)
    val = [elem + k for elem in val]

    val = val - XORMatrix

    val = val % (2 * k - 1)

    val = [elem - (k - 2) for elem in val]

    random.seed(oldSeed)
    return val


# seeds
oldSeed = 1
seed = 2

# start program timer
startTimer = perf_counter()

# defining block size
block_size = 8

# defining k restraint
k = 15

# Quantization Matrix
QUANTIZATION_MAT = np.array(
    [[16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55], [14, 13, 16, 24, 40, 57, 69, 56],
     [14, 17, 22, 29, 51, 87, 80, 62], [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92],
     [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]])
quality = 50
if quality >= 50:
    QUANTIZATION_MAT = QUANTIZATION_MAT * (100 - quality) / 50
else:
    QUANTIZATION_MAT = QUANTIZATION_MAT * 50 / quality
QUANTIZATION_MAT = QUANTIZATION_MAT.astype(int)
QUANTIZATION_MAT[QUANTIZATION_MAT == 0] = 1
print("Level: " + str(quality))

# reading image in grayscale style
img = cv2.imread('C:\\Users\zakir\Downloads\\original_cat.png', cv2.IMREAD_GRAYSCALE)

# get size of the image
[h, w] = img.shape

# No of blocks needed : Calculation

height = h
width = w
h = np.float32(h)
w = np.float32(w)

nbh = math.ceil(h / block_size)
nbh = np.int32(nbh)

nbw = math.ceil(w / block_size)
nbw = np.int32(nbw)

# height of padded image
H = block_size * nbh

# width of padded image
W = block_size * nbw

# create a numpy zero matrix with size of H,W
padded_img = np.zeros((H, W))

# or this other way here
padded_img[0:height, 0:width] = img[0:height, 0:width]

# statistics
coeff = []
length = []
string = []

nbh = 20
nbw = 20
height_padding = 50
width_padding = 50
cv2.imwrite('C:\\Users\zakir\Downloads\\original.bmp', np.uint8(
    padded_img[height_padding * block_size:height_padding * block_size + nbh * block_size,
    width_padding * block_size:width_padding * block_size + nbw * block_size]))
pixels = dct(nbh, nbw, height_padding, width_padding, padded_img, block_size, seed, oldSeed)

# ---------------------------------------------------------------------------------------------------------------------#

f_path = "C:\\Users\zakir\Downloads\\Peppers.png"  # put your file path here
wavelet = 'haar'
mode = 'symmetric'
level = 1
dim1 = 512
dim2 = 512

print("Mode:", mode, "Wavelet:", wavelet, "Level:", level, "Dimensions:", dim1, "x", dim2, "Greyscale")


def preprocess_image(image_path, dim1, dim2):
    cover_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_resized = cv2.resize(cover_image, (dim1, dim2))
    newDim = dim1
    for x in range(0, level):
        newDim = newDim / 2
    print("IWT band size: ", newDim)

    return image_resized


f = preprocess_image(f_path, dim1, dim2)

plt.imshow(f, cmap='gray')
plt.title("Generic Image")
plt.axis('off')
plt.show()

print("Pixel Count: ", dim1 * dim2)


def apply_wavelet_transform(image, wavelet, level):
    # Apply 2D discrete wavelet transform
    coefficients = pywt.wavedec2(image, wavelet, level=level)

    return coefficients


print("#-------------------------------------------------------------#")

cover_coeffs = apply_wavelet_transform(f, wavelet, level)
LL, LH, HH = cover_coeffs[0], cover_coeffs[1][0], cover_coeffs[1][1]

block_size = 8
rows, cols = 256, 256
# Pre Rounding statistics, generally not needed

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

# print("Average Zeros per 8x8", np.mean(PreRoundtotal_zeros_per_block))

# plt.scatter(range(len(PreRoundnum_zeros_LL_blocks)), PreRoundnum_zeros_LL_blocks, label='LL', alpha=.5)
# plt.scatter(range(len(PreRoundnum_zeros_LH_blocks)), PreRoundnum_zeros_LH_blocks, label='LH', alpha=0.5)
# plt.scatter(range(len(PreRoundnum_zeros_HH_blocks)), PreRoundnum_zeros_HH_blocks, label='HH', alpha=0.5)

# plt.xlabel('Block Index')
# plt.ylabel('Number of Zeros')
# plt.title('Occurrences of Zeros')
# plt.legend()
# plt.xlim(0, 500)

# plt.show()

# plt.scatter(range(len(PreRoundtotal_zeros_per_block)), PreRoundtotal_zeros_per_block)

# plt.xlabel('Block Index')
# plt.ylabel('Total Number of Zeros')
# plt.title('Total Zeros per 8x8 Block')
# plt.xlim(0, 500)
# plt.show()

LL_zeros = np.count_nonzero(LL == 0)
LH_zeros = np.count_nonzero(LH == 0)
HH_zeros = np.count_nonzero(HH == 0)
# print("baseline total zeros: ", LL_zeros, LH_zeros, HH_zeros)

min_zeros = min(LL_zeros, LH_zeros, HH_zeros)
max_zeros = max(LL_zeros, LH_zeros, HH_zeros)


# print("Minimum number of zeros before rounding:", min_zeros)
# print("Maximum number of zeros before rounding:", max_zeros)

# -------------------------------------------------------------------------------------------------------------#
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

# print("Rounded total zeros")
print(LL_Roundedzeros, LH_Roundedzeros, HH_Roundedzeros)

# print("New zeros created")
# print((LL_Roundedzeros - LL_zeros), (LH_Roundedzeros - LH_zeros), (HH_Roundedzeros - HH_zeros))


# Overall image statistics abstracted over data from each 8x8

num_zeros_LL_blocks = []
num_zeros_LH_blocks = []
num_zeros_HH_blocks = []
average_per_cube = 0
total_zeros_per_block = []

for r in range(0, rows, block_size):
    for c in range(0, cols, block_size):
        block_LL = RoundedLL[r:r + block_size, c:c + block_size]
        block_LH = RoundedLH[r:r + block_size, c:c + block_size]
        block_HH = RoundedHH[r:r + block_size, c:c + block_size]

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
maxLL = max(num_zeros_LL_blocks)
minLL = min(num_zeros_LL_blocks)
medLL = np.median(num_zeros_LL_blocks)
modeLL = sta.mode(num_zeros_LL_blocks)
maxLH = max(num_zeros_LH_blocks)
minLH = min(num_zeros_LH_blocks)
medLH = np.median(num_zeros_LH_blocks)
modeLH = sta.mode(num_zeros_LH_blocks)
maxHH = max(num_zeros_HH_blocks)
minHH = min(num_zeros_HH_blocks)
medHH = np.median(num_zeros_HH_blocks)
modeHH = sta.mode(num_zeros_HH_blocks)

print("Average number of zeros in LL blocks:", average_zeros_LL, "Max:", maxLL, "min:", minLL, "median:", medLL, "mode:"
      , modeLL)
print("Average number of zeros in LH blocks:", average_zeros_LH, "Max:", maxLH, "min:", minLH, "median:", medLH, "mode:"
      , modeLH)
print("Average number of zeros in HH blocks:", average_zeros_HH, "Max:", maxHH, "min:", minHH, "median:", medHH, "mode:"
      , modeHH)

plt.hist(total_zeros_per_block, bins=32, color="blue", edgecolor='black')
plt.xlabel("Number of zeros")
plt.ylabel("frequency")
plt.title("Embed space per 8x8")
plt.show()

total_minus_zero = []
for val in total_zeros_per_block:
    if val != 0:
        total_minus_zero.append(val)
LH_minus_zero = []
for val in num_zeros_LH_blocks:
    if val != 0:
        total_minus_zero.append(val)

plt.hist(total_minus_zero, bins=32, color="blue", edgecolor='black')
plt.xlabel("Number of zeros")
plt.ylabel("frequency")
plt.title("Histogram of Zeros per 8x8 Block")
plt.show()

plt.hist(num_zeros_LH_blocks, bins=32, color="blue", edgecolor='black')
plt.xlabel("Number of zeros")
plt.ylabel("frequency")
plt.title("Histogram of Zeros per 8x8 Block LH band")
plt.show()

plt.hist(num_zeros_HH_blocks, bins=32, color="blue", edgecolor='black')
plt.xlabel("Number of zeros")
plt.ylabel("frequency")
plt.title("Histogram of Zeros per 8x8 Block HH band")
plt.show()


def sort_by_subarray_length(arrays):
    # Get array lengths
    array_lengths = np.array([len(arr) for arr in arrays])

    # Get indices that would sort the array_lengths in descending order
    sorted_indices = np.argsort(array_lengths)[::-1]

    # Use sorted indices to reorder the original arrays
    sorted_arrays = [arrays[i] for i in sorted_indices]

    # Return sorted arrays and corresponding indices
    return sorted_arrays, sorted_indices


sortedPixels, keys = sort_by_subarray_length(pixels)

# Display the result
# print("Sorted arrays:", sortedPixels)
# print("Corresponding indices:", keys)

# valsHisto = []
# for int in LH:
#    if int < 14 and int > -16:
#        valsHisto.append(int)


def flatten_8x8_cubes(input_array):
    dim1, dim2 = input_array.shape
    cubes = []

    # Iterate over the array in 8x8 chunks
    for i in range(0, dim1, 8):
        for j in range(0, dim2, 8):
            # Extract 8x8 cube
            cube = input_array[i:i + 8, j:j + 8]

            # Flatten the cube and append to the list
            cubes.append(cube.flatten())

    # Concatenate the flattened cubes to get the final flattened array
    flattened_array = np.concatenate(cubes)

    return flattened_array


tempLH = flatten_8x8_cubes(LH)


def embed(pixels, zeros, LH):
    pixel_index = 0
    zero_index = 0
    counter = 0
    positions = []

    for block in range(0, len(LH), 64):
        zero = zeros[zero_index]
        position = block

        if pixel_index < len(pixels):
            if len(pixels[pixel_index]) <= zero:
                LH[position] = len(pixels[pixel_index])
                position += 1
                pixel = 0
                for val in range(0, 64):
                    if 15 >= LH[position] >= -15 and pixel < len(pixels[pixel_index]):
                        # Update LH[position] with the pixel value
                        LH[position] = pixels[pixel_index][pixel]
                        counter += 1
                        pixel += 1
                    position += 1
                pixel_index += 1
                positions.append(zero_index)
            zero_index += 1
        elif pixel_index >= len(pixels):
            LH[position] = 0


    print(counter)
    return LH, positions


# first num = num numbers to embed
tempLH, positions = embed(sortedPixels, num_zeros_LH_blocks, tempLH)
print(len(positions))



def unflatten_8x8_cubes(flattened_array, shape):
    dim1, dim2 = shape
    cube_size = 8
    num_cubes_dim1 = dim1 // cube_size
    num_cubes_dim2 = dim2 // cube_size
    reshaped_array = flattened_array.reshape(num_cubes_dim1, num_cubes_dim2, cube_size, cube_size)
    original_array = np.zeros(shape)
    for i in range(num_cubes_dim1):
        for j in range(num_cubes_dim2):
            original_array[i * cube_size:(i + 1) * cube_size, j * cube_size:(j + 1) * cube_size] = reshaped_array[i, j]

    return original_array


RoundedLH = unflatten_8x8_cubes(tempLH, LH.shape)


def reconstruct_image(coeffs, wavelet):
    # Reconstruct the image from wavelet coefficients
    reconstructed_image = pywt.waverec2(coeffs, wavelet)
    return reconstructed_image


total_values = 0
for sub_array in pixels:
    total_values += len(sub_array)
print("Total number of values:", total_values)

cover_coeffs = (RoundedLL, (RoundedLH, None, RoundedHH))

cover = reconstruct_image(cover_coeffs, wavelet)
plt.imshow(cover, cmap='gray')
plt.title("RoundedEmbeded")
plt.axis('off')
plt.show()

a = np.ravel(RoundedLH)
b = np.ravel(RoundedHH)
c = np.ravel(RoundedLL)
# square root rule says 64 bins
# plt.hist(a, bins=64, color='blue', label="LH", alpha=0.5)
# plt.hist(b, bins=64, color='red', label="HH", alpha=0.5)
# plt.hist(c, bins=64, color='green', label="LL", alpha=0.5)
# plt.legend()
# plt.xlabel("Coefficient")
# plt.ylabel("Coefficient frequency")
# plt.title("coefficients")
# plt.xlim()
# plt.show()

# Testing Reversibility


new_coeffs = apply_wavelet_transform(cover, wavelet, level)
LL, LH, HH = new_coeffs[0], new_coeffs[1][0], new_coeffs[1][1]



def compare_arrays(original, embed, threshold=0.0001):
    if len(original) != len(embed):
        return False
    for a1, a2 in zip(original, embed):
        if len(a1) != len(a2):
            return False
        for x1, x2 in zip(a1, a2):
            if np.round(x1) - np.round(x2) > 0:
                return False
    return True


print(compare_arrays(RoundedLH, LH))

# TODO verify data is good

test = reconstruct_image(new_coeffs, wavelet)

plt.imshow(test, cmap='gray')
plt.title("ReConstructed")
plt.axis('off')
plt.show()

flatLH = flatten_8x8_cubes(LH)

def undo_embed(LH):
    embedded_pixels = []
    counter = 0

    for blocks in range(0, len(LH), 64):
        block_start = blocks + 1
        num_extract = 0
        position = block_start
        # Extract the number of embedded values from the first element of the block
        if block_start >= len(LH):
            break
        num_embedded_values = np.round(LH[blocks])
        # If there are embedded values, extract the subarray
        if num_embedded_values > 0:
            embedded_block = []
            for i in range(block_start, block_start + 64):
                if num_extract >= num_embedded_values:
                    break
                if -15 <= np.round(LH[i]) <= 15:
                    embedded_block.append(LH[position])  # Append to embedded_pixels
                    num_extract += 1
                    counter += 1
                position += 1  # Move this line inside the loop
            embedded_pixels.append(embedded_block)
    print("Num extracted", counter)
    return embedded_pixels





OrderPixel = undo_embed(flatLH)


check = []
sortedPixels, keys = sort_by_subarray_length(pixels)
for x in range(len(pixels)):
    check.append(np.max(pixels[x]))

print(np.max(check))


def unsort_by_subarray_length(sorted_arrays, sorted_indices):
    # Create a mapping from sorted indices to original indices
    original_indices = np.argsort(sorted_indices)

    # Use the mapping to unsort the sorted arrays
    unsorted_arrays = [sorted_arrays[i] for i in original_indices]

    return unsorted_arrays

OrderPixel = unsort_by_subarray_length(OrderPixel, keys)
print(len(OrderPixel))




# NOTES

# positions - goes left to right, each entry represents a cube
# keys is the original position for the pixels values prior to be resorted

# TODO reverse everything



# np.round(Order_pixels)
