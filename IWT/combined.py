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
    # print input.shape

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
        # print ('v:',v,', h:',h,', i:',i)
        if ((h + v) % 2) == 0:  # going up

            if (v == vmin):
                # print(1)

                output[v, h] = input[i]  # if we got to the first line

                if (h == hmax):
                    v = v + 1
                else:
                    h = h + 1

                i = i + 1

            elif ((h == hmax - 1) and (v < vmax)):  # if we got to the last column
                # print(2)
                output[v, h] = input[i]
                v = v + 1
                i = i + 1

            elif ((v > vmin) and (h < hmax - 1)):  # all other cases
                # print(3)
                output[v, h] = input[i]
                v = v - 1
                h = h + 1
                i = i + 1


        else:  # going down

            if ((v == vmax - 1) and (h <= hmax - 1)):  # if we got to the last line
                # print(4)
                output[v, h] = input[i]
                h = h + 1
                i = i + 1

            elif (h == hmin):  # if we got to the first column
                # print(5)
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
            # print(7)
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

    # print(vmax ,hmax )

    i = 0

    output = np.zeros((vmax * hmax))
    # ----------------------------------

    while ((v < vmax) and (h < hmax)):

        if ((h + v) % 2) == 0:  # going up

            if (v == vmin):
                # print(1)
                output[i] = input[v, h]  # if we got to the first line

                if (h == hmax):
                    v = v + 1
                else:
                    h = h + 1

                i = i + 1

            elif ((h == hmax - 1) and (v < vmax)):  # if we got to the last column
                # print(2)
                output[i] = input[v, h]
                v = v + 1
                i = i + 1

            elif ((v > vmin) and (h < hmax - 1)):  # all other cases
                # print(3)
                output[i] = input[v, h]
                v = v - 1
                h = h + 1
                i = i + 1


        else:  # going down

            if ((v == vmax - 1) and (h <= hmax - 1)):  # if we got to the last line
                # print(4)
                output[i] = input[v, h]
                h = h + 1
                i = i + 1

            elif (h == hmin):  # if we got to the first column
                # print(5)
                output[i] = input[v, h]

                if (v == vmax - 1):
                    h = h + 1
                else:
                    v = v + 1

                i = i + 1

            elif ((v < vmax - 1) and (h > hmin)):  # all other cases
                # print(6)
                output[i] = input[v, h]
                v = v + 1
                h = h - 1
                i = i + 1

        if ((v == vmax - 1) and (h == hmax - 1)):  # bottom right element
            # print(7)
            output[i] = input[v, h]
            break

    # print ('v:',v,', h:',h,', i:',i)
    return output


def scale(val):
    remainder = []
    extra = 0
    for i in range(len(val)):
        individual_rem = []
        while abs(val[i]) > k:
            individual_rem.append(val[i] % (k))
            val[i] = val[i] // (k)
            extra += 1
        remainder.append(individual_rem)
    return remainder, extra


def dct(nbh, nbw, height_pad, width_pad, img, blockS, seed, oldSeed):
    pixelBlocks = []
    remainder = []
    scalar = []
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
            append, extra = scale(stream)
            remainder.append(append)
            val = XOR(stream, seed, oldSeed)
            stream = val

            # stats
            coeff.append(max(abs(np.array(stream))))
            string.append(stream)
            length.append(len(stream) + extra)

            pixelBlocks.append(stream)
    return pixelBlocks, remainder


def unscale(val, remainder):
    for i in range(len(val)):
        counter = 0
        if len(remainder[i]) > 0:
            while (counter < len(remainder[i])):
                val[i] = (val[i] * (k)) + remainder[i][len(remainder[i]) - counter - 1]
                counter += 1


coeffI = []
stringI = []
lengthI = []
def idct(nbh, nbw, pixels, remainder, blockS, seed, oldSeed):
    img = np.zeros([nbh * blockS, nbw * blockS])
    for o in range(len(pixels)):
        val = unXOR(pixels[o], seed, oldSeed)
        pixels[o] = val
        unscale(pixels[o], remainder[o])

        # stats
        coeffI.append(max(abs(np.array(pixels[o]))))

        stringI.append(np.copy(pixels[o]))
        lengthI.append(len(pixels[o]))

        rowInd = int(o / nbw)
        colInd = (o % nbw)
        array = np.zeros(blockS * blockS).astype(int)
        k = 0
        i = 0
        j = 0
        while k < array.shape[0]:
            array[k] = pixels[o][i]
            if (i + 3 < len(pixels[o])):
                j = int(abs(pixels[o][i + 3]))
            if j == 0:
                k = k + 1
            else:
                k = k + j + 1
            i = i + 2
            if i >= len(pixels[o]):
                break
        array = np.reshape(array, (blockS, blockS))
        i = 0
        j = 0
        k = 0
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


def XOR(val, seed, oldSeed):
    XORMatrix = np.zeros(len(val))
    random.seed(seed)
    XORMatrix = np.array([elem + random.uniform(0, 1) * (2 * k) for elem in
                          XORMatrix]).astype(int)
    val = [elem + k for elem in val]
    val = val + XORMatrix
    val = val % (2 * k + 1)
    random.seed(oldSeed)
    return val


def unXOR(val, seed, oldSeed):
    XORMatrix = np.zeros(len(val))
    random.seed(seed)
    XORMatrix = np.array([elem + random.uniform(0, 1) * (2 * k) for elem in
                          XORMatrix]).astype(int)
    val = val - XORMatrix
    val = val % (2 * k + 1)
    val = [elem - k for elem in val]
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
cv2.imwrite('original.bmp', np.uint8(
    padded_img[height_padding * block_size:height_padding * block_size + nbh * block_size,
    width_padding * block_size:width_padding * block_size + nbw * block_size]))
pixels, remainder = dct(nbh, nbw, height_padding, width_padding, padded_img, block_size, seed, oldSeed)

#---------------------------------------------------------------------------------------------------------------------#

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

#print("Average Zeros per 8x8", np.mean(PreRoundtotal_zeros_per_block))

#plt.scatter(range(len(PreRoundnum_zeros_LL_blocks)), PreRoundnum_zeros_LL_blocks, label='LL', alpha=.5)
#plt.scatter(range(len(PreRoundnum_zeros_LH_blocks)), PreRoundnum_zeros_LH_blocks, label='LH', alpha=0.5)
#plt.scatter(range(len(PreRoundnum_zeros_HH_blocks)), PreRoundnum_zeros_HH_blocks, label='HH', alpha=0.5)

#plt.xlabel('Block Index')
#plt.ylabel('Number of Zeros')
#plt.title('Occurrences of Zeros')
#plt.legend()
#plt.xlim(0, 500)

#plt.show()

#plt.scatter(range(len(PreRoundtotal_zeros_per_block)), PreRoundtotal_zeros_per_block)

#plt.xlabel('Block Index')
#plt.ylabel('Total Number of Zeros')
#plt.title('Total Zeros per 8x8 Block')
#plt.xlim(0, 500)
#plt.show()

LL_zeros = np.count_nonzero(LL == 0)
LH_zeros = np.count_nonzero(LH == 0)
HH_zeros = np.count_nonzero(HH == 0)
#print("baseline total zeros: ", LL_zeros, LH_zeros, HH_zeros)

min_zeros = min(LL_zeros, LH_zeros, HH_zeros)
max_zeros = max(LL_zeros, LH_zeros, HH_zeros)

#print("Minimum number of zeros before rounding:", min_zeros)
#print("Maximum number of zeros before rounding:", max_zeros)

#-------------------------------------------------------------------------------------------------------------#
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

#print("Rounded total zeros")
print(LL_Roundedzeros, LH_Roundedzeros, HH_Roundedzeros)

#print("New zeros created")
#print((LL_Roundedzeros - LL_zeros), (LH_Roundedzeros - LH_zeros), (HH_Roundedzeros - HH_zeros))

# Overall image statistics abstracted over data from each 8x8

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
print(len(num_zeros_LH_blocks))

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

def sortPixel(pixels):
    sortedPixels = sorted(pixels, key=len, reverse=TRUE)
    return sortedPixels


sortedPixels = sortPixel(pixels)
print(sortedPixels[0])
positions = []
print(RoundedLH)
def embed(pixels, coeffs, zeroByBlock):
    cube_index = 0
    pixel_index = 0
    x_coord = 0
    y_coord = 0
    counter = 0

    while pixel_index < len(pixels):
        for i in range(0, len(coeffs), 8):
            if pixel_index >= len(pixels):
                break
            for j in range(0, len(coeffs), 8):
                zero = zeroByBlock[cube_index]
                cube_index += 1
                if pixel_index >= len(pixels):
                    break
                if pixel_index + 1 >= len(pixels) and len(pixels[pixel_index]) <= zero:
                    for x in range(len(pixels[pixel_index])):
                        coeffs[x_coord][y_coord] = pixels[pixel_index][x] - 15
                        counter += 1
                        x_coord += 1
                        if x_coord >= 255:
                            y_coord += 1
                            x_coord = 0
                    positions.append(cube_index - 1)
                elif len(pixels[pixel_index]) <= zero and zero > len(pixels[pixel_index + 1]):
                    for x in range(len(pixels[pixel_index])):
                        coeffs[x_coord][y_coord] = pixels[pixel_index][x] - 15
                        counter += 1
                        x_coord += 1
                        if x_coord >= 255:
                            y_coord += 1
                            x_coord = 0
                    positions.append(cube_index - 1)

                pixel_index += 1


    print(counter)



# TODO figure out indexing

embed(sortedPixels, RoundedLH, num_zeros_LH_blocks)
print(len(pixels))
print(len(positions))
print(positions)
print(num_zeros_LH_blocks)
def reconstruct_image(coeffs, wavelet):
    # Reconstruct the image from wavelet coefficients
    reconstructed_image = pywt.waverec2(coeffs, wavelet)
    return reconstructed_image

baseLine = reconstruct_image(cover_coeffs, wavelet)
plt.imshow(baseLine, cmap='gray')
plt.title("RoundedPreEmbed")
plt.axis('off')
plt.show()
print(np.count_nonzero(RoundedLH == 0))

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

# note how the zeros are handled
print("===============")
print(RoundedLH[0])
print(len(pixels[0]))

#flattened = []
#for val in range(len(cover_coeffs)):
#    for vals in range(len(cover_coeffs[0])):
#        for tup in cover_coeffs[val]:
#            flattened.extend(tup)

a = np.ravel(RoundedLH)
b = np.ravel(RoundedHH)
c = np.ravel(RoundedLL)
# square root rule says 64 bins
#plt.hist(a, bins=64, color='blue', label="LH", alpha=0.5)
#plt.hist(b, bins=64, color='red', label="HH", alpha=0.5)
#plt.hist(c, bins=64, color='green', label="LL", alpha=0.5)
#plt.legend()
#plt.xlabel("Coefficient")
#plt.ylabel("Coefficient frequency")
#plt.title("coefficients")
#plt.xlim()
#plt.show()

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
            if abs(x1 - x2) > threshold:
                return False
    return True

print(compare_arrays(RoundedLH, LH))

# TODO verify data is good

test = reconstruct_image(new_coeffs, wavelet)

plt.imshow(test, cmap='gray')
plt.title("ReConstructed")
plt.axis('off')
plt.show()

# TODO sorting 8x8 to 8x8, assign biuggest space reqs first
# change to a sort find
# biggest duland, sequential mine, biggest first, smallest last
# restrict to availible space must be larger than next smallest
