import matplotlib.pyplot as plt
import numpy as np
import math
import random
import pywt
from time import perf_counter
import cv2

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
    
    #print input.shape

    # initializing the variables
    #----------------------------------
    h = 0
    v = 0

    vmin = 0
    hmin = 0

    output = np.zeros((vmax, hmax))

    i = 0
    #----------------------------------

    while ((v < vmax) and (h < hmax)): 
        #print ('v:',v,', h:',h,', i:',i)       
        if ((h + v) % 2) == 0:                 # going up
            
            if (v == vmin):
                #print(1)
                
                output[v, h] = input[i]        # if we got to the first line

                if (h == hmax):
                    v = v + 1
                else:
                    h = h + 1                        

                i = i + 1

            elif ((h == hmax -1 ) and (v < vmax)):   # if we got to the last column
                #print(2)
                output[v, h] = input[i] 
                v = v + 1
                i = i + 1

            elif ((v > vmin) and (h < hmax -1 )):    # all other cases
                #print(3)
                output[v, h] = input[i] 
                v = v - 1
                h = h + 1
                i = i + 1

        
        else:                                    # going down

            if ((v == vmax -1) and (h <= hmax -1)):       # if we got to the last line
                #print(4)
                output[v, h] = input[i] 
                h = h + 1
                i = i + 1
        
            elif (h == hmin):                  # if we got to the first column
                #print(5)
                output[v, h] = input[i] 
                if (v == vmax -1):
                    h = h + 1
                else:
                    v = v + 1
                i = i + 1
                                
            elif((v < vmax -1) and (h > hmin)):     # all other cases
                output[v, h] = input[i] 
                v = v + 1
                h = h - 1
                i = i + 1




        if ((v == vmax-1) and (h == hmax-1)):          # bottom right element
            #print(7)            
            output[v, h] = input[i] 
            break


    return output

def zigzag(input):
    #initializing the variables
    #----------------------------------
    h = 0
    v = 0

    vmin = 0
    hmin = 0

    vmax = input.shape[0]
    hmax = input.shape[1]
    
    #print(vmax ,hmax )

    i = 0

    output = np.zeros(( vmax * hmax))
    #----------------------------------

    while ((v < vmax) and (h < hmax)):
        
        if ((h + v) % 2) == 0:                 # going up
            
            if (v == vmin):
                #print(1)
                output[i] = input[v, h]        # if we got to the first line

                if (h == hmax):
                    v = v + 1
                else:
                    h = h + 1                        

                i = i + 1

            elif ((h == hmax -1 ) and (v < vmax)):   # if we got to the last column
                #print(2)
                output[i] = input[v, h] 
                v = v + 1
                i = i + 1

            elif ((v > vmin) and (h < hmax -1 )):    # all other cases
                #print(3)
                output[i] = input[v, h] 
                v = v - 1
                h = h + 1
                i = i + 1

        
        else:                                    # going down

            if ((v == vmax -1) and (h <= hmax -1)):       # if we got to the last line
                #print(4)
                output[i] = input[v, h] 
                h = h + 1
                i = i + 1
        
            elif (h == hmin):                  # if we got to the first column
                #print(5)
                output[i] = input[v, h] 

                if (v == vmax -1):
                    h = h + 1
                else:
                    v = v + 1

                i = i + 1

            elif ((v < vmax -1) and (h > hmin)):     # all other cases
                #print(6)
                output[i] = input[v, h] 
                v = v + 1
                h = h - 1
                i = i + 1




        if ((v == vmax-1) and (h == hmax-1)):          # bottom right element
            #print(7)            
            output[i] = input[v, h] 
            break

    #print ('v:',v,', h:',h,', i:',i)
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
            block = img[rowInd : rowInd2, colInd : colInd2]
            DCT = cv2.dct(block)
            DCT_normalized = np.divide(DCT,QUANTIZATION_MAT).astype(int)  
            
            #selecting DCT values using compression
            reordered2 = zigzag(DCT)
            reordered = zigzag(DCT_normalized)
            reordered2 = np.array([0 if abs(reordered[i]) < 1 else reordered2[i] for i in range(len(reordered2))])   
            
            stream = get_run_length_encoding(reordered2)
            
            #before stats
            #coeff.append(max(abs(np.array(stream))))
            #string.append(np.copy(stream))
            #length.append(len(stream))

            #scaling the values
            append, extra = scale(stream)
            remainder.append(append)
            val = XOR(stream, seed, oldSeed)
            stream = val

            #stats
            coeff.append(max(abs(np.array(stream))))
            string.append(stream)
            length.append(len(stream) + extra)

            pixelBlocks.append(stream)
    return pixelBlocks, remainder

def unscale(val, remainder):
    for i in range(len(val)):
        counter = 0
        if len(remainder[i]) > 0:
            while(counter < len(remainder[i])):
                val[i] = (val[i] * (k)) + remainder[i][len(remainder[i]) - counter - 1]
                counter += 1

def idct(nbh, nbw, pixels, remainder, blockS, seed, oldSeed):
    img = np.zeros([nbh * blockS, nbw * blockS])
    for o in range(len(pixels)):
        val = unXOR(pixels[o], seed, oldSeed)
        pixels[o] = val
        unscale(pixels[o], remainder[o])

        #stats
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
            if(i+3 < len(pixels[o])):
                j = int(abs(pixels[o][i + 3]))
            if j == 0:
                k = k + 1
            else:
                k = k + j + 1
            i = i + 2
            if i >= len(pixels[o]):
                break
        array = np.reshape(array,(blockS, blockS))
        i = 0
        j = 0
        k = 0
        padded_img = np.zeros((blockS, blockS))
        while i < blockS:
            j = 0
            while j < blockS:        
                temp_stream = array[i:i + 8, j:j + 8]                
                block = inverse_zigzag(temp_stream.flatten(), blockS, blockS)                         
                padded_img[i:i + 8,j:j + 8] = cv2.idct(block)        
                j = j + 8        
            i = i + 8
        img[(rowInd * blockS):(rowInd + 1) * blockS, colInd * blockS:(colInd + 1) * blockS] = np.rint(padded_img)
    return img

def XOR(val, seed, oldSeed):
    XORMatrix = np.zeros(len(val))
    random.seed(seed)
    XORMatrix = np.array([elem + random.uniform(0,1) * (2 * k) for elem in 
                 XORMatrix]).astype(int)
    val = [elem + k for elem in val]
    val = val + XORMatrix
    val = val % (2 * k + 1)
    random.seed(oldSeed)
    return val

def unXOR(val, seed, oldSeed):
    XORMatrix = np.zeros(len(val))
    random.seed(seed)
    XORMatrix = np.array([elem + random.uniform(0,1) * (2 * k) for elem in 
                 XORMatrix]).astype(int)
    val = val - XORMatrix
    val = val % (2 * k + 1)
    val = [elem - k for elem in val]
    random.seed(oldSeed)
    return val
    
#seeds
oldSeed = 1
seed = 2

#start program timer
startTimer = perf_counter()

# defining block size
block_size = 8

#defining k restraint
k = 15

# Quantization Matrix 
QUANTIZATION_MAT = np.array([[16,11,10,16,24,40,51,61],[12,12,14,19,26,58,60,55],[14,13,16,24,40,57,69,56 ],[14,17,22,29,51,87,80,62],[18,22,37,56,68,109,103,77],[24,35,55,64,81,104,113,92],[49,64,78,87,103,121,120,101],[72,92,95,98,112,100,103,99]])
quality = 50
if quality >= 50:
    QUANTIZATION_MAT = QUANTIZATION_MAT * (100 - quality)/50
else:
    QUANTIZATION_MAT = QUANTIZATION_MAT * 50/quality
QUANTIZATION_MAT = QUANTIZATION_MAT.astype(int)  
QUANTIZATION_MAT[QUANTIZATION_MAT == 0] = 1
print("Level: "+str(quality))

# reading image in grayscale style
img = cv2.imread('original_cat.png', cv2.IMREAD_GRAYSCALE)


# get size of the image
[h , w] = img.shape

# No of blocks needed : Calculation

height = h
width = w
h = np.float32(h) 
w = np.float32(w) 

nbh = math.ceil(h/block_size)
nbh = np.int32(nbh)

nbw = math.ceil(w/block_size)
nbw = np.int32(nbw)

# height of padded image
H =  block_size * nbh

# width of padded image
W =  block_size * nbw

# create a numpy zero matrix with size of H,W
padded_img = np.zeros((H,W))

# or this other way here
padded_img[0:height,0:width] = img[0:height,0:width]

#statistics
coeff = []
length = []
string = []

nbh = 20
nbw = 20
height_padding = 50
width_padding = 50
cv2.imwrite('original.bmp', np.uint8(padded_img[height_padding * block_size:height_padding * block_size + nbh * block_size, width_padding * block_size:width_padding * block_size + nbw * block_size]))
pixels, remainder = dct(nbh, nbw, height_padding, width_padding, padded_img, block_size, seed, oldSeed)