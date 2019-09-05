import cv2
import numpy as np
import random
import os
from random import shuffle

#filename = '000005.jpg'
#oriimg = cv2.imread (filename)
#height, width, depth = oriimg.shape

#imgScale = W/width
#newX,newY = oriimg.shape[1]*imgScale, oriimg.shape[0]*imgScale
#newimg = cv2.resize(oriimg,(int(newX),int(newY)))
#img = cv2.resize(oriimg,(200,300),interpolation=cv2.INTER_CUBIC)

def resize(image,array):
    
### resize of Image
    ref_images =[cv2.INTER_CUBIC, cv2.INTER_AREA,cv2.INTER_LINEAR, cv2.INTER_NEAREST,cv2.INTER_LANCZOS4]
    im = random.randint(0,4)
    #oriimg = cv2.imread (image)
    #print(im)
    img = cv2.resize(image,(300,300),interpolation = ref_images[im])
    #cv2.imshow("Show by CV2",img)
    #cv2.waitKey(0)
    #cv2.imwrite("resizeimg.jpg",img)
    return img,array


#img[0,0]=[0 ,0, 0]
#print(img[0,0])

#ref_noise =['gauss','s&p','poisson','speckle']
#no = random.randint(0,3)
#print(ref_noise[0])

##### noise function

def s_p(image,array):
    row, col, cha = image.shape
    size = int(np.ceil(col*0.01))
    for i in range(row):
        j = np.random.randint(0,col, size=size , dtype=np.int32)
        image[i,j] = [0, 0, 0]
        k = np.random.randint(0,col, size=size, dtype=np.int32)
        image[i,k] = [255, 255, 255]
    return image,array
    

#img = s_p(img)
#cv2.imshow("Show by CV2",img)
#cv2.waitKey(0)


def brightness(image, array):
    factor = 32
    image = image.astype(np.float32)
    bright= random.randint(-factor,factor)
    row, col, cha = image.shape
    for i in range(row):
        for j in range(col):
            image[i,j] = image[i,j] + [bright, bright, bright]
            for k in range(3):
                if image[i,j,k] > 255:
                    image[i,j,k]= 255
                if image[i,j,k] < 0:
                    image[i,j,k] = 0
    image = image.astype(np.uint8)
    return image,array
            


def contrast(image,array):
    factor = 1.5
    image = image.astype(np.float32)
    cont= random.uniform(0.5,factor)
    row, col, cha = image.shape
    for i in range(row):
        for j in range(col):
            image[i,j] = [cont*image[i,j,0], cont*image[i,j,1], cont*image[i,j,2]]
            for k in range(3):
                if image[i,j,k] > 255:
                    image[i,j,k]= 255
                if image[i,j,k] < 0:
                    image[i,j,k] = 0
    image = image.astype(np.uint8)
    return image,array




def hue(image,array):
    factor = 18
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image = image.astype(np.float32)
    Hue= random.randint(-factor,factor)
    row, col, cha = image.shape
    for i in range(row):
        for j in range(col):
            image[i,j,0] = image[i,j,0] + Hue
            if image[i,j,0] > 179:
                image[i,j,0]= image[i,j,0]- 180
            if image[i,j,0] < 0:
                image[i,j,0] = image[i,j,0]+ 180
    image = image.astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    return image,array

def saturation(image,array):
    factor = 1.5
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image = image.astype(np.float32)
    satu= random.uniform(0.5,factor)
    row, col, cha = image.shape
    for i in range(row):
        for j in range(col):
            image[i,j,1] = satu*image[i,j,1]
            if image[i,j,1] > 255:
                image[i,j,1]= 255
            if image[i,j,1] < 0:
                image[i,j,1] = 0
    image = image.astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    return image,array

def flip_horizontal (image,array):

    image=cv2.flip(image, +1)
    for i in range(len(array)):
        
        array[i,1] = 1-array[i,1]
        array[i,3] = 1-array[i,3]

        dummy = array[i,1]
        array[i,1] = array[i,3]
        array[i,3] = dummy
    
    return image,array

def flip_vertical (image,array):

    image=cv2.flip(image, 0)
    for i in range(len(array)):
        
        array[i,2] = 1-array[i,2]
        array[i,4] = 1-array[i,4]

        dummy = array[i,2]
        array[i,2] = array[i,4]
        array[i,4] = dummy
    
    return image,array

def channel(image,array):
    row,col,cha = image.shape
    image1= np.zeros((row,col,cha))
    random_channel =list(range(3))
    #print(random_channel)
    shuffle(random_channel)
    #print(random_channel)
    for i in range(3):
        image1[:,:,i]=image[:,:,random_channel[i]]
    image1 = image1.astype(np.uint8)
    return image1,array

#img = resize(oriimg)
#img = brightness(img)
#img = contrast(img)
#img = hue(img)
#img = saturation(img)
#img = flip_horizontal(img)
#img = channel(img)

#cv2.imshow("Show by CV2",img)
#cv2.waitKey(0)






