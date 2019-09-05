import numpy as np
import cv2
import random
import multiprocessing as mp
from rot import rotation
from screw import screw

def randomplacement(image,gt_array):

    width = image.shape[1]
    #print(width)
    height = image.shape[0]
    #print(height)

    if (width>=300 or height >=300):
        c = max(width,height)
        #print(c)
        if  c==width:
            ratio = width/300
            #print(ratio)
            rgb = cv2.resize(image, (300,int(height/ratio)))
            #print(rgb.shape)
            p = np.random.randint(0,(300-int(height/ratio)+1),dtype=np.int32)
            #print(p)
            rgb= cv2.copyMakeBorder(rgb,p,(300-(int(height/ratio)+p)),0,0,cv2.BORDER_CONSTANT,value=(0,0,0))
            for i in range(len(gt_array)):
                gt_array[i,2] = gt_array[i,2]* int(height/ratio) + p
                gt_array[i,2] = gt_array[i,2]/300
                gt_array[i,4] = gt_array[i,4]* int(height/ratio) + p
                gt_array[i,4] = gt_array[i,4]/300
        if c==height:
            ratio = height/300
            rgb = cv2.resize(image, (int(width/ratio),300))
            p = np.random.randint(0,(300-int(width/ratio)+1),dtype=np.int32)
            rgb= cv2.copyMakeBorder(rgb,0,0,p,(300-(p+int(width/ratio))),cv2.BORDER_CONSTANT,value=(0,0,0))
            for i in range(len(gt_array)):
                gt_array[i,1] = gt_array[i,1]*int(width/ratio) + p
                gt_array[i,1] = gt_array[i,1]/300
                gt_array[i,3] = gt_array[i,3]*int(width/ratio) + p
                gt_array[i,3] = gt_array[i,3]/300
            #print(5)

    if (width < 300 and height < 300):
        prob= random.uniform(0, 1)
        if prob > 0:
            image = zoom(image)
            #print('Use zoom')
        width = image.shape[1]
        #print(width)
        height = image.shape[0]
        #print(height)
        start_x = np.random.randint(0,300-width+1,dtype=np.int32)
        #print(start_x)
        start_y = np.random.randint(0,300-height+1,dtype=np.int32)
        #print(start_y)
        rgb= cv2.copyMakeBorder(image,start_y,(300-(start_y+height)),start_x,(300-(start_x+width)),cv2.BORDER_CONSTANT,value=(0,0,0))
        for i in range(len(gt_array)):
            gt_array[i,1] = gt_array[i,1]*width + start_x
            gt_array[i,2] = gt_array[i,2]*height + start_y
            gt_array[i,3] = gt_array[i,3]*width + start_x
            gt_array[i,4] = gt_array[i,4]*height + start_y

            gt_array[i,1] = gt_array[i,1]/300
            gt_array[i,2] = gt_array[i,2]/300
            gt_array[i,3] = gt_array[i,3]/300
            gt_array[i,4] = gt_array[i,4]/300


    return rgb,gt_array

def zoom(image):
    im_height = image.shape[0]
    im_width = image.shape[1]
    d = max(im_height,im_width)
    e = 300-d
    f = np.random.randint(0,e,dtype=np.int32)
    if d== im_width:
        new_width = im_width+f
        ratio_zoom = im_width/new_width
        zoom_image = cv2.resize(image, (new_width,int(im_height/ratio_zoom)))
        d=0
    if d== im_height:
        new_height = im_height +f
        ratio_zoom = im_height/new_height
        zoom_image = cv2.resize(image, (int(im_width/ratio_zoom), new_height))
        d=0
    return zoom_image
        
        
#(300-(p+(height-(width-300))))
if __name__ == "__main__":
    width = 500
    height = 375
    image_input = 'C:/Users/user/Desktop/new/tensorflow-without-a-phd-master/tensorflow-mnist-tutorial/SVHN/' + '000005.jpg'
    #image_input = 'C:/Users/user/Desktop/' + 'robot1.png'

    image = cv2.imread(image_input)
    array = np.zeros((5,6))

    array[0,0] = 1
    array[0,1] = 263/width
    array[0,2] = 211/height
    array[0,3] = 324/width
    array[0,4] = 339/height
    array[0,5] = 1

    array[1,0] = 1
    array[1,1] = 165/width
    array[1,2] = 264/height
    array[1,3] = 253/width
    array[1,4] = 372/height
    array[1,5] = 1

    array[2,0] = 1
    array[2,1] = 5/width
    array[2,2] = 244/height
    array[2,3] = 67/width
    array[2,4] = 374/height
    array[2,5] = 1

    array[3,0] = 1
    array[3,1] = 241/width
    array[3,2] = 194/height
    array[3,3] = 295/width
    array[3,4] = 299/height
    array[3,5] = 1

    array[4,0] = 1
    array[4,1] = 277/width
    array[4,2] = 186/height
    array[4,3] = 312/width
    array[4,4] = 220/height
    array[4,5] = 1
    
#    array = np.zeros((4,6))
#    array[0,0] = 1
#    array[0,1] = 1/width
#    array[0,2] = 235/height
#    array[0,3] = 182/width
#    array[0,4] = 388/height
#    array[0,5] = 1
#
#    array[1,0] = 1
#    array[1,1] = 210/width
#    array[1,2] = 36/height
#    array[1,3] = 336/width
#    array[1,4] = 482/height
#    array[1,5] = 1

#    array[2,0] = 1
#    array[2,1] = 46/width
#    array[2,2] = 82/height
#    array[2,3] = 170/width
#    array[2,4] = 365/height
#    array[2,5] = 1

#    array[3,0] = 1
#    array[3,1] = 11/width
#    array[3,2] = 181/height
#    array[3,3] = 142/width
#    array[3,4] = 419/height
#    array[3,5] = 1

    #array[4,0] = 1
    #array[4,1] = 277/width
    #array[4,2] = 186/height
    #array[4,3] = 312/width
    #array[4,4] = 220/height
    #array[4,5] = 1

    #ratio1 = 500/250
    #image = cv2.resize(image, (512,512))
    image,array = rotation(image,array)
    image,array = screw(image,array)
    rgb1,gt_array = randomplacement(image,array)
    print(10)
    #rgb_screw1,gt_array = rotation(rgb_screw1,gt_array)
    for i in range(len(gt_array)):
        cv2.rectangle(rgb1,(int(gt_array[i,1]*300),int(gt_array[i,2]*300)),(int(gt_array[i,3]*300),int(gt_array[i,4]*300)),(0,0,255),2)
    cv2.imwrite('result_placement.png', rgb1)    
