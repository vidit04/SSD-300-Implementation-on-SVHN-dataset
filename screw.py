import numpy as np
import cv2
import random
import multiprocessing as mp

#width = 500
#height = 375

#image_input = 'C:/Users/user/Desktop/new/tensorflow-without-a-phd-master/tensorflow-mnist-tutorial/SVHN/' + '000005.jpg'
#image = cv2.imread (image_input)
#corners = gt_array[:,1:6]

def screw_corner(shear_factor1,corner):

    corner = np.reshape(corner,(1,4))

    corner[0,0] = corner[0,0] + corner[0,1] * shear_factor1
    #corner[0,1]
    corner[0,2] = corner[0,2] + corner[0,3] * shear_factor1
    #corner[0,3] = corner[0,3] + 
    
    corner[0,0] = corner[0,0]
    corner[0,1] = corner[0,1]
    corner[0,2] = corner[0,2]
    corner[0,3] = corner[0,3]

    return corner

def flip1(image1,gt_array1):
    #gt_array = np.reshape(gt_array,(len(gt_array),6))
    image2=cv2.flip(image1, +1)
    for i in range(len(gt_array1)):
        gt_array1[i,1] = image1.shape[1] - gt_array1[i,1]
        gt_array1[i,3] = image1.shape[1] - gt_array1[i,3]

        dummy = gt_array1[i,3]
        gt_array1[i,3] = gt_array1[i,1]
        gt_array1[i,1] = dummy
    
    return image2, gt_array1
    
def flip2(image3,gt_array2):
    #gt_array = np.reshape(gt_array,(len(gt_array),6))
    image4=cv2.flip(image3, +1)
    for i in range(len(gt_array2)):
        gt_array2[i,1] = image3.shape[1] - gt_array2[i,1]
        gt_array2[i,3] = image3.shape[1] - gt_array2[i,3]

        dummy = gt_array2[i,3]
        gt_array2[i,3] = gt_array2[i,1]
        gt_array2[i,1] = dummy

    return image4, gt_array2

def screw(image,gt_array):


    img_height = image.shape[0]
    img_width = image.shape[1]
    for i in range(len(gt_array)):
        gt_array[i,1] = gt_array[i,1]*img_width
        gt_array[i,2] = gt_array[i,2]*img_height
        gt_array[i,3] = gt_array[i,3]*img_width
        gt_array[i,4] = gt_array[i,4]*img_height

    shear_factor = np.random.randint(55,115,dtype=np.int32)/100
    par_screw = 1
    prob_choice  = random.uniform(0, 1)
    if prob_choice > 0.5:
        shear_factor = np.random.randint(-115,-55,dtype=np.int32)/100
        par_screw = -1

    
    #shear_factor = 2.0

    if shear_factor < 0:
        image,gt_array = flip1(image,gt_array)

    #for i in range(len(gt_array)):
    #    cv2.rectangle(image,(int(gt_array[i,1]),int(gt_array[i,2])),(int(gt_array[i,3]),int(gt_array[i,4])),(255,0,0),2)
    #cv2.imwrite('result_afterflip.png', image)

    M = np.array([[1, abs(shear_factor), 0],[0,1,0]])
    new_width = img_width + abs(shear_factor*img_height)

    #bboxes[:,[0,2]] += ((bboxes[:,[1,3]]) * abs(shear_factor) ).astype(int)

    rgb_screw = cv2.warpAffine(image, M, (int(new_width), img_height))
    #scale_factor_x = new_width / img_width

    rgb_screw = cv2.resize(rgb_screw, (int(new_width),img_height))
    #rgb_rot = cv2.resize(rgb_rot, (img_width,img_height))

    for i in range(len(gt_array)):
        rot_coor = screw_corner(abs(shear_factor),gt_array[i,1:5])
        gt_array[i,1:5] = rot_coor
        #gt_array[i,1] = gt_array[i,1]*img_width
        #gt_array[i,2] = gt_array[i,2]*img_height
        #gt_array[i,3] = gt_array[i,3]*img_width
        #gt_array[i,4] = gt_array[i,4]*img_height
        #cv2.rectangle(rgb_screw,(int(gt_array[i,1]),int(gt_array[i,2])),(int(gt_array[i,3]),int(gt_array[i,4])),(0,0,0),2)

    if shear_factor < 0:
        rgb_screw,gt_array  = flip2(rgb_screw,gt_array)

    #for i in range(len(gt_array)):
    #    cv2.rectangle(rgb_screw,(int(gt_array[i,1]),int(gt_array[i,2])),(int(gt_array[i,3]),int(gt_array[i,4])),(0,0,255),2)

    for i in range(len(gt_array)):
        gt_array[i,1] = gt_array[i,1]/int(new_width)
        gt_array[i,2] = gt_array[i,2]/img_height
        gt_array[i,3] = gt_array[i,3]/int(new_width)
        gt_array[i,4] = gt_array[i,4]/img_height

    return rgb_screw, gt_array, par_screw

#ymin = 264
#xmax = 253
#ymax = 372

#x1 = xmin
#y1 = ymin
#x2 = xmax
#y2 = ymin
#x3 = xmax
#y3 = ymax
#x4 = xmin
#y4 = ymax

#corner = np.ones((4,3))
#corner[0,0] = x1
#corner[0,1] = y1
#corner[1,0] = x2
#corner[1,1] = y2
#corner[2,0] = x3
#corner[2,1] = y3
#corner[3,0] = x4
#corner[3,1] = y4

#cv2.rectangle(image,(165,264),(253,372),(255,0,0),2)
#cv2.rectangle(image,(5,244),(67,374),(0,255,0),2)
#cv2.rectangle(image,(263,211),(324,339),(0,0,255),2)
#cv2.rectangle(image,(241,194),(295,299),(150,150,150),2)
#cv2.rectangle(image,(277,186),(312,220),(236,62,213),2)

#M = cv2.getRotationMatrix2D((500 / 2, 375 / 2), -45, 1)

#cos = np.abs(M[0, 0])
#sin = np.abs(M[0, 1])

#new_width = height*sin + width*cos
#new_height = height*cos + width*sin

#M[0, 2] += (new_width / 2) - (width/2)
#M[1, 2] += (new_height / 2) - (height/2)

#calculated_box = np.dot(M,corner.T).T

#x_coor = calculated_box[:,0]
#y_coor = calculated_box[:,1]
#x_coor = np.reshape(x_coor,(-1,4))
#y_coor = np.reshape(y_coor,(-1,4))

#xmin = np.min(x_coor,1).reshape(-1,1)
#ymin = np.min(y_coor,1).reshape(-1,1)
#xmax = np.max(x_coor,1).reshape(-1,1)
#ymax = np.max(y_coor,1).reshape(-1,1)

#rgb_rot = cv2.warpAffine(image, M, (int(new_width), int(new_height)))


#scale_factor_x = rgb_rot.shape[1]/ width
#scale_factor_y = rgb_rot.shape[0]/ height
#rgb_rot = cv2.resize(rgb_rot, (width,height))

#xmin = xmin/scale_factor_x
#ymin = ymin/scale_factor_y
#xmax = xmax/scale_factor_x
#ymax = ymax/scale_factor_y

#cv2.rectangle(rgb_rot,(int(xmin),int(ymin)),(int(xmax),int(ymax)),(0,0,0),2)
#cv2.imwrite('result_rot.png', rgb_rot)

if __name__ == "__main__":
    width = 500
    height = 375
    image_input = 'C:/Users/user/Desktop/new/tensorflow-without-a-phd-master/tensorflow-mnist-tutorial/SVHN/' + '000005.jpg'
    image = cv2.imread (image_input)
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
    
    rgb_screw1 , gt_array =screw(image,array)
    height = rgb_screw1.shape[0]
    width = rgb_screw1.shape[1]
    
    #rgb_screw1,gt_array = rotation(rgb_screw1,gt_array)
    for i in range(len(gt_array)):
        cv2.rectangle(rgb_screw1,(int(gt_array[i,1]*width),int(gt_array[i,2]*height)),(int(gt_array[i,3]*width),int(gt_array[i,4]*height)),(255,0,255),2)    
    cv2.imwrite('result_screw.png', rgb_screw1)
