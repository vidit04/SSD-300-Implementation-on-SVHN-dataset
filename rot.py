import numpy as np
import cv2
import random
import multiprocessing as mp

#width = 500
#height = 375

#image_input = 'C:/Users/user/Desktop/new/tensorflow-without-a-phd-master/tensorflow-mnist-tutorial/SVHN/' + '000005.jpg'
#image = cv2.imread (image_input)
#corners = gt_array[:,1:6]
def rot_corner(M1,corner):

    corner = np.reshape(corner,(1,4))

    x1_corner = corner[0,0]
    y1_corner = corner[0,1]
    x2_corner = corner[0,2]
    y2_corner = corner[0,1]
    x3_corner = corner[0,2]
    y3_corner = corner[0,3]
    x4_corner = corner[0,0]
    y4_corner = corner[0,3]

    corner_array = np.ones((4,3))
    new_corners = np.zeros((1,4))

    corner_array[0,0] = x1_corner
    corner_array[0,1] = y1_corner
    corner_array[1,0] = x2_corner
    corner_array[1,1] = y2_corner
    corner_array[2,0] = x3_corner
    corner_array[2,1] = y3_corner
    corner_array[3,0] = x4_corner
    corner_array[3,1] = y4_corner

    new_calculated_box = np.dot(M1,corner_array.T).T

    x_coor1 = new_calculated_box[:,0]
    y_coor1 = new_calculated_box[:,1]
    x_coor1 = np.reshape(x_coor1,(-1,4))
    y_coor1 = np.reshape(y_coor1,(-1,4))

    xmin_new = np.min(x_coor1,1).reshape(-1,1)
    ymin_new = np.min(y_coor1,1).reshape(-1,1)
    xmax_new = np.max(x_coor1,1).reshape(-1,1)
    ymax_new = np.max(y_coor1,1).reshape(-1,1)

    new_corners[0,0] = xmin_new
    new_corners[0,1] = ymin_new
    new_corners[0,2] = xmax_new
    new_corners[0,3] = ymax_new

    return new_corners


def rotation(image,gt_array,par_screw):


    img_height = image.shape[0]
    img_width = image.shape[1]
    for i in range(len(gt_array)):
        gt_array[i,1] = gt_array[i,1]*img_width
        gt_array[i,2] = gt_array[i,2]*img_height
        gt_array[i,3] = gt_array[i,3]*img_width
        gt_array[i,4] = gt_array[i,4]*img_height

    if par_screw ==1:
        angle = np.random.randint(10,40,dtype=np.int32)
        prob_choice1= random.uniform(0, 1)
        if prob_choice1 > 0.5:
            angle = np.random.randint(190,220,dtype=np.int32)

    if par_screw ==-1:
        angle = np.random.randint(-40,-10,dtype=np.int32)
        prob_choice2= random.uniform(0, 1)
        if prob_choice2 > 0.5:
            angle = np.random.randint(140,170,dtype=np.int32)
        
    #angle = np.random.randint(-15,15,dtype=np.int32)
    #prob_choice= random.uniform(0, 1)
    #if prob_choice > 0.5:
    #    angle = np.random.randint(165,195,dtype=np.int32)
    

    M = cv2.getRotationMatrix2D((int(img_width / 2), int(img_height / 2)), angle, 1)

    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    new_width = img_height*sin + img_width*cos
    new_height = img_height*cos + img_width*sin

    M[0, 2] += (new_width / 2) - (img_width/2)
    M[1, 2] += (new_height / 2) - (img_height/2)

    rgb_rot = cv2.warpAffine(image, M, (int(new_width), int(new_height)))

    #scale_factor_x = rgb_rot.shape[1]/ img_width
    #scale_factor_y = rgb_rot.shape[0]/ img_height
    rgb_rot = cv2.resize(rgb_rot, (int(new_width),int(new_height)))

    for i in range(len(gt_array)):
        rot_coor = rot_corner(M,gt_array[i,1:5])
        gt_array[i,1:5] = rot_coor
        #gt_array[i,1] = gt_array[i,1]*img_width
        #gt_array[i,2] = gt_array[i,2]*img_height
        #gt_array[i,3] = gt_array[i,3]*img_width
        #gt_array[i,4] = gt_array[i,4]*img_height
        #cv2.rectangle(rgb_rot,(int(gt_array[i,1]),int(gt_array[i,2])),(int(gt_array[i,3]),int(gt_array[i,4])),(0,0,0),2)

    for i in range(len(gt_array)):
        gt_array[i,1] = gt_array[i,1]/int(new_width)
        gt_array[i,2] = gt_array[i,2]/int(new_height)
        gt_array[i,3] = gt_array[i,3]/int(new_width)
        gt_array[i,4] = gt_array[i,4]/int(new_height)

    return rgb_rot, gt_array

#xmin = 165
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

    rgb_rot1 , gt_array =rotation(image,array)
    height = rgb_rot1.shape[0]
    width = rgb_rot1.shape[1]
    for i in range(len(gt_array)):
        cv2.rectangle(rgb_rot1,(int(gt_array[i,1]*width),int(gt_array[i,2]*height)),(int(gt_array[i,3]*width),int(gt_array[i,4]*height)),(0,255,255),2)
    cv2.imwrite('result_rot.png', rgb_rot1)
