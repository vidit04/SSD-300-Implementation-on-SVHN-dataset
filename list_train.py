import numpy as np
import cv2
import random
from math import log
import multiprocessing as mp
from randomplacement import randomplacement
from rot import rotation
from screw import screw
from Arg1 import s_p, resize,brightness,contrast,hue,saturation,flip_horizontal,flip_vertical,channel


def grid_cen (grid ,mini, maxi, aspect_ratio):

    if grid == 38:
        grid1 = 300/8
    if grid == 19:
        grid1 = 300/16
    if grid == 10:
        grid1 = 300/32
    if grid == 5:
        grid1  = 300/64
    if grid == 3:
        grid1 = 300/100
    if grid == 1:
        grid1 = 300/300

    fact = 1/(grid1)
    x_cen_anc = np.zeros((grid,grid))
    y_cen_anc = np.zeros((grid,grid))
    box =[]
    small_boxes = []
    big_boxes = []
    other_boxes1 = []
    other_boxes2 = []
    other_boxes3 = []
    other_boxes4 = []
    for i in range(grid):
        for j in range(grid):
            x_cen_anc[i,j] = (j + 0.5)*fact
            y_cen_anc[i,j]= (i + 0.5)*fact

    for i in range(grid):
        for j in range(grid):

            ####  Small box
            width_anc = mini/300
            height_anc = mini/300
            box =[x_cen_anc[i,j], y_cen_anc[i,j], width_anc, height_anc]
            small_boxes.append(box)

            #### Large box
            width_anc = np.sqrt(mini*maxi)/300
            height_anc = np.sqrt(mini*maxi)/300
            box =[x_cen_anc[i,j], y_cen_anc[i,j], width_anc, height_anc]
            big_boxes.append(box)

            ### other boxes with ratio
            for k in aspect_ratio:
                width_anc = (mini * np.sqrt(k))/300
                height_anc = (mini /np.sqrt(k))/300
                box =[x_cen_anc[i,j], y_cen_anc[i,j], width_anc, height_anc]
                if k==2:
                    other_boxes1.append(box)
                if k==3:
                    other_boxes3.append(box)

                width_anc = (mini / np.sqrt(k))/300
                height_anc = (mini * np.sqrt(k))/300
                box =[x_cen_anc[i,j], y_cen_anc[i,j], width_anc, height_anc]
                if k==2:
                    other_boxes2.append(box)
                if k==3:
                    other_boxes4.append(box)
                    
    return (small_boxes, big_boxes, other_boxes1,other_boxes2, other_boxes3, other_boxes4)




def label2id(label):
    if label == 1:
        x=1
    if label == 2:
        x=2
    if label == 3:
        x=3
    if label == 4:
        x=4
    if label == 5:
        x=5
    if label == 6:
        x=6
    if label == 7:
        x=7
    if label == 8:
        x=8
    if label == 9:
        x=6
    if label == 10:
        x=0
    return int(x)


def rect_overlap(ac, gt):
    ac_len = len(ac)
    gt_len = len(gt)
    #print(ac_len)
    #print(gt_len)
    overlap_per = np.zeros((ac_len,gt_len),dtype=np.float32)
    for j in range(gt_len):
        for i in range(ac_len):
            x_cen_gt=gt[j,0]
            y_cen_gt=gt[j,1]
            width_gt=gt[j,2]
            height_gt=gt[j,3]

            #print(x_cen_gt)
            #print(y_cen_gt)
            #print(width_gt)
            #print(height_gt)

            xmax_gt = x_cen_gt + width_gt/2
            xmin_gt = x_cen_gt - width_gt/2
            ymax_gt = y_cen_gt + height_gt/2
            ymin_gt = y_cen_gt - height_gt/2

            #print(xmin_gt)
            #print(ymin_gt)
            #print(xmax_gt)
            #print(ymax_gt)
            #print(ymin_gt)
            
            x_cen_ac=ac[i,0]
            y_cen_ac=ac[i,1]
            width_ac=ac[i,2]
            height_ac=ac[i,3]

            #print(x_cen_ac)
            #print(y_cen_ac)
            #print(width_ac)
            #print(height_ac)
            
            xmax_ac = x_cen_ac + width_ac/2
            xmin_ac = x_cen_ac - width_ac/2
            ymax_ac = y_cen_ac + height_ac/2
            ymin_ac = y_cen_ac - height_ac/2
            #print(xmax_ac)
            #print(xmin_ac)
            #print(ymax_ac)
            #print(ymin_ac)

            if (ymax_gt <= ymin_ac) or (ymin_gt >= ymax_ac) or (xmax_gt <= xmin_ac) or (xmax_ac <= xmin_gt):
                Area_I = 0
            else:
                Area_I = abs((max(xmin_gt,xmin_ac) - min(xmax_gt, xmax_ac))) * abs((max(ymin_gt,ymin_ac) - min(ymax_gt, ymax_ac)))
                overlap_per[i,j] = Area_I/(abs(xmin_gt - xmax_gt)*abs(ymin_gt - ymax_gt) + abs(xmin_ac - xmax_ac)*abs(ymin_ac - ymax_ac) - Area_I)
    return overlap_per




def label_data(anc_box_array,img_array1,cat):
    ol = rect_overlap(anc_box_array,img_array1[:,1:5])
    #print (ol)
    maxi_overlap_index = np.argmax(ol, axis=1)
    #print(maxi_overlap_index.shape)
    maxi_overlap_index = np.reshape(maxi_overlap_index,(len(maxi_overlap_index)))
    #print(maxi_overlap_index.shape)
    anc_box_array1 = np.zeros((anc_box_array.shape[0],anc_box_array.shape[1]))
    img_array2 =  np.zeros((img_array1.shape[0],img_array1.shape[1]))
    #print(maxi_overlap_index)
    #print(ol[1035,0])
    #print(ol[1169,1])
    #print(ol[1147,2])
    #print(ol[953,3])
    #print(ol)
    #print(anc_box_array)
    #print(img_array1)
    ther_hold = 0.5
    image_label_arr = np.zeros((len(ol),cat),dtype=np.int32)
    for i in range(len(ol)):
        image_label_arr[i,cat-1] = 1
    local_label_arr = np.zeros((len(ol),4), dtype=np.float32)
    overlap_bool = ol > ther_hold
    #print (overlap_bool.shape)

    ### for overlap greater than 0.5

    ### change to absolute
    
    for i in range(len(anc_box_array)):
        anc_box_array1[i,0] = 300 * anc_box_array[i,0]
        anc_box_array1[i,1] = 300 * anc_box_array[i,1]
        anc_box_array1[i,2] = 300 * anc_box_array[i,2]
        anc_box_array1[i,3] = 300 * anc_box_array[i,3]
    #print(anc_box_array1[i,0])

    for i in range(len(img_array1)):
        img_array2[i,0] = img_array1[i,0]
        img_array2[i,1] = 300 * img_array1[i,1]
        img_array2[i,2] = 300 * img_array1[i,2]
        img_array2[i,3] = 300 * img_array1[i,3]
        img_array2[i,4] = 300 * img_array1[i,4]
        img_array2[i,5] = img_array1[i,5]
        
        
    for j in range(len(overlap_bool[0])):
        for i in range(len(overlap_bool)):
            if overlap_bool[i,j] == True:
                #print(i)
                #print(j)
                image_label_arr[i,:] =  np.zeros((1,cat),dtype=np.int32)
                image_label_arr[i,int(img_array2[j,0])] = 1
                local_label_arr[i,0]= ((img_array2[j,1] - anc_box_array1[i,0])/anc_box_array1[i,2])*10
                local_label_arr[i,1]= ((img_array2[j,2] - anc_box_array1[i,1])/anc_box_array1[i,3])*10
                local_label_arr[i,2]= (log(img_array2[j,3]/anc_box_array1[i,2]))*5
                local_label_arr[i,3]= (log(img_array2[j,4]/anc_box_array1[i,3]))*5
    ### for maximum overlap
    
    for i in range(len(ol)):
        if overlap_bool[i,maxi_overlap_index[i]] == True:
            image_label_arr[i,:] = np.zeros((1,cat),dtype=np.int32)
            image_label_arr[i,int(img_array2[maxi_overlap_index[i],0])] = 1
            local_label_arr[i,0]= ((img_array2[maxi_overlap_index[i],1] - anc_box_array1[i,0])/anc_box_array1[i,2])*10
            local_label_arr[i,1]= ((img_array2[maxi_overlap_index[i],2] - anc_box_array1[i,1])/anc_box_array1[i,3])*10
            local_label_arr[i,2]= (log(img_array2[maxi_overlap_index[i],3]/anc_box_array1[i,2]))*5
            local_label_arr[i,3]= (log(img_array2[maxi_overlap_index[i],4]/anc_box_array1[i,3]))*5
    #print(image_label_arr)
    return (image_label_arr, local_label_arr)


def image_data_gen(q,list_pass,cat,a1,b1,c1,d1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3,a4,b4,c4,d4,e4,f4,a5,b5,c5,d5,a6,b6,c6,d6):

    feed_arr1 = np.ones((8732,cat), dtype=np.int32 )
    loc_feed_arr1 = np.ones((8732,4),dtype = np.float32)
    image_feed = np.ones((300,300,3),dtype = np.uint8)
    image = list_pass[0]
    image = cv2.imread (image)

    img_width = int(list_pass[1])
    img_height = int(list_pass[2])
    gt_array = np.zeros((len(list_pass[3]),6))
    for i in range(len(list_pass[3])):
        gt_array[i,0] = label2id(int(float(list_pass[3][i])))
        gt_array[i,1] = float(list_pass[4][i])/img_width
        gt_array[i,2] = float(list_pass[5][i])/img_height
        gt_array[i,3] = float(list_pass[6][i])/img_width
        gt_array[i,4] = float(list_pass[7][i])/img_height
        gt_array[i,5] = 0

    #### convertion
    for i in range(len(gt_array)):
        gt_xmin = gt_array[i,1]-(gt_array[i,3]/2)
        gt_ymin = gt_array[i,2]-(gt_array[i,4]/2)
        gt_xmax = gt_array[i,1]+(gt_array[i,3]/2)
        gt_ymax = gt_array[i,2]+(gt_array[i,4]/2)
        gt_array[i,1] = gt_xmin
        gt_array[i,2] = gt_ymin
        gt_array[i,3] = gt_xmax
        gt_array[i,4] = gt_ymax


    prob_org = random.uniform(0, 1)
    if prob_org > 0.2:
        ## Salt & Pepper
        prob_s_p = random.uniform(0, 1)
        if prob_s_p > 0.3:
            image,gt_array = s_p(image,gt_array)
            #print('salt_papper added')
        else:
            image = image
            gt_array = gt_array
            #print('salt_papper not added')

        ## Brightness
        prob_bri = random.uniform(0, 1)
        if prob_bri > 0.3:
            image,gt_array = brightness(image,gt_array)
            #print('brightness added')
        else:
            image = image
            gt_array = gt_array
            #print('brightness not added')

        ### Contrast
        prob_cont = random.uniform(0, 1)
        if prob_cont > 0.3:
            image,gt_array = contrast(image,gt_array)
            #print('Contrast added')
        else:
            image = image
            gt_array = gt_array
            #print('Contrast not added')

        #### Hue
        prob_hue = random.uniform(0, 1)
        if prob_hue > 0.3:
            image,gt_array = hue(image,gt_array)
            #print('hue added')
        else:
            image = image
            gt_array = gt_array
            #print('hue not added')

        ####Saturation
        prob_sat = random.uniform(0, 1)
        if prob_sat > 0.3:
            image,gt_array = saturation(image,gt_array)
            #print('saturation added')
        else:
            image = image
            gt_array = gt_array
            #print('saturation not added')

        ##flip_horizontal
        #prob_flip = random.uniform(0, 1)
        #if prob_flip > 0.5:
        #    image,gt_array = flip_horizontal(image,gt_array)
        #    print('flip added')
        #else:
        #    image = image
        #    gt_array = gt_array
        #    print('flip not added')
        ##flip_vertical
        #prob_flip_v = random.uniform(0, 1)
        #if prob_flip_v > 0.5:
        #    image,gt_array = flip_vertical(image,gt_array)
        #    print('flip_v added')
        #else:
        #    image = image
        #    gt_array = gt_array
        #    print('flip_v not added')
        #### Screw
        prob_screw = random.uniform(0, 1)
        if prob_screw > 0:
            image,gt_array, parameter = screw(image,gt_array)
            #print('screw added')
        else:
            image = image
            gt_array = gt_array
            #print('screw not added')
        #### rotation
        prob_rot = random.uniform(0,1)
        if prob_rot > 0:
            image,gt_array = rotation(image,gt_array,parameter)
            #print('rotation added')
        else:
            image = image
            gt_array = gt_array
            #print('rotation not added')
        ### Channel
        prob_Channel = random.uniform(0, 1)
        if prob_Channel > 0:
            image,gt_array = channel(image,gt_array)
            #print('Channel added')
        else:
            image = image
            gt_array = gt_array
            #print('Channel not added')
        #### randomplacement
        prob_place = random.uniform(0, 1)
        if prob_place > 0:
            image,gt_array = randomplacement(image,gt_array)
            #print('placement added')
        else:
            image = image
            gt_array = gt_array
            #print('placement not added')
        ###Resize
        image,gt_array = resize(image,gt_array)
        #print(gt_array)
    else:
    ###Resize
        image,gt_array = randomplacement(image,gt_array)
        image,gt_array = resize(image,gt_array)
        #print(gt_array)

    ###### Conversion
    for i in range(len(gt_array)):
        gt_x_cen = (gt_array[i,1]+gt_array[i,3])/2
        gt_y_cen = (gt_array[i,2]+gt_array[i,4])/2
        gt_width = gt_array[i,3]-gt_array[i,1]
        gt_height = gt_array[i,4]-gt_array[i,2]
        gt_array[i,1] = gt_x_cen
        gt_array[i,2] = gt_y_cen
        gt_array[i,3] = gt_width
        gt_array[i,4] = gt_height

    ### Call for classification array & localization array

    img_array = gt_array
    clas_feed_4_3_1, local_feed_4_3_1 = label_data(a1,img_array,cat)
    clas_feed_4_3_2, local_feed_4_3_2 = label_data(b1,img_array,cat)
    clas_feed_4_3_3, local_feed_4_3_3 = label_data(c1,img_array,cat)
    clas_feed_4_3_4, local_feed_4_3_4 = label_data(d1,img_array,cat)
    #print(clas_feed_4_3_1[782,:])
    #print(clas_feed_4_3_1[781,:])

    clas_feed_7_1_1, local_feed_7_1_1 = label_data(a2,img_array,cat)
    clas_feed_7_1_2, local_feed_7_1_2 = label_data(b2,img_array,cat)
    clas_feed_7_1_3, local_feed_7_1_3 = label_data(c2,img_array,cat)
    clas_feed_7_1_4, local_feed_7_1_4 = label_data(d2,img_array,cat)
    clas_feed_7_1_5, local_feed_7_1_5 = label_data(e2,img_array,cat)
    clas_feed_7_1_6, local_feed_7_1_6 = label_data(f2,img_array,cat)

    clas_feed_8_2_1, local_feed_8_2_1 = label_data(a3,img_array,cat)
    clas_feed_8_2_2, local_feed_8_2_2 = label_data(b3,img_array,cat)
    clas_feed_8_2_3, local_feed_8_2_3 = label_data(c3,img_array,cat)
    clas_feed_8_2_4, local_feed_8_2_4 = label_data(d3,img_array,cat)
    clas_feed_8_2_5, local_feed_8_2_5 = label_data(e3,img_array,cat)
    clas_feed_8_2_6, local_feed_8_2_6 = label_data(f3,img_array,cat)

    clas_feed_9_2_1, local_feed_9_2_1 = label_data(a4,img_array,cat)
    clas_feed_9_2_2, local_feed_9_2_2 = label_data(b4,img_array,cat)
    clas_feed_9_2_3, local_feed_9_2_3 = label_data(c4,img_array,cat)
    clas_feed_9_2_4, local_feed_9_2_4 = label_data(d4,img_array,cat)
    clas_feed_9_2_5, local_feed_9_2_5 = label_data(e4,img_array,cat)
    clas_feed_9_2_6, local_feed_9_2_6 = label_data(f4,img_array,cat)

    clas_feed_10_2_1, local_feed_10_2_1 = label_data(a5,img_array,cat)
    clas_feed_10_2_2, local_feed_10_2_2 = label_data(b5,img_array,cat)
    clas_feed_10_2_3, local_feed_10_2_3 = label_data(c5,img_array,cat)
    clas_feed_10_2_4, local_feed_10_2_4 = label_data(d5,img_array,cat)

    clas_feed_11_2_1, local_feed_11_2_1 = label_data(a6,img_array,cat)
    clas_feed_11_2_2, local_feed_11_2_2 = label_data(b6,img_array,cat)
    clas_feed_11_2_3, local_feed_11_2_3 = label_data(c6,img_array,cat)
    clas_feed_11_2_4, local_feed_11_2_4 = label_data(d6,img_array,cat)
    #print(clas_feed_11_2_1)
    #data = data 

    feed_arr1[:,:] = np.concatenate((clas_feed_4_3_1, clas_feed_4_3_2, clas_feed_4_3_3 ,clas_feed_4_3_4, clas_feed_7_1_1, clas_feed_7_1_2, clas_feed_7_1_3, clas_feed_7_1_4, clas_feed_7_1_5, clas_feed_7_1_6, clas_feed_8_2_1, clas_feed_8_2_2, clas_feed_8_2_3, clas_feed_8_2_4, clas_feed_8_2_5, clas_feed_8_2_6, clas_feed_9_2_1, clas_feed_9_2_2, clas_feed_9_2_3, clas_feed_9_2_4, clas_feed_9_2_5, clas_feed_9_2_6, clas_feed_10_2_1, clas_feed_10_2_2, clas_feed_10_2_3, clas_feed_10_2_4, clas_feed_11_2_1, clas_feed_11_2_2, clas_feed_11_2_3, clas_feed_11_2_4), axis=0)
    loc_feed_arr1[:,:] = np.concatenate((local_feed_4_3_1, local_feed_4_3_2, local_feed_4_3_3 ,local_feed_4_3_4, local_feed_7_1_1, local_feed_7_1_2, local_feed_7_1_3, local_feed_7_1_4, local_feed_7_1_5, local_feed_7_1_6, local_feed_8_2_1, local_feed_8_2_2, local_feed_8_2_3, local_feed_8_2_4, local_feed_8_2_5, local_feed_8_2_6, local_feed_9_2_1, local_feed_9_2_2, local_feed_9_2_3, local_feed_9_2_4, local_feed_9_2_5, local_feed_9_2_6, local_feed_10_2_1, local_feed_10_2_2, local_feed_10_2_3, local_feed_10_2_4, local_feed_11_2_1, local_feed_11_2_2, local_feed_11_2_3, local_feed_11_2_4), axis=0)
    image_feed[:,:,:] = image
    gt_array_feed = gt_array
    #gt_array_feed= np.concatenate((gt_array_feed,gt_array),axis = 0)

    for i in range(len(gt_array_feed)):
        gt_array_feed[i,1] = 300 * gt_array_feed[i,1]
        gt_array_feed[i,2] = 300 * gt_array_feed[i,2]
        gt_array_feed[i,3] = 300 * gt_array_feed[i,3]
        gt_array_feed[i,4] = 300 * gt_array_feed[i,4]

    feed = np.concatenate((feed_arr1,loc_feed_arr1), axis=1)
    q.put(feed)
    q.put(image_feed)
    q.put(gt_array_feed)
    #return feed, image_feed, gt_array_feed
    return None

def image_data_gen_valid(q,list_pass_valid,cat,a1,b1,c1,d1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3,a4,b4,c4,d4,e4,f4,a5,b5,c5,d5,a6,b6,c6,d6):

    feed_arr1 = np.ones((8732,cat), dtype=np.int32 )
    loc_feed_arr1 = np.ones((8732,4),dtype = np.float32)
    image_feed = np.ones((300,300,3),dtype = np.uint8)
    image = list_pass_valid[0]
    image = cv2.imread (image)

    img_width = int(list_pass_valid[1])
    img_height = int(list_pass_valid[2])
    gt_array = np.zeros((len(list_pass_valid[3]),6))
    for i in range(len(list_pass_valid[3])):
        gt_array[i,0] = label2id(int(float(list_pass_valid[3][i])))
        gt_array[i,1] = float(list_pass_valid[4][i])/img_width
        gt_array[i,2] = float(list_pass_valid[5][i])/img_height
        gt_array[i,3] = float(list_pass_valid[6][i])/img_width
        gt_array[i,4] = float(list_pass_valid[7][i])/img_height
        gt_array[i,5] = 0

    #### convertion
    for i in range(len(gt_array)):
        gt_xmin = gt_array[i,1]-(gt_array[i,3]/2)
        gt_ymin = gt_array[i,2]-(gt_array[i,4]/2)
        gt_xmax = gt_array[i,1]+(gt_array[i,3]/2)
        gt_ymax = gt_array[i,2]+(gt_array[i,4]/2)
        gt_array[i,1] = gt_xmin
        gt_array[i,2] = gt_ymin
        gt_array[i,3] = gt_xmax
        gt_array[i,4] = gt_ymax


    prob_org = random.uniform(0, 1)
    if prob_org > 0.2:
        ## Salt & Pepper
        #prob_s_p = random.uniform(0, 1)
        #if prob_s_p > 0.3:
        #    image,gt_array = s_p(image,gt_array)
        #    print('salt_papper added')
        #else:
        #    image = image
        #    gt_array = gt_array
        #    print('salt_papper not added')

        ## Brightness
        #prob_bri = random.uniform(0, 1)
        #if prob_bri > 0.3:
        #    image,gt_array = brightness(image,gt_array)
        #    print('brightness added')
        #else:
        #    image = image
        #    gt_array = gt_array
        #    print('brightness not added')

        ### Contrast
        #prob_cont = random.uniform(0, 1)
        #if prob_cont > 0.3:
        #    image,gt_array = contrast(image,gt_array)
        #    print('Contrast added')
        #else:
        #    image = image
        #    gt_array = gt_array
        #    print('Contrast not added')

        #### Hue
        #prob_hue = random.uniform(0, 1)
        #if prob_hue > 0.3:
        #    image,gt_array = hue(image,gt_array)
        #    print('hue added')
        #else:
        #    image = image
        #    gt_array = gt_array
        #    print('hue not added')

        ####Saturation
        #prob_sat = random.uniform(0, 1)
        #if prob_sat > 0.3:
        #    image,gt_array = saturation(image,gt_array)
        #    print('saturation added')
        #else:
        #    image = image
        #    gt_array = gt_array
        #    print('saturation not added')

        ##flip_horizontal
        #prob_flip = random.uniform(0, 1)
        #if prob_flip > 0.5:
        #    image,gt_array = flip_horizontal(image,gt_array)
        #    print('flip added')
        #else:
        #    image = image
        #    gt_array = gt_array
        #    print('flip not added')
        ##flip_vertical
        #prob_flip_v = random.uniform(0, 1)
        #if prob_flip_v > 0.5:
        #    image,gt_array = flip_vertical(image,gt_array)
        #    print('flip_v added')
        #else:
        #    image = image
        #    gt_array = gt_array
        #    print('flip_v not added')
        #### Screw
        prob_screw = random.uniform(0, 1)
        if prob_screw > 0:
            image,gt_array,parameter = screw(image,gt_array)
            #print('screw added')
        else:
            image = image
            gt_array = gt_array
            #print('screw not added')
        #### rotation
        prob_rot = random.uniform(0,1)
        if prob_rot > 0:
            image,gt_array = rotation(image,gt_array,parameter)
            #print('rotation added')
        else:
            image = image
            gt_array = gt_array
            #print('rotation not added')
        ### Channel
        #prob_Channel = random.uniform(0, 1)
        #if prob_Channel > 0:
        #    image,gt_array = channel(image,gt_array)
        #    print('Channel added')
        #else:
        #    image = image
        #    gt_array = gt_array
        #    print('Channel not added')
        #### randomplacement
        prob_place = random.uniform(0, 1)
        if prob_place > 0:
            image,gt_array = randomplacement(image,gt_array)
            #print('placement added')
        else:
            image = image
            gt_array = gt_array
            #print('placement not added')
        ###Resize
        image,gt_array = resize(image,gt_array)
        #print(gt_array)
    else:
    ###Resize
        image,gt_array = randomplacement(image,gt_array)
        image,gt_array = resize(image,gt_array)
        #print(gt_array)

    ###### Conversion
    for i in range(len(gt_array)):
        gt_x_cen = (gt_array[i,1]+gt_array[i,3])/2
        gt_y_cen = (gt_array[i,2]+gt_array[i,4])/2
        gt_width = gt_array[i,3]-gt_array[i,1]
        gt_height = gt_array[i,4]-gt_array[i,2]
        gt_array[i,1] = gt_x_cen
        gt_array[i,2] = gt_y_cen
        gt_array[i,3] = gt_width
        gt_array[i,4] = gt_height

    ### Call for classification array & localization array

    img_array = gt_array
    clas_feed_4_3_1, local_feed_4_3_1 = label_data(a1,img_array,cat)
    clas_feed_4_3_2, local_feed_4_3_2 = label_data(b1,img_array,cat)
    clas_feed_4_3_3, local_feed_4_3_3 = label_data(c1,img_array,cat)
    clas_feed_4_3_4, local_feed_4_3_4 = label_data(d1,img_array,cat)
    #print(clas_feed_4_3_1[782,:])
    #print(clas_feed_4_3_1[781,:])

    clas_feed_7_1_1, local_feed_7_1_1 = label_data(a2,img_array,cat)
    clas_feed_7_1_2, local_feed_7_1_2 = label_data(b2,img_array,cat)
    clas_feed_7_1_3, local_feed_7_1_3 = label_data(c2,img_array,cat)
    clas_feed_7_1_4, local_feed_7_1_4 = label_data(d2,img_array,cat)
    clas_feed_7_1_5, local_feed_7_1_5 = label_data(e2,img_array,cat)
    clas_feed_7_1_6, local_feed_7_1_6 = label_data(f2,img_array,cat)

    clas_feed_8_2_1, local_feed_8_2_1 = label_data(a3,img_array,cat)
    clas_feed_8_2_2, local_feed_8_2_2 = label_data(b3,img_array,cat)
    clas_feed_8_2_3, local_feed_8_2_3 = label_data(c3,img_array,cat)
    clas_feed_8_2_4, local_feed_8_2_4 = label_data(d3,img_array,cat)
    clas_feed_8_2_5, local_feed_8_2_5 = label_data(e3,img_array,cat)
    clas_feed_8_2_6, local_feed_8_2_6 = label_data(f3,img_array,cat)

    clas_feed_9_2_1, local_feed_9_2_1 = label_data(a4,img_array,cat)
    clas_feed_9_2_2, local_feed_9_2_2 = label_data(b4,img_array,cat)
    clas_feed_9_2_3, local_feed_9_2_3 = label_data(c4,img_array,cat)
    clas_feed_9_2_4, local_feed_9_2_4 = label_data(d4,img_array,cat)
    clas_feed_9_2_5, local_feed_9_2_5 = label_data(e4,img_array,cat)
    clas_feed_9_2_6, local_feed_9_2_6 = label_data(f4,img_array,cat)

    clas_feed_10_2_1, local_feed_10_2_1 = label_data(a5,img_array,cat)
    clas_feed_10_2_2, local_feed_10_2_2 = label_data(b5,img_array,cat)
    clas_feed_10_2_3, local_feed_10_2_3 = label_data(c5,img_array,cat)
    clas_feed_10_2_4, local_feed_10_2_4 = label_data(d5,img_array,cat)

    clas_feed_11_2_1, local_feed_11_2_1 = label_data(a6,img_array,cat)
    clas_feed_11_2_2, local_feed_11_2_2 = label_data(b6,img_array,cat)
    clas_feed_11_2_3, local_feed_11_2_3 = label_data(c6,img_array,cat)
    clas_feed_11_2_4, local_feed_11_2_4 = label_data(d6,img_array,cat)
    #print(clas_feed_11_2_1)
    #data = data 

    feed_arr1[:,:] = np.concatenate((clas_feed_4_3_1, clas_feed_4_3_2, clas_feed_4_3_3 ,clas_feed_4_3_4, clas_feed_7_1_1, clas_feed_7_1_2, clas_feed_7_1_3, clas_feed_7_1_4, clas_feed_7_1_5, clas_feed_7_1_6, clas_feed_8_2_1, clas_feed_8_2_2, clas_feed_8_2_3, clas_feed_8_2_4, clas_feed_8_2_5, clas_feed_8_2_6, clas_feed_9_2_1, clas_feed_9_2_2, clas_feed_9_2_3, clas_feed_9_2_4, clas_feed_9_2_5, clas_feed_9_2_6, clas_feed_10_2_1, clas_feed_10_2_2, clas_feed_10_2_3, clas_feed_10_2_4, clas_feed_11_2_1, clas_feed_11_2_2, clas_feed_11_2_3, clas_feed_11_2_4), axis=0)
    loc_feed_arr1[:,:] = np.concatenate((local_feed_4_3_1, local_feed_4_3_2, local_feed_4_3_3 ,local_feed_4_3_4, local_feed_7_1_1, local_feed_7_1_2, local_feed_7_1_3, local_feed_7_1_4, local_feed_7_1_5, local_feed_7_1_6, local_feed_8_2_1, local_feed_8_2_2, local_feed_8_2_3, local_feed_8_2_4, local_feed_8_2_5, local_feed_8_2_6, local_feed_9_2_1, local_feed_9_2_2, local_feed_9_2_3, local_feed_9_2_4, local_feed_9_2_5, local_feed_9_2_6, local_feed_10_2_1, local_feed_10_2_2, local_feed_10_2_3, local_feed_10_2_4, local_feed_11_2_1, local_feed_11_2_2, local_feed_11_2_3, local_feed_11_2_4), axis=0)
    image_feed[:,:,:] = image
    gt_array_feed = gt_array
    #gt_array_feed= np.concatenate((gt_array_feed,gt_array),axis = 0)

    for i in range(len(gt_array_feed)):
        gt_array_feed[i,1] = 300 * gt_array_feed[i,1]
        gt_array_feed[i,2] = 300 * gt_array_feed[i,2]
        gt_array_feed[i,3] = 300 * gt_array_feed[i,3]
        gt_array_feed[i,4] = 300 * gt_array_feed[i,4]

    feed = np.concatenate((feed_arr1,loc_feed_arr1), axis=1)
    q.put(feed)
    q.put(image_feed)
    q.put(gt_array_feed)
    #return feed, image_feed, gt_array_feed
    return None
