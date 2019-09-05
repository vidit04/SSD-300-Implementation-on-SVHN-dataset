import cv2
import numpy as np
import random
import os
from random import shuffle
from math import exp
from collections import defaultdict

def non_maxi_sup(result_box2,result_box_prob2):
  #count=0
  #for i in range(len(result_box2)):
  #  if result_box2[i,0] !=0:
  #    count +=1
  #print(count)
  #count1=0
  #for i in range(len(result_box_prob2)):
  #  if result_box_prob2[i,0] !=0:
  #    count1 +=1
  #print(count1)

  
  result_box_prob_index = np.argsort(-result_box_prob2,axis=0)
  #result_box_prob_index = np.reshape(result_box_prob_index, (len(result_box_prob_index)))
  result_box_prob1 = result_box_prob2[result_box_prob_index,:]
  result_box_prob1 = np.reshape(result_box_prob1, (len(result_box_prob1),1))
  result_box1 = result_box2[result_box_prob_index,:]
  result_box1 = np.reshape(result_box1, (len(result_box1),6))
  
  result_box = result_box1[:200,:]
  result_box_prob = result_box_prob1[:200,:]
  
  dict_result = defaultdict()
  dict_num = defaultdict(int)
  dict_prob = defaultdict()
  for i in range(len(result_box)):
    dict_num[result_box[i,0]] += 1 
    if dict_num[result_box[i,0]] == 1:
      dict_result[result_box[i,0]] = result_box[i,:]
      dict_prob[result_box[i,0]] = result_box_prob[i,:]
      dict_result[result_box[i,0]] = np.reshape(dict_result[result_box[i,0]],(1,6))
      dict_prob[result_box[i,0]] = np.reshape(dict_prob[result_box[i,0]],(1,1))
    if dict_num[result_box[i,0]] > 1:
      dummy_result_box = result_box[i,:]
      dummy_result_box_prob = result_box_prob[i,:]
      dummy_result_box = np.reshape(dummy_result_box, (1,6))
      dummy_result_box_prob = np.reshape(dummy_result_box_prob,(1,1))
      dict_result[result_box[i,0]] = np.concatenate((dict_result[result_box[i,0]],dummy_result_box), axis = 0)
      dict_prob[result_box[i,0]] = np.concatenate((dict_prob[result_box[i,0]],dummy_result_box_prob), axis = 0)

  #print(dict_result[0].shape)
  #print(dict_prob[0].shape)
  #print(dict_result[1].shape)
  #print(dict_result[1])
  #print(dict_prob[1])
  #print(dict_result[6].shape)
  #print(dict_result[6])
  #print(dict_prob[6])

  dict1 = dict_result
  dict2 = dict_prob
  k=0

  for i in dict1:
    cat_array = dict1[i]
    cat_prob = dict2[i]
    sort_index = np.argsort(cat_prob, axis = 0)
    sort_index = np.reshape(sort_index,(len(cat_array)))
    cat_prob = cat_prob[sort_index,:]
    
    cat_prob = np.reshape(cat_prob,(len(cat_prob),1))
    cat_array = cat_array[sort_index,:]
    cat_array = np.reshape(cat_array,(len(cat_array),6))

    #print(cat_array.shape)
    #print(cat_prob.shape)
    j = 0
    while True:
      #print(j)
      array_2_com = cat_array[len(cat_array)-1,:]
      array_2_com = np.reshape(array_2_com,(1,6))
      prob_2_com = cat_prob[len(cat_prob)-1,:]
      prob_2_com = np.reshape(prob_2_com,(1,1))
      cat_array = np.delete(cat_array,len(cat_array)-1,axis=0)
      cat_prob = np.delete(cat_prob,len(cat_prob)-1,axis=0)
     
      if j ==0:
        cat_array_2_go = array_2_com
        cat_prob_2_go = prob_2_com
        cat_array_2_go = np.reshape(cat_array_2_go,(1,6))
        cat_prob_2_go = np.reshape(cat_prob_2_go,(1,1))
      if j>0:
        array_dummy = array_2_com
        prob_dummy = prob_2_com
        array_dummy = np.reshape(array_dummy,(1,6))
        prob_dummy = np.reshape(prob_dummy,(1,1))
        cat_array_2_go = np.concatenate((cat_array_2_go,array_dummy), axis = 0)
        cat_prob_2_go = np.concatenate((cat_prob_2_go,prob_dummy), axis = 0)
      
      if len(cat_array) ==0:
        #print('Loop didnt break 1')
        #print(i)
        #print(j)
        break
      ol_nms = overlap_nms(cat_array[:,1:5],array_2_com[:,1:5])
      ol_nms_1 = ol_nms > 0.45
      ol_index = np.where(ol_nms_1 == True)
      cat_array = np.delete(cat_array,ol_index[0],axis=0)
      cat_prob = np.delete(cat_prob,ol_index[0],axis=0)
      
      if len(cat_array) ==0:
        #print('Loop didnt break 2')
        #print(i)
        #print(j)
        break
      j = j+1

    if k ==0:
      dict3 = cat_array_2_go
      dict4 = cat_prob_2_go
    if k>0:
      dummy_dict3 = cat_array_2_go
      dummy_dict4 = cat_prob_2_go
      dict3 = np.concatenate((dict3,dummy_dict3), axis = 0)
      dict4 = np.concatenate((dict4,dummy_dict4), axis = 0)
    k =k+1
  #print(dict3.shape)
  #print(dict4.shape)
  return dict3, dict4

  
def overlap_nms(ac, gt):
    gt = np.reshape(gt, (1, 4))
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

