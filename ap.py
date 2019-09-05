import cv2
import numpy as np
import random
import os
from random import shuffle
from math import exp
from collections import defaultdict
from nms import non_maxi_sup

def A_P(gt_array_feed,boxes,anc_array_feed,batch,cat):
  #batch = 1
  gt_array = gt_array_feed
  #default_factory = lambda: list(range(0,5))
  #a =[]
  number = defaultdict(lambda : [])
  sub_num = defaultdict(int)
  
  for i in range(len(gt_array)):
    sub_num[gt_array[i,0]] += 1
    number[gt_array[i,0]].append(gt_array[i,5])
  number2 = defaultdict(lambda : defaultdict(int))
    #for i in range(len(gt_array)):
        #number2[gt_array[i,0]][gt_array[i,5]] +=1
    #print(number2)

    #print(number)
  number1 = defaultdict(dict)
  for i in range(len(gt_array)):
    number2[gt_array[i,0]][gt_array[i,5]] +=1
    if number2[gt_array[i,0]][gt_array[i,5]] == 1:
      number1[gt_array[i,0]][gt_array[i,5]] = gt_array[i,1:5]
      number1[gt_array[i,0]][gt_array[i,5]] = np.reshape(number1[gt_array[i,0]][gt_array[i,5]], (1, 4))
      bool_arr = np.zeros((1,1),dtype = np.bool)
      number1[gt_array[i,0]][gt_array[i,5]] = np.concatenate((number1[gt_array[i,0]][gt_array[i,5]],bool_arr),axis =1)
    if number2[gt_array[i,0]][gt_array[i,5]] > 1:
      dummy = gt_array[i,1:5]
      dummy = np.reshape(dummy, (1, 4))
      dummy = np.concatenate((dummy,bool_arr),axis =1)
      number1[gt_array[i,0]][gt_array[i,5]] = np.concatenate((number1[gt_array[i,0]][gt_array[i,5]],dummy), axis = 0)
  #print(number2)
  #print(sub_num)
  #print(number1[13])
  #print(boxes.shape)
  #print(b)
  #boxes = np.reshape(boxes, (69856, 24))
  #clas_boxes = boxes[:,:,:20]
  #loc_boxes = boxes[:,:,20:]
  clas_boxes = boxes[:,:,:cat-1]
  loc_boxes = boxes[:,:,cat:]
  sample_no_array = np.zeros((batch,8732,1))
  loc_boxes = np.concatenate((loc_boxes,sample_no_array),axis = 2)
    
  for i in range(batch):
    for j in range(loc_boxes.shape[1]):
      loc_boxes[i,j,4] = i+1
  #print(loc_boxes[7,8731,4])
    
  loc_boxes = np.array((loc_boxes),dtype = np.float64)
  #print(loc_boxes.dtype) 
    
  loc_boxes_decoded = decode(loc_boxes,anc_array_feed,batch)
  #print(loc_boxes)
    
  clas_boxes = np.reshape(clas_boxes, (batch*8732, cat-1))
  loc_boxes_decoded = np.reshape(loc_boxes_decoded, (batch*8732, 5))
    
  label_map = np.argmax(clas_boxes, axis =1)
    
  #label_map = np.array((label_map),dtype = np.int32)
  #print(label_map.shape)
  count = 0
    
  for i in range(len(label_map)):
    if label_map[i] !=0:
      count +=1
  #print(count)
    
  prob_array = clas_boxes[np.arange(batch*8732),label_map]
    
  label_map = np.reshape(label_map, (batch*8732, 1))
  loc_boxes_decoded = np.concatenate((label_map,loc_boxes_decoded),axis = 1)
  loc_boxes_decoded = np.reshape(loc_boxes_decoded,(batch,8732,6))
  prob_array = np.reshape(prob_array,(batch,8732,1))
  
  dict5 = defaultdict()
  dict6 = defaultdict()
  
  for i in range(batch):
    if i ==0:
      d1 ,d2 = non_maxi_sup(loc_boxes_decoded[i,:,:],prob_array[i,:,:])
      dict5[i] = d1
      dict6[i] = d2
    if i>0:
      dummy_d1,dummy_d2 = non_maxi_sup(loc_boxes_decoded[i,:,:],prob_array[i,:,:])
      d1  = np.concatenate((d1,dummy_d1),axis = 0)
      d2 = np.concatenate((d2,dummy_d2),axis = 0)
      dict5[i] = dummy_d1
      dict6[i] = dummy_d2

  #print(d1)
  #print(d2)

  loc_boxes_decoded = d1
  prob_array = d2
  
  #print(clas_boxes[1,8])
  #print(prob_array)
    
  #print(loc_boxes_decoded[69854,5])
    
    #print(number)
    
  average_pre = defaultdict()
  number3 = defaultdict()
  number4 = defaultdict()
  number5 = defaultdict(int)
    
  for i in range(len(loc_boxes_decoded)):
    number5[loc_boxes_decoded[i,0]] +=1
    if number5[loc_boxes_decoded[i,0]] == 1:
      number3[loc_boxes_decoded[i,0]] = loc_boxes_decoded[i,:]
      number4[loc_boxes_decoded[i,0]] = prob_array[i]
      number3[loc_boxes_decoded[i,0]] = np.reshape(number3[loc_boxes_decoded[i,0]], (1, 6))
      number4[loc_boxes_decoded[i,0]] = np.reshape(number4[loc_boxes_decoded[i,0]], (1, 1))
    if number5[loc_boxes_decoded[i,0]] > 1:
      dummy1 = loc_boxes_decoded[i,:]
      dummy1 = np.reshape(dummy1, (1, 6))
      dummy2 = prob_array[i]
      dummy2 = np.reshape(dummy2, (1, 1))
      #dummy = np.concatenate((dummy,bool_arr),axis =1)
      number3[loc_boxes_decoded[i,0]] = np.concatenate((number3[loc_boxes_decoded[i,0]],dummy1), axis = 0)
      number4[loc_boxes_decoded[i,0]] = np.concatenate((number4[loc_boxes_decoded[i,0]],dummy2), axis = 0)
  #print(number5)
  #print(number4)
  #print(number3)
  #print(number3[11].shape)
  #print(number4[11].shape)
  for label in number1:
    average_pre[label] = 0
    for label2 in number3:
      if label == label2:
        label_boxes = number3[label]
        label_prob = number4[label]
        sort_prob = np.argsort(-label_prob, axis =0)
        sort_prob = np.reshape(sort_prob, (len(sort_prob)))
        label_prob = label_prob[sort_prob,:]
        label_prob = np.reshape(label_prob,(len(label_prob),1))
        label_boxes = label_boxes[sort_prob,:]
        label_boxes = np.reshape(label_boxes,(len(label_boxes),6))
        true_positives = np.zeros((len(label_boxes),1),dtype = np.int32)
        count = 0
        for i in range(len(label_boxes)):
          for key in number1[label]:
            if label_boxes[i,5] == key:
              overlap_matrix = overlap_avg_prec(label_boxes[i,1:5],number1[label][key])
              overlap_max_index = np.argmax(overlap_matrix, axis = 1)
              overlap_max = overlap_matrix[0,overlap_max_index]
              
              if overlap_max > 0.5 :
                
                if number1[label][key][overlap_max_index,4] == False:
                  number1[label][key][overlap_max_index,4] = True
                  true_positives[i,0] = 1
        for i in range(len(label_boxes)):
          if true_positives[i,0] == True:
            count = count + 1
        #print(count)
        false_positives = np.ones((len(label_boxes),1),dtype = np.int32)
        prec_axis = np.zeros((len(label_boxes),1),dtype = np.float32)
        recal_axis = np.zeros((len(label_boxes),1),dtype = np.float32)
        for i in range(len(label_boxes)):
          false_positives[i,0] = false_positives[i,0] - true_positives[i,0]
        true_positives = np.cumsum(true_positives)
        false_positives = np.cumsum(false_positives)
        #print(true_positives)
        #print(false_positives)
        #print(true_positives.shape)
        #print(true_positives.shape)
        true_positives = np.reshape(true_positives, (len(label_boxes), 1))
        false_positives = np.reshape(false_positives, (len(label_boxes), 1))
        for i in range(len(label_boxes)):
          prec_axis[i,0] = true_positives[i,0]/(true_positives[i,0]+false_positives[i,0])
          recal_axis[i,0] = true_positives[i,0]/sub_num[label]
        value = 0
        for i in range(0,11,1):
          value_mat = np.zeros((len(label_boxes),1))
          for j in range(len(label_boxes)):
            if recal_axis[j,0] >= (i/10):
              value_mat[j,0] = prec_axis[j,0]
          max_value_index = np.argmax(value_mat,axis =0)
          max_value_index = np.reshape(max_value_index, (1))
          value_max = value_mat[max_value_index,0]
          value = value + value_max
        average_pre[label] = value/11
  #print(average_pre)    
  mean_AP = 0
  for label in number1:
    mean_AP = mean_AP + average_pre[label]
  mean_AP = mean_AP/len(number1)
  #print(mean_AP)
  return mean_AP, dict5, dict6

def overlap_avg_prec(ac, gt):
    ac = np.reshape(ac, (1, 4))
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

def decode(input_boxes,anc_array_feed,batch):
  
  #batch =8  
  input_boxes[input_boxes>500] = 500
  output_boxes = np.zeros((batch,8732,5))
  for i in range(batch):
    for j in range(8732):
      output_boxes[i,j,0] = ((input_boxes[i,j,0]/10)*anc_array_feed[j,2]) + anc_array_feed[j,0]
      output_boxes[i,j,1] = ((input_boxes[i,j,1]/10)*anc_array_feed[j,3]) + anc_array_feed[j,1]
      output_boxes[i,j,2] = (exp(input_boxes[i,j,2]/5))* anc_array_feed[j,2]
      output_boxes[i,j,3] = (exp(input_boxes[i,j,3]/5))* anc_array_feed[j,3]
      output_boxes[i,j,4] = input_boxes[i,j,4]
  
  return output_boxes

def covert(box_tb):
  convert_box_tb = np.zeros((len(box_tb),6))
  for i in range(len(box_tb)):
    x_cen_gt=box_tb[i,1]
    y_cen_gt=box_tb[i,2]
    width_gt=box_tb[i,3]
    height_gt=box_tb[i,4]

            #print(x_cen_gt)
            #print(y_cen_gt)
            #print(width_gt)
            #print(height_gt)
    
    convert_box_tb[i,0] = box_tb[i,0]
    convert_box_tb[i,1] = x_cen_gt + width_gt/2
    convert_box_tb[i,2] = x_cen_gt - width_gt/2
    convert_box_tb[i,3] = y_cen_gt + height_gt/2
    convert_box_tb[i,4] = y_cen_gt - height_gt/2
    convert_box_tb[i,5] = box_tb[i,5]
    
    
    if convert_box_tb[i,1] > 1000:
      convert_box_tb[i,1] = 1000
    if convert_box_tb[i,1] < -100:
      convert_box_tb[i,1] = -100
    if convert_box_tb[i,2] > 1000:
      convert_box_tb[i,2] = 1000
    if convert_box_tb[i,2] < -100:
      convert_box_tb[i,2] = -100
    if convert_box_tb[i,3] > 1000:
      convert_box_tb[i,3] = 1000
    if convert_box_tb[i,3] < -100:
      convert_box_tb[i,3] = -100
    if convert_box_tb[i,4] > 1000:
      convert_box_tb[i,4] = 1000
    if convert_box_tb[i,4] < -100:
      convert_box_tb[i,4] = -100
      
  return convert_box_tb

def image_for_tb(image_feed,dict5,dict6,batch,ther):
  #batch =8
  image_tb = np.zeros((3,512,512,3), dtype = np.uint8)
  for j in range(3):
    box_tb, prob_tb = dict5[j], dict6[j]
    convert_box_tb = covert(box_tb)
    img = image_feed[j,:,:,:]
    img = cv2.resize(img,(512,512))
    img_copy = img
    for k in range(len(convert_box_tb)):
      if prob_tb[k,0] > ther:
        if (int(convert_box_tb[k,0])==0 or int(convert_box_tb[k,0])==1 or int(convert_box_tb[k,0])==2 or int(convert_box_tb[k,0])==3 or int(convert_box_tb[k,0])==4):
          color = color_tuple(int(convert_box_tb[k,0]))
          
          cv2.rectangle(img_copy,(int((convert_box_tb[k,2]/300)*512),int((convert_box_tb[k,4]/300)*512)),(int((convert_box_tb[k,1]/300)*512),int((convert_box_tb[k,3]/300)*512)), color,2)
          cv2.rectangle(img_copy,(int(((convert_box_tb[k,2]/300)*512)-1),int((convert_box_tb[k,3]/300)*512)),(int(((convert_box_tb[k,1]/300)*512)+1),(int((convert_box_tb[k,3]/300)*512)+20)),color,cv2.FILLED)
          #cv2.rectangle(img_copy,(int(convert_box_tb[k,2]),int(convert_box_tb[k,4])),(int(convert_box_tb[k,1]),int(convert_box_tb[k,3])), (128,128,0),2)
          #cv2.rectangle(img_copy,(int(convert_box_tb[k,2]-1),int(convert_box_tb[k,3])),(int(convert_box_tb[k,1]+1),int(convert_box_tb[k,3]+20)), (0,128,128),cv2.FILLED)
          font = cv2.FONT_HERSHEY_DUPLEX

          #cv2.putText(img_copy, str(convert_box_tb[k,0]),((int((convert_box_tb[k,2]/300)*512)+5),(int((convert_box_tb[k,3]/300)*512)+5)),font,0.5,(255,255,255),1,cv2.LINE_AA)
          #cv2.putText(img_copy, str(prob_tb[k,0]),((int((convert_box_tb[k,2]/300)*512)+15),(int((convert_box_tb[k,3]/300)*512)+5)),font,0.5,(255,255,255),1,cv2.LINE_AA)

          cv2.putText(img_copy, str(int(convert_box_tb[k,0])),((int((convert_box_tb[k,2]/300)*512)+5),(int((convert_box_tb[k,3]/300)*512)+15)),font,0.5,(20,20,20),1,cv2.LINE_AA)
          cv2.putText(img_copy,'||',((int((convert_box_tb[k,2]/300)*512)+25),(int((convert_box_tb[k,3]/300)*512)+15)),font,0.5,(20,20,20),1,cv2.LINE_AA)
          percent = str(round((prob_tb[k,0]*100),2))+'%'
          cv2.putText(img_copy, percent,((int((convert_box_tb[k,2]/300)*512)+45),(int((convert_box_tb[k,3]/300)*512)+15)),font,0.5,(20,20,20),1,cv2.LINE_AA)
          cv2.addWeighted(img_copy,0.7,img,0.3,0,img)
          #cv2.imwrite('Tensorboard'+str(k)+'.png', image)
          
        if (int(convert_box_tb[k,0])==5 or int(convert_box_tb[k,0])==6 or int(convert_box_tb[k,0])==7 or int(convert_box_tb[k,0])==8):
          color = color_tuple(int(convert_box_tb[k,0]))
          cv2.rectangle(img_copy,(int((convert_box_tb[k,2]/300)*512),int((convert_box_tb[k,4]/300)*512)),(int((convert_box_tb[k,1]/300)*512),int((convert_box_tb[k,3]/300)*512)), color,2)
          cv2.rectangle(img_copy,(int(((convert_box_tb[k,2]/300)*512)-1),int(((convert_box_tb[k,4]/300)*512)-20)),(int(((convert_box_tb[k,1]/300)*512)+1),int((convert_box_tb[k,4]/300)*512)),color,cv2.FILLED)

          #cv2.rectangle(img_copy,(int(convert_box_tb[k,2]),int(convert_box_tb[k,4])),(int(convert_box_tb[k,1]),int(convert_box_tb[k,3])), (128,128,0),2)
          #cv2.rectangle(img_copy,(int(convert_box_tb[k,2]-1),int(convert_box_tb[k,3])),(int(convert_box_tb[k,1]+1),int(convert_box_tb[k,3]+20)), (0,128,128),cv2.FILLED)
          font = cv2.FONT_HERSHEY_DUPLEX

          #cv2.putText(img_copy, str(convert_box_tb[k,0]),((int((convert_box_tb[k,2]/300)*512)+5),(int((convert_box_tb[k,3]/300)*512)+5)),font,0.5,(255,255,255),1,cv2.LINE_AA)
          #cv2.putText(img_copy, str(prob_tb[k,0]),((int((convert_box_tb[k,2]/300)*512)+15),(int((convert_box_tb[k,3]/300)*512)+5)),font,0.5,(255,255,255),1,cv2.LINE_AA)

          cv2.putText(img_copy, str(int(convert_box_tb[k,0])),((int((convert_box_tb[k,2]/300)*512)+5),int((((convert_box_tb[k,4]/300)*512)-20)+15)),font,0.5,(20,20,20),1,cv2.LINE_AA)
          cv2.putText(img_copy,'||',((int((convert_box_tb[k,2]/300)*512)+25),int((((convert_box_tb[k,4]/300)*512)-20)+15)),font,0.5,(20,20,20),1,cv2.LINE_AA)
          percent = str(round((prob_tb[k,0]*100),2))+'%'
          cv2.putText(img_copy, percent,((int((convert_box_tb[k,2]/300)*512)+45),int((((convert_box_tb[k,4]/300)*512)-20)+15)),font,0.5,(20,20,20),1,cv2.LINE_AA)
          cv2.addWeighted(img_copy,0.7,img,0.3,0,img)
        #cv2.imwrite('Tensorboard'+str(k)+'.png', image)
    img[img>255] = 255
    img[img<0] = 0
    img = img.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image_tb[j,:,:,:] = img
  #for i in range(3):
    #cv2.rectangle(image,(int(gt_array[i,1]*300),int(gt_array[i,2]*300)),(int(gt_array[i,3]*300),int(gt_array[i,4]*300)),(0,0,255),2)
    #cv2.imwrite('tensorboard'+str(i)+'.png', image_tb[i,:,:,:])
  return image_tb

def color_tuple(digit):
  if digit == 0:
    color = (35,142,107)
  if digit == 1:
    color = (90,120,150)
  if digit == 2:
    color = (32,11,119)
  if digit == 3:
    color = (0,74,111)
  if digit == 4:
    color = (255,153,255)
  if digit == 5:
    color = (255,128,0)
  if digit == 6:
    color = (70,70,70)
  if digit == 7:
    color = (60,20,220)
  if digit == 8:
    color = (30,70,250)
  return color
