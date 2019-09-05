import numpy as np
import tensorflow as tf
from train_data_gen import grid_maker
import cv2
from nms import non_maxi_sup
from math import exp
from ap import covert,decode,color_tuple
from collections import defaultdict
from train_read import testMain
import multiprocessing as mp
from list_train import image_data_gen, image_data_gen_valid

def image_for_tb(image_feed,dict5,dict6,batch,ther):
  #batch =8
  image_tb = np.zeros((batch,300,300,3), dtype = np.uint8)
  for j in range(batch):
    box_tb, prob_tb = dict5[j], dict6[j]
    print(j)
    print(box_tb.shape)
    print(prob_tb.shape)
    convert_box_tb = covert(box_tb)
    img = image_feed[j,:,:,:]
    img = cv2.resize(img,(300,300))
    img_copy = img
    for k in range(len(convert_box_tb)):
      if prob_tb[k,0] > ther:
        if (int(convert_box_tb[k,0])==0 or int(convert_box_tb[k,0])==1 or int(convert_box_tb[k,0])==2 or int(convert_box_tb[k,0])==3 or int(convert_box_tb[k,0])==4):
          color = color_tuple(int(convert_box_tb[k,0]))
          
          cv2.rectangle(img_copy,(int((convert_box_tb[k,2])),int((convert_box_tb[k,4]))),(int((convert_box_tb[k,1])),int((convert_box_tb[k,3]))), color,2)
          cv2.rectangle(img_copy,(int(((convert_box_tb[k,2]))-1),int((convert_box_tb[k,3]))),(int(((convert_box_tb[k,1]))+1),(int((convert_box_tb[k,3]))+20)),color,cv2.FILLED)
          #cv2.rectangle(img_copy,(int(convert_box_tb[k,2]),int(convert_box_tb[k,4])),(int(convert_box_tb[k,1]),int(convert_box_tb[k,3])), (128,128,0),2)
          #cv2.rectangle(img_copy,(int(convert_box_tb[k,2]-1),int(convert_box_tb[k,3])),(int(convert_box_tb[k,1]+1),int(convert_box_tb[k,3]+20)), (0,128,128),cv2.FILLED)
          font = cv2.FONT_HERSHEY_DUPLEX

          #cv2.putText(img_copy, str(convert_box_tb[k,0]),((int((convert_box_tb[k,2]/300)*512)+5),(int((convert_box_tb[k,3]/300)*512)+5)),font,0.5,(255,255,255),1,cv2.LINE_AA)
          #cv2.putText(img_copy, str(prob_tb[k,0]),((int((convert_box_tb[k,2]/300)*512)+15),(int((convert_box_tb[k,3]/300)*512)+5)),font,0.5,(255,255,255),1,cv2.LINE_AA)

          cv2.putText(img_copy, str(int(convert_box_tb[k,0])),((int((convert_box_tb[k,2]))+5),(int((convert_box_tb[k,3]))+15)),font,0.5,(20,20,20),1,cv2.LINE_AA)
          cv2.putText(img_copy,'||',((int((convert_box_tb[k,2]))+25),(int((convert_box_tb[k,3]))+15)),font,0.5,(20,20,20),1,cv2.LINE_AA)
          percent = str(round((prob_tb[k,0]*100),2))+'%'
          cv2.putText(img_copy, percent,((int((convert_box_tb[k,2]))+45),(int((convert_box_tb[k,3]))+15)),font,0.5,(20,20,20),1,cv2.LINE_AA)
          cv2.addWeighted(img_copy,0.7,img,0.3,0,img)
          #cv2.imwrite('Tensorboard'+str(k)+'.png', image)
          
        if (int(convert_box_tb[k,0])==5 or int(convert_box_tb[k,0])==6 or int(convert_box_tb[k,0])==7 or int(convert_box_tb[k,0])==8):
          color = color_tuple(int(convert_box_tb[k,0]))
          cv2.rectangle(img_copy,(int((convert_box_tb[k,2])),int((convert_box_tb[k,4]))),(int((convert_box_tb[k,1])),int((convert_box_tb[k,3]))), color,2)
          cv2.rectangle(img_copy,(int(((convert_box_tb[k,2]))-1),int(((convert_box_tb[k,4]))-20)),(int(((convert_box_tb[k,1]))+1),int((convert_box_tb[k,4]))),color,cv2.FILLED)

          #cv2.rectangle(img_copy,(int(convert_box_tb[k,2]),int(convert_box_tb[k,4])),(int(convert_box_tb[k,1]),int(convert_box_tb[k,3])), (128,128,0),2)
          #cv2.rectangle(img_copy,(int(convert_box_tb[k,2]-1),int(convert_box_tb[k,3])),(int(convert_box_tb[k,1]+1),int(convert_box_tb[k,3]+20)), (0,128,128),cv2.FILLED)
          font = cv2.FONT_HERSHEY_DUPLEX

          #cv2.putText(img_copy, str(convert_box_tb[k,0]),((int((convert_box_tb[k,2]/300)*512)+5),(int((convert_box_tb[k,3]/300)*512)+5)),font,0.5,(255,255,255),1,cv2.LINE_AA)
          #cv2.putText(img_copy, str(prob_tb[k,0]),((int((convert_box_tb[k,2]/300)*512)+15),(int((convert_box_tb[k,3]/300)*512)+5)),font,0.5,(255,255,255),1,cv2.LINE_AA)

          cv2.putText(img_copy, str(int(convert_box_tb[k,0])),((int((convert_box_tb[k,2]))+5),int((((convert_box_tb[k,4]))-20)+15)),font,0.5,(20,20,20),1,cv2.LINE_AA)
          cv2.putText(img_copy,'||',((int((convert_box_tb[k,2]))+25),int((((convert_box_tb[k,4]))-20)+15)),font,0.5,(20,20,20),1,cv2.LINE_AA)
          percent = str(round((prob_tb[k,0]*100),2))+'%'
          cv2.putText(img_copy, percent,((int((convert_box_tb[k,2]))+45),int((((convert_box_tb[k,4]))-20)+15)),font,0.5,(20,20,20),1,cv2.LINE_AA)
          cv2.addWeighted(img_copy,0.7,img,0.3,0,img)
    img[img>255] = 255
    img[img<0] = 0
    img = img.astype(np.uint8)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite('valid_detect_'+str(j)+'.png', img)

    #image_tb[j,:,:,:] = img
  #for i in range(3):
    #cv2.rectangle(image,(int(gt_array[i,1]*300),int(gt_array[i,2]*300)),(int(gt_array[i,3]*300),int(gt_array[i,4]*300)),(0,0,255),2)
    #cv2.imwrite('tensorboard'+str(i)+'.png', image_tb[i,:,:,:])
  #return image_tb
def image_for_tb_(image_feed,dict5,dict6,batch,ther):
  #batch =8
  image_tb = np.zeros((960,720,3), dtype = np.uint8)
  for j in range(1):
    box_tb, prob_tb = dict5, dict6
    convert_box_tb = covert(box_tb)
    #convert_box_tb = box_tb
    img = image_feed
    img = cv2.resize(img,(960,720))
    img_copy = img
    for k in range(len(convert_box_tb)):
      if prob_tb[k,0] > ther:
        if (int(convert_box_tb[k,0])==0 or int(convert_box_tb[k,0])==1 or int(convert_box_tb[k,0])==2 or int(convert_box_tb[k,0])==3 or int(convert_box_tb[k,0])==4):
          color = color_tuple(int(convert_box_tb[k,0]))
          
          cv2.rectangle(img_copy,(int((convert_box_tb[k,2]/900)*960),int((convert_box_tb[k,4]/750)*720)),(int((convert_box_tb[k,1]/900)*960),int((convert_box_tb[k,3]/750)*720)), color,2)
          cv2.rectangle(img_copy,(int(((convert_box_tb[k,2]/900)*960)-1),int((convert_box_tb[k,3]/750)*720)),(int(((convert_box_tb[k,1]/900)*960)+1),(int((convert_box_tb[k,3]/750)*720)+20)),color,cv2.FILLED)
          #cv2.rectangle(img_copy,(int(convert_box_tb[k,2]),int(convert_box_tb[k,4])),(int(convert_box_tb[k,1]),int(convert_box_tb[k,3])), (128,128,0),2)
          #cv2.rectangle(img_copy,(int(convert_box_tb[k,2]-1),int(convert_box_tb[k,3])),(int(convert_box_tb[k,1]+1),int(convert_box_tb[k,3]+20)), (0,128,128),cv2.FILLED)
          font = cv2.FONT_HERSHEY_DUPLEX

          #cv2.putText(img_copy, str(convert_box_tb[k,0]),((int((convert_box_tb[k,2]/300)*512)+5),(int((convert_box_tb[k,3]/300)*512)+5)),font,0.5,(255,255,255),1,cv2.LINE_AA)
          #cv2.putText(img_copy, str(prob_tb[k,0]),((int((convert_box_tb[k,2]/300)*512)+15),(int((convert_box_tb[k,3]/300)*512)+5)),font,0.5,(255,255,255),1,cv2.LINE_AA)

          cv2.putText(img_copy, str(int(convert_box_tb[k,0])),((int((convert_box_tb[k,2]/900)*960)+5),(int((convert_box_tb[k,3]/750)*720)+15)),font,0.5,(20,20,20),1,cv2.LINE_AA)
          cv2.putText(img_copy,'||',((int((convert_box_tb[k,2]/900)*960)+25),(int((convert_box_tb[k,3]/750)*720)+15)),font,0.5,(20,20,20),1,cv2.LINE_AA)
          percent = str(round((prob_tb[k,0]*100),2))+'%'
          cv2.putText(img_copy, percent,((int((convert_box_tb[k,2]/900)*960)+45),(int((convert_box_tb[k,3]/750)*720)+15)),font,0.5,(20,20,20),1,cv2.LINE_AA)
          cv2.addWeighted(img_copy,0.7,img,0.3,0,img)
          #cv2.imwrite('Tensorboard'+str(k)+'.png', image)
          
        if (int(convert_box_tb[k,0])==5 or int(convert_box_tb[k,0])==6 or int(convert_box_tb[k,0])==7 or int(convert_box_tb[k,0])==8):
          color = color_tuple(int(convert_box_tb[k,0]))
          cv2.rectangle(img_copy,(int((convert_box_tb[k,2]/900)*960),int((convert_box_tb[k,4]/750)*720)),(int((convert_box_tb[k,1]/900)*960),int((convert_box_tb[k,3]/750)*720)), color,2)
          cv2.rectangle(img_copy,(int(((convert_box_tb[k,2]/900)*960)-1),int(((convert_box_tb[k,4]/750)*720)-20)),(int(((convert_box_tb[k,1]/900)*960)+1),int((convert_box_tb[k,4]/750)*720)),color,cv2.FILLED)

          #cv2.rectangle(img_copy,(int(convert_box_tb[k,2]),int(convert_box_tb[k,4])),(int(convert_box_tb[k,1]),int(convert_box_tb[k,3])), (128,128,0),2)
          #cv2.rectangle(img_copy,(int(convert_box_tb[k,2]-1),int(convert_box_tb[k,3])),(int(convert_box_tb[k,1]+1),int(convert_box_tb[k,3]+20)), (0,128,128),cv2.FILLED)
          font = cv2.FONT_HERSHEY_DUPLEX

          #cv2.putText(img_copy, str(convert_box_tb[k,0]),((int((convert_box_tb[k,2]/300)*512)+5),(int((convert_box_tb[k,3]/300)*512)+5)),font,0.5,(255,255,255),1,cv2.LINE_AA)
          #cv2.putText(img_copy, str(prob_tb[k,0]),((int((convert_box_tb[k,2]/300)*512)+15),(int((convert_box_tb[k,3]/300)*512)+5)),font,0.5,(255,255,255),1,cv2.LINE_AA)

          cv2.putText(img_copy, str(int(convert_box_tb[k,0])),((int((convert_box_tb[k,2]/900)*960)+5),int((((convert_box_tb[k,4]/750)*720)-20)+15)),font,0.5,(20,20,20),1,cv2.LINE_AA)
          cv2.putText(img_copy,'||',((int((convert_box_tb[k,2]/900)*960)+25),int((((convert_box_tb[k,4]/750)*720)-20)+15)),font,0.5,(20,20,20),1,cv2.LINE_AA)
          percent = str(round((prob_tb[k,0]*100),2))+'%'
          cv2.putText(img_copy, percent,((int((convert_box_tb[k,2]/900)*960)+45),int((((convert_box_tb[k,4]/750)*720)-20)+15)),font,0.5,(20,20,20),1,cv2.LINE_AA)
          cv2.addWeighted(img_copy,0.7,img,0.3,0,img)
    img[img>255] = 255
    img[img<0] = 0
    img = img.astype(np.uint8)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image_tb = img
  
    #cv2.rectangle(image,(int(gt_array[i,1]*300),int(gt_array[i,2]*300)),(int(gt_array[i,3]*300),int(gt_array[i,4]*300)),(0,0,255),2)
    cv2.imwrite('detect_robot'+str(j+1)+'.png', image_tb)
  #return image_tb


def multi_gen_valid(list_pass_valid,batch_valid,cat):

    a1,b1,c1,d1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3,a4,b4,c4,d4,e4,f4,a5,b5,c5,d5,a6,b6,c6,d6 = grid_maker()
    feed_valid = np.ones( (batch,8732,cat+4), dtype=np.float32 )
    image_feed_valid = np.ones((batch,300,300,3),dtype = np.uint8)

    q1_v = mp.Queue()
    q2_v = mp.Queue()
    q3_v = mp.Queue()
    q4_v = mp.Queue()
    q5_v = mp.Queue()
    q6_v = mp.Queue()
    q7_v = mp.Queue()
    q8_v = mp.Queue()
    q9_v = mp.Queue()
    q10_v = mp.Queue()
    q11_v = mp.Queue()
    q12_v = mp.Queue()
    q13_v = mp.Queue()
    q14_v = mp.Queue()
    q15_v = mp.Queue()
    q16_v = mp.Queue()
    q17_v = mp.Queue()
    q18_v = mp.Queue()
    q19_v = mp.Queue()
    q20_v = mp.Queue()
    q21_v = mp.Queue()
    q22_v = mp.Queue()
    q23_v = mp.Queue()
    q24_v = mp.Queue()
    q25_v = mp.Queue()
    q26_v = mp.Queue()
    q27_v = mp.Queue()
    q28_v = mp.Queue()
    q29_v = mp.Queue()
    q30_v = mp.Queue()
    q31_v = mp.Queue()
    q32_v = mp.Queue()


    p1_v = mp.Process(target=image_data_gen_valid, args=(q1_v,list_pass_valid[0],cat,a1,b1,c1,d1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3,a4,b4,c4,d4,e4,f4,a5,b5,c5,d5,a6,b6,c6,d6))
    p2_v = mp.Process(target=image_data_gen_valid, args=(q2_v,list_pass_valid[1],cat,a1,b1,c1,d1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3,a4,b4,c4,d4,e4,f4,a5,b5,c5,d5,a6,b6,c6,d6))
    p3_v = mp.Process(target=image_data_gen_valid, args=(q3_v,list_pass_valid[2],cat,a1,b1,c1,d1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3,a4,b4,c4,d4,e4,f4,a5,b5,c5,d5,a6,b6,c6,d6))
    p4_v = mp.Process(target=image_data_gen_valid, args=(q4_v,list_pass_valid[3],cat,a1,b1,c1,d1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3,a4,b4,c4,d4,e4,f4,a5,b5,c5,d5,a6,b6,c6,d6))
    p5_v = mp.Process(target=image_data_gen_valid, args=(q5_v,list_pass_valid[4],cat,a1,b1,c1,d1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3,a4,b4,c4,d4,e4,f4,a5,b5,c5,d5,a6,b6,c6,d6))
    p6_v = mp.Process(target=image_data_gen_valid, args=(q6_v,list_pass_valid[5],cat,a1,b1,c1,d1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3,a4,b4,c4,d4,e4,f4,a5,b5,c5,d5,a6,b6,c6,d6))
    p7_v = mp.Process(target=image_data_gen_valid, args=(q7_v,list_pass_valid[6],cat,a1,b1,c1,d1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3,a4,b4,c4,d4,e4,f4,a5,b5,c5,d5,a6,b6,c6,d6))
    p8_v = mp.Process(target=image_data_gen_valid, args=(q8_v,list_pass_valid[7],cat,a1,b1,c1,d1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3,a4,b4,c4,d4,e4,f4,a5,b5,c5,d5,a6,b6,c6,d6))
    p9_v = mp.Process(target=image_data_gen_valid, args=(q9_v,list_pass_valid[8],cat,a1,b1,c1,d1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3,a4,b4,c4,d4,e4,f4,a5,b5,c5,d5,a6,b6,c6,d6))
    p10_v = mp.Process(target=image_data_gen_valid, args=(q10_v,list_pass_valid[9],cat,a1,b1,c1,d1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3,a4,b4,c4,d4,e4,f4,a5,b5,c5,d5,a6,b6,c6,d6))
    p11_v = mp.Process(target=image_data_gen_valid, args=(q11_v,list_pass_valid[10],cat,a1,b1,c1,d1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3,a4,b4,c4,d4,e4,f4,a5,b5,c5,d5,a6,b6,c6,d6))
    p12_v = mp.Process(target=image_data_gen_valid, args=(q12_v,list_pass_valid[11],cat,a1,b1,c1,d1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3,a4,b4,c4,d4,e4,f4,a5,b5,c5,d5,a6,b6,c6,d6))
    p13_v = mp.Process(target=image_data_gen_valid, args=(q13_v,list_pass_valid[12],cat,a1,b1,c1,d1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3,a4,b4,c4,d4,e4,f4,a5,b5,c5,d5,a6,b6,c6,d6))
    p14_v = mp.Process(target=image_data_gen_valid, args=(q14_v,list_pass_valid[13],cat,a1,b1,c1,d1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3,a4,b4,c4,d4,e4,f4,a5,b5,c5,d5,a6,b6,c6,d6))
    p15_v = mp.Process(target=image_data_gen_valid, args=(q15_v,list_pass_valid[14],cat,a1,b1,c1,d1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3,a4,b4,c4,d4,e4,f4,a5,b5,c5,d5,a6,b6,c6,d6))
    p16_v = mp.Process(target=image_data_gen_valid, args=(q16_v,list_pass_valid[15],cat,a1,b1,c1,d1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3,a4,b4,c4,d4,e4,f4,a5,b5,c5,d5,a6,b6,c6,d6))
    p17_v = mp.Process(target=image_data_gen_valid, args=(q17_v,list_pass_valid[16],cat,a1,b1,c1,d1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3,a4,b4,c4,d4,e4,f4,a5,b5,c5,d5,a6,b6,c6,d6))
    p18_v = mp.Process(target=image_data_gen_valid, args=(q18_v,list_pass_valid[17],cat,a1,b1,c1,d1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3,a4,b4,c4,d4,e4,f4,a5,b5,c5,d5,a6,b6,c6,d6))
    p19_v = mp.Process(target=image_data_gen_valid, args=(q19_v,list_pass_valid[18],cat,a1,b1,c1,d1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3,a4,b4,c4,d4,e4,f4,a5,b5,c5,d5,a6,b6,c6,d6))
    p20_v = mp.Process(target=image_data_gen_valid, args=(q20_v,list_pass_valid[19],cat,a1,b1,c1,d1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3,a4,b4,c4,d4,e4,f4,a5,b5,c5,d5,a6,b6,c6,d6))
    p21_v = mp.Process(target=image_data_gen_valid, args=(q21_v,list_pass_valid[20],cat,a1,b1,c1,d1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3,a4,b4,c4,d4,e4,f4,a5,b5,c5,d5,a6,b6,c6,d6))
    p22_v = mp.Process(target=image_data_gen_valid, args=(q22_v,list_pass_valid[21],cat,a1,b1,c1,d1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3,a4,b4,c4,d4,e4,f4,a5,b5,c5,d5,a6,b6,c6,d6))
    p23_v = mp.Process(target=image_data_gen_valid, args=(q23_v,list_pass_valid[22],cat,a1,b1,c1,d1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3,a4,b4,c4,d4,e4,f4,a5,b5,c5,d5,a6,b6,c6,d6))
    p24_v = mp.Process(target=image_data_gen_valid, args=(q24_v,list_pass_valid[23],cat,a1,b1,c1,d1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3,a4,b4,c4,d4,e4,f4,a5,b5,c5,d5,a6,b6,c6,d6))
    p25_v = mp.Process(target=image_data_gen_valid, args=(q25_v,list_pass_valid[24],cat,a1,b1,c1,d1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3,a4,b4,c4,d4,e4,f4,a5,b5,c5,d5,a6,b6,c6,d6))
    p26_v = mp.Process(target=image_data_gen_valid, args=(q26_v,list_pass_valid[25],cat,a1,b1,c1,d1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3,a4,b4,c4,d4,e4,f4,a5,b5,c5,d5,a6,b6,c6,d6))
    p27_v = mp.Process(target=image_data_gen_valid, args=(q27_v,list_pass_valid[26],cat,a1,b1,c1,d1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3,a4,b4,c4,d4,e4,f4,a5,b5,c5,d5,a6,b6,c6,d6))
    p28_v = mp.Process(target=image_data_gen_valid, args=(q28_v,list_pass_valid[27],cat,a1,b1,c1,d1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3,a4,b4,c4,d4,e4,f4,a5,b5,c5,d5,a6,b6,c6,d6))
    p29_v = mp.Process(target=image_data_gen_valid, args=(q29_v,list_pass_valid[28],cat,a1,b1,c1,d1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3,a4,b4,c4,d4,e4,f4,a5,b5,c5,d5,a6,b6,c6,d6))
    p30_v = mp.Process(target=image_data_gen_valid, args=(q30_v,list_pass_valid[29],cat,a1,b1,c1,d1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3,a4,b4,c4,d4,e4,f4,a5,b5,c5,d5,a6,b6,c6,d6))
    p31_v = mp.Process(target=image_data_gen_valid, args=(q31_v,list_pass_valid[30],cat,a1,b1,c1,d1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3,a4,b4,c4,d4,e4,f4,a5,b5,c5,d5,a6,b6,c6,d6))
    p32_v = mp.Process(target=image_data_gen_valid, args=(q32_v,list_pass_valid[31],cat,a1,b1,c1,d1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3,a4,b4,c4,d4,e4,f4,a5,b5,c5,d5,a6,b6,c6,d6))

    p1_v.start()
    p2_v.start()
    p3_v.start()
    p4_v.start()
    p5_v.start()
    p6_v.start()
    p7_v.start()
    p8_v.start()
    p9_v.start()
    p10_v.start()
    p11_v.start()
    p12_v.start()
    p13_v.start()
    p14_v.start()
    p15_v.start()
    p16_v.start()
    p17_v.start()
    p18_v.start()
    p19_v.start()
    p20_v.start()
    p21_v.start()
    p22_v.start()
    p23_v.start()
    p24_v.start()
    p25_v.start()
    p26_v.start()
    p27_v.start()
    p28_v.start()
    p29_v.start()
    p30_v.start()
    p31_v.start()
    p32_v.start()

    
    feed_valid[0,:,:] =q1_v.get()
    feed_valid[1,:,:] =q2_v.get()
    feed_valid[2,:,:] =q3_v.get()
    feed_valid[3,:,:] =q4_v.get()
    feed_valid[4,:,:] =q5_v.get()
    feed_valid[5,:,:] =q6_v.get()
    feed_valid[6,:,:] =q7_v.get()
    feed_valid[7,:,:] =q8_v.get()
    feed_valid[8,:,:] =q9_v.get()
    feed_valid[9,:,:] =q10_v.get()
    feed_valid[10,:,:] =q11_v.get()
    feed_valid[11,:,:] =q12_v.get()
    feed_valid[12,:,:] =q13_v.get()
    feed_valid[13,:,:] =q14_v.get()
    feed_valid[14,:,:] =q15_v.get()
    feed_valid[15,:,:] =q16_v.get()
    feed_valid[16,:,:] =q17_v.get()
    feed_valid[17,:,:] =q18_v.get()
    feed_valid[18,:,:] =q19_v.get()
    feed_valid[19,:,:] =q20_v.get()
    feed_valid[20,:,:] =q21_v.get()
    feed_valid[21,:,:] =q22_v.get()
    feed_valid[22,:,:] =q23_v.get()
    feed_valid[23,:,:] =q24_v.get()
    feed_valid[24,:,:] =q25_v.get()
    feed_valid[25,:,:] =q26_v.get()
    feed_valid[26,:,:] =q27_v.get()
    feed_valid[27,:,:] =q28_v.get()
    feed_valid[28,:,:] =q29_v.get()
    feed_valid[29,:,:] =q30_v.get()
    feed_valid[30,:,:] =q31_v.get()
    feed_valid[31,:,:] =q32_v.get()

    image_feed_valid[0,:,:,:] = q1_v.get()
    image_feed_valid[1,:,:,:] = q2_v.get()
    image_feed_valid[2,:,:,:] = q3_v.get()
    image_feed_valid[3,:,:,:] = q4_v.get()
    image_feed_valid[4,:,:,:] = q5_v.get()
    image_feed_valid[5,:,:,:] = q6_v.get()
    image_feed_valid[6,:,:,:] = q7_v.get()
    image_feed_valid[7,:,:,:] = q8_v.get()
    image_feed_valid[8,:,:,:] = q9_v.get()
    image_feed_valid[9,:,:,:] = q10_v.get()
    image_feed_valid[10,:,:,:] = q11_v.get()
    image_feed_valid[11,:,:,:] = q12_v.get()
    image_feed_valid[12,:,:,:] = q13_v.get()
    image_feed_valid[13,:,:,:] = q14_v.get()
    image_feed_valid[14,:,:,:] = q15_v.get()
    image_feed_valid[15,:,:,:] = q16_v.get()
    image_feed_valid[16,:,:,:] = q17_v.get()
    image_feed_valid[17,:,:,:] = q18_v.get()
    image_feed_valid[18,:,:,:] = q19_v.get()
    image_feed_valid[19,:,:,:] = q20_v.get()
    image_feed_valid[20,:,:,:] = q21_v.get()
    image_feed_valid[21,:,:,:] = q22_v.get()
    image_feed_valid[22,:,:,:] = q23_v.get()
    image_feed_valid[23,:,:,:] = q24_v.get()
    image_feed_valid[24,:,:,:] = q25_v.get()
    image_feed_valid[25,:,:,:] = q26_v.get()
    image_feed_valid[26,:,:,:] = q27_v.get()
    image_feed_valid[27,:,:,:] = q28_v.get()
    image_feed_valid[28,:,:,:] = q29_v.get()
    image_feed_valid[29,:,:,:] = q30_v.get()
    image_feed_valid[30,:,:,:] = q31_v.get()
    image_feed_valid[31,:,:,:] = q32_v.get()

    z1_v = q1_v.get()
    z2_v = q2_v.get()
    z3_v = q3_v.get()
    z4_v = q4_v.get()
    z5_v = q5_v.get()
    z6_v = q6_v.get()
    z7_v = q7_v.get()
    z8_v = q8_v.get()
    z9_v = q9_v.get()
    z10_v = q10_v.get()
    z11_v = q11_v.get()
    z12_v = q12_v.get()
    z13_v = q13_v.get()
    z14_v = q14_v.get()
    z15_v = q15_v.get()
    z16_v = q16_v.get()
    z17_v = q17_v.get()
    z18_v = q18_v.get()
    z19_v = q19_v.get()
    z20_v = q20_v.get()
    z21_v = q21_v.get()
    z22_v = q22_v.get()
    z23_v = q23_v.get()
    z24_v = q24_v.get()
    z25_v = q25_v.get()
    z26_v = q26_v.get()
    z27_v = q27_v.get()
    z28_v = q28_v.get()
    z29_v = q29_v.get()
    z30_v = q30_v.get()
    z31_v = q31_v.get()
    z32_v = q32_v.get()

    p1_v.join()
    p2_v.join()
    p3_v.join()
    p4_v.join()
    p5_v.join()
    p6_v.join()
    p7_v.join()
    p8_v.join()
    p9_v.join()
    p10_v.join()
    p11_v.join()
    p12_v.join()
    p13_v.join()
    p14_v.join()
    p15_v.join()
    p16_v.join()
    p17_v.join()
    p18_v.join()
    p19_v.join()
    p20_v.join()
    p21_v.join()
    p22_v.join()
    p23_v.join()
    p24_v.join()
    p25_v.join()
    p26_v.join()
    p27_v.join()
    p28_v.join()
    p29_v.join()
    p30_v.join()
    p31_v.join()
    p32_v.join()

    for i in range(len(z1_v)):
        z1_v[i,5] = 1
    for i in range(len(z2_v)):
        z2_v[i,5] = 2
    for i in range(len(z3_v)):
        z3_v[i,5] = 3
    for i in range(len(z4_v)):
        z4_v[i,5] = 4
    for i in range(len(z5_v)):
        z5_v[i,5] = 5
    for i in range(len(z6_v)):
        z6_v[i,5] = 6
    for i in range(len(z7_v)):
        z7_v[i,5] = 7
    for i in range(len(z8_v)):
        z8_v[i,5] = 8
    for i in range(len(z9_v)):
        z9_v[i,5] = 9
    for i in range(len(z10_v)):
        z10_v[i,5] = 10
    for i in range(len(z11_v)):
        z11_v[i,5] = 11
    for i in range(len(z12_v)):
        z12_v[i,5] = 12
    for i in range(len(z13_v)):
        z13_v[i,5] = 13
    for i in range(len(z14_v)):
        z14_v[i,5] = 14
    for i in range(len(z15_v)):
        z15_v[i,5] = 15
    for i in range(len(z16_v)):
        z16_v[i,5] = 16
    for i in range(len(z17_v)):
        z17_v[i,5] = 17
    for i in range(len(z18_v)):
        z18_v[i,5] = 18
    for i in range(len(z19_v)):
        z19_v[i,5] = 19
    for i in range(len(z20_v)):
        z20_v[i,5] = 20
    for i in range(len(z21_v)):
        z21_v[i,5] = 21
    for i in range(len(z22_v)):
        z22_v[i,5] = 22
    for i in range(len(z23_v)):
        z23_v[i,5] = 23
    for i in range(len(z24_v)):
        z24_v[i,5] = 24
    for i in range(len(z25_v)):
        z25_v[i,5] = 25
    for i in range(len(z26_v)):
        z26_v[i,5] = 26
    for i in range(len(z27_v)):
        z27_v[i,5] = 27
    for i in range(len(z28_v)):
        z28_v[i,5] = 28
    for i in range(len(z29_v)):
        z29_v[i,5] = 29
    for i in range(len(z30_v)):
        z30_v[i,5] = 30
    for i in range(len(z31_v)):
        z31_v[i,5] = 31
    for i in range(len(z32_v)):
        z32_v[i,5] = 32

    gt_array_feed_valid = np.concatenate((z1_v,z2_v,z3_v,z4_v,z5_v,z6_v,z7_v,z8_v,z9_v,z10_v,z11_v,z12_v,z13_v,z14_v,z15_v,z16_v,z17_v,z18_v,z19_v,z20_v,z21_v,z22_v,z23_v,z24_v,z25_v,z26_v,z27_v,z28_v,z29_v,z30_v,z31_v,z32_v),axis = 0)
        #print(gt_array_feed)

    anc_array_feed_valid = np.concatenate((a1,b1,c1,d1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3,a4,b4,c4,d4,e4,f4,a5,b5,c5,d5,a6,b6,c6,d6),axis = 0)

    for i in range(8732):

        anc_array_feed_valid[i,0] = 300 * anc_array_feed_valid[i,0]
        anc_array_feed_valid[i,1] = 300 * anc_array_feed_valid[i,1]
        anc_array_feed_valid[i,2] = 300 * anc_array_feed_valid[i,2]
        anc_array_feed_valid[i,3] = 300 * anc_array_feed_valid[i,3]

    return feed_valid,image_feed_valid, gt_array_feed_valid, anc_array_feed_valid
 
if __name__=='__main__':

    #list_training_data = testMain()
    #list_extra_data = testMain_extra()
    #list_training_data = list_training_data + list_extra_data
    print('Reading test dataset...')
    dataset_3 = 'test'
    list_valid_data = testMain(dataset_3)
    epochs = 10
    batch = 32
    batch_valid = batch
    cat= 10
    ther =0.5

    sess = tf.Session()
    new_saver = tf.train.import_meta_graph('C:/Users/guptav/Desktop/SVHN/checkpoint/model.ckpt-11040.meta')
    new_saver.restore(sess, 'C:/Users/guptav/Desktop/SVHN/checkpoint/model.ckpt-11040')

    image_input = sess.graph.get_tensor_by_name('image_input:0')
    output_softmax = sess.graph.get_tensor_by_name('output_softmax:0')

    list_pass_valid = []
    r_no_v = np.random.randint(0,len(list_valid_data),size = batch, dtype=np.int32)
    for k in range(batch):
      random_image = list_valid_data[r_no_v[k]]
      list_pass_valid.append(random_image)

    feed_valid,image_feed_valid, gt_array_feed_valid, anc_array_feed_valid = multi_gen_valid(list_pass_valid,batch_valid,cat)
    anc_array_feed = anc_array_feed_valid 

    boxes = sess.run(output_softmax, feed_dict = {image_input : image_feed_valid})
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
    print(label_map.shape)
    count = 0
    
    for i in range(len(label_map)):
        if label_map[i] !=0:
            count +=1
    print(count)
    
    prob_array = clas_boxes[np.arange(batch*8732),label_map]
    
    label_map = np.reshape(label_map, (batch*8732, 1))
    loc_boxes_decoded = np.concatenate((label_map,loc_boxes_decoded),axis = 1)
    loc_boxes_decoded = np.reshape(loc_boxes_decoded,(batch,8732,6))
    prob_array = np.reshape(prob_array,(batch,8732,1))

    #d1 ,d2 = non_maxi_sup(loc_boxes_decoded[0,:,:],prob_array[0,:,:])
    #image_for_tb(image1,d1,d2,batch,ther)

    dict5 = defaultdict()
    dict6 = defaultdict()
  
    for i in range(batch):
        #if i ==0:
        d1 ,d2 = non_maxi_sup(loc_boxes_decoded[i,:,:],prob_array[i,:,:])
        dict5[i] = d1
        dict6[i] = d2
        #if i>0:
            #dummy_d1,dummy_d2 = non_maxi_sup(loc_boxes_decoded[i,:,:],prob_array[i,:,:])
            #d1  = np.concatenate((d1,dummy_d1),axis = 0)
            #d2 = np.concatenate((d2,dummy_d2),axis = 0)
            #dict5[i] = dummy_d1
            #dict6[i] = dummy_d2



    #batch = 1
    image_for_tb(image_feed_valid,dict5,dict6,batch,ther)

