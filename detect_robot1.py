import numpy as np
import tensorflow as tf
from train_data_gen import grid_maker
import cv2
from nms import non_maxi_sup
from math import exp
from ap import covert,decode,color_tuple
from collections import defaultdict
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
        #cv2.imwrite('Tensorboard'+str(k)+'.png', image)
    img[img>255] = 255
    img[img<0] = 0
    img = img.astype(np.uint8)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite('robot_part_'+str(j)+'.png', img)

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
    cv2.imshow("detect_robot1",image_tb)
    print('Press any keyboard key to close the window.')
    cv2.waitKey(0)
    #cv2.rectangle(image,(int(gt_array[i,1]*300),int(gt_array[i,2]*300)),(int(gt_array[i,3]*300),int(gt_array[i,4]*300)),(0,0,255),2)
    cv2.imwrite('detect_robot'+str(j+1)+'.png', image_tb)
    print('Image File detect_robot1.png is saved.')
  #return image_tb



batch = 20
cat = 10
ther = 0.55
shift =300
image = cv2.imread ('robot1.png')
#image = np.ones((1,300,300,3),dtype = np.uint8)
#image[0,:,:,:] = image1

image = cv2.resize(image,(900,750))
img_random = np.zeros((20,300,300,3),dtype = np.uint8)
coor_x = np.zeros((4,5),dtype= np.float32)
coor_y = np.zeros((4,5),dtype= np.float32)
k=0
for j in range(4):
    for i in range(5):
        start_x = int(300*i*0.5)
        coor_x[j,i] = start_x 
        start_y = int(300*j*0.5)
        coor_y[j,i] = start_y
        img_random[k,:,:,:] = image[start_y:start_y+shift, start_x:start_x+shift]
        #cv2.imwrite('image'+str(k)+'.png', img_random[k,:,:,:])
        k=k+1

a1,b1,c1,d1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3,a4,b4,c4,d4,e4,f4,a5,b5,c5,d5,a6,b6,c6,d6 = grid_maker()
anc_array_feed = np.concatenate((a1,b1,c1,d1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3,a4,b4,c4,d4,e4,f4,a5,b5,c5,d5,a6,b6,c6,d6),axis = 0)
for i in range(8732):
    anc_array_feed[i,0] = 300 * anc_array_feed[i,0]
    anc_array_feed[i,1] = 300 * anc_array_feed[i,1]
    anc_array_feed[i,2] = 300 * anc_array_feed[i,2]
    anc_array_feed[i,3] = 300 * anc_array_feed[i,3]

sess = tf.Session()
new_saver = tf.train.import_meta_graph('final_ckpt_files/model.ckpt-final.meta')
new_saver.restore(sess, 'final_ckpt_files/model.ckpt-final')

image_input = sess.graph.get_tensor_by_name('image_input:0')
output_softmax = sess.graph.get_tensor_by_name('output_softmax:0')

boxes = sess.run(output_softmax, feed_dict = {image_input : img_random})
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
#image_for_tb(img_random,dict5,dict6,batch,ther)

#d1 ,d2 = non_maxi_sup(dict5[0],dict6[0])
coor_x = np.reshape(coor_x,(20,1))
coor_y = np.reshape(coor_y,(20,1))

for i in range(batch):
    if i ==0:
        #print(i)
        #print(dict5[0].shape)
        box_tb, prob_tb, shift_x, shift_y = dict5[i], dict6[i], coor_x[i,0], coor_y[i,0]
        for j in range(len(box_tb)):
          box_tb[j,1] = box_tb[j,1] + shift_x
          box_tb[j,2] = box_tb[j,2] + shift_y
        #print(shift_x)
        #print(shift_y)
        #convert_box_tb = convert_image(box_tb,shift_x,shift_y)
        #print(convert_box_tb.shape)
    if i > 0:
        box_tb_dummy, prob_tb_dummy, shift_x, shift_y = dict5[i], dict6[i], coor_x[i,0], coor_y[i,0]
        #print(shift_x)
        #print(shift_y)
        for j in range(len(box_tb_dummy)):
          box_tb_dummy[j,1] = box_tb_dummy[j,1] + shift_x
          box_tb_dummy[j,2] = box_tb_dummy[j,2] + shift_y
        #convert_box_tb_dummy = convert_image(box_tb,shift_x,shift_y)
        box_tb  = np.concatenate((box_tb ,box_tb_dummy),axis = 0)
        prob_tb = np.concatenate((prob_tb,prob_tb_dummy),axis = 0)
        #print(convert_box_tb_dummy)
#for i in range(len(convert_box_tb)):
#    convert_box_tb[i,5]=1
#print(box_tb.shape)
#print(prob_tb.shape)
f1 ,f2 = non_maxi_sup(box_tb,prob_tb)
#print(f1.shape)
#print(f2.shape)
batch = 1
image_for_tb_(image,f1,f2,batch,ther)
