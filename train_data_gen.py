import numpy as np
import xml.etree.ElementTree as ET
from math import log
import cv2
import random
from tqdm import tqdm
import requests
import math
import tarfile
import zipfile
import os.path
from list_train import grid_cen

def file_input():
    list45 = []
    list1 = []
    
    file = open("C:/Users/user/Desktop/new/tensorflow-without-a-phd-master/tensorflow-mnist-tutorial/File/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt",'r')
    #file1 = open("C:/Users/user/Desktop/new/tensorflow-without-a-phd-master/tensorflow-mnist-tutorial/File/VOCdevkit/VOC2012/ImageSets/Main/trainval.txt",'r')
    for i in file:
        #print(i.strip())
        list45.append(i.strip())
        Filename = 'C:/Users/user/Desktop/new/tensorflow-without-a-phd-master/tensorflow-mnist-tutorial/File/VOCdevkit/VOC2007/Annotations/' + i.strip() + '.xml'
        list1.append(Filename)
    print(len(list1))

    #for i in file1:
        #print(i.strip())
    #   list.append(i.strip())
    #   Filename = 'C:/Users/user/Desktop/new/tensorflow-without-a-phd-master/tensorflow-mnist-tutorial/File/VOCdevkit/VOC2012/Annotations/' + i.strip() + '.xml'
    #   list1.append(Filename)
    #print(len(list1))
    #print(list)
    #print(list1)
    #list = np.array(list)
    #list1 = np.array(list1)
    #print(list)
    #print(list1)
    #print(list[0])
    # print(list1[0])
    list1_d = []
    for i in range(100):
        list1_10 = list1[i]
        list1_d.append(list1_10)
    print(len(list1_d))



    label_name=[]
    x_cen_box=[]
    y_cen_box=[]
    width_box=[]
    height_box=[]
    #width_list=[]
    #height_list=[]
    #filename_list = []
    list2 = []
    for i in list1_d:
        tree = ET.parse(i)
        root = tree.getroot()
        root.findall(".")
        filename = root.findall("./filename")[0].text
        #print(filename)
        width = root.findall("./size/width")[0].text
        #print(width)
        height = root.findall("./size/height")[0].text
        #print(height)
        obj = root.findall("./object")
        for i in obj:
            name = i.findall("./name")[0].text
            #print(name)
            xmin = int(i.findall("./bndbox/xmin")[0].text)
            #print(xmin)
            ymin = int(i.findall("./bndbox/ymin")[0].text)
            #print(ymin)
            xmax = int(i.findall("./bndbox/xmax")[0].text)
            #print(xmax)
            ymax = int(i.findall("./bndbox/ymax")[0].text)
            x = (xmax+xmin)/2
            y = (ymax+ymin)/2
            w = xmax - xmin
            h = ymax - ymin
            label_name.append(name)
            x_cen_box.append(x)
            y_cen_box.append(y)
            width_box.append(w)
            height_box.append(h)
            Data_img = [filename, width, height ,label_name, x_cen_box, y_cen_box, width_box, height_box]
        
        print(Data_img)
        list2.append(Data_img)
        label_name=[]
        x_cen_box=[]
        y_cen_box=[]
        width_box=[]
        height_box=[]
    print(list2)

    return list2

    #print(list2[0][4][0])
    #print(list2[0][5][0])
    #print(list2[0][6][0])
    #print(list2[0][7][0])
def grid_maker():
    
    layer_1 = [38, 30, 60, [2]]
    layer_2 = [19 ,60, 111, [2,3]]
    layer_3 = [10, 111, 162, [2,3]]
    layer_4 = [5, 162, 213, [2,3]]
    layer_5 = [3, 213, 264, [2]]
    layer_6 = [1, 264, 315, [2]]
    #layers = [layer_1, layer_2, layer_3, layer_4, layer_5, layer_6]

    a1, b1, c1, d1, e1, f1= grid_cen (layer_1[0], layer_1[1], layer_1[2], layer_1[3])
    a2, b2, c2, d2, e2, f2= grid_cen (layer_2[0], layer_2[1], layer_2[2], layer_2[3])
    a3, b3, c3, d3, e3, f3= grid_cen (layer_3[0], layer_3[1], layer_3[2], layer_3[3])
    a4, b4, c4, d4, e4, f4= grid_cen (layer_4[0], layer_4[1], layer_4[2], layer_4[3])
    a5, b5, c5, d5, e5, f5= grid_cen (layer_5[0], layer_5[1], layer_5[2], layer_5[3])
    a6, b6, c6, d6, e6, f6= grid_cen (layer_6[0], layer_6[1], layer_6[2], layer_6[3])

    a1 = np.array((a1), dtype= np.float32)
    b1 = np.array((b1), dtype= np.float32)
    c1 = np.array((c1), dtype= np.float32)
    d1 = np.array((d1), dtype= np.float32)

    a2 = np.array((a2), dtype= np.float32)
    b2 = np.array((b2), dtype= np.float32)
    c2 = np.array((c2), dtype= np.float32)
    d2 = np.array((d2), dtype= np.float32)
    e2 = np.array((e2), dtype= np.float32)
    f2 = np.array((f2), dtype= np.float32)

    a3 = np.array((a3), dtype= np.float32)
    b3 = np.array((b3), dtype= np.float32)
    c3 = np.array((c3), dtype= np.float32)
    d3 = np.array((d3), dtype= np.float32)
    e3 = np.array((e3), dtype= np.float32)
    f3 = np.array((f3), dtype= np.float32)

    a4 = np.array((a4), dtype= np.float32)
    b4 = np.array((b4), dtype= np.float32)
    c4 = np.array((c4), dtype= np.float32)
    d4 = np.array((d4), dtype= np.float32)
    e4 = np.array((e4), dtype= np.float32)
    f4 = np.array((f4), dtype= np.float32)

    a5 = np.array((a5), dtype= np.float32)
    b5 = np.array((b5), dtype= np.float32)
    c5 = np.array((c5), dtype= np.float32)
    d5 = np.array((d5), dtype= np.float32)


    a6 = np.array((a6), dtype= np.float32)
    b6 = np.array((b6), dtype= np.float32)
    c6 = np.array((c6), dtype= np.float32)
    d6 = np.array((d6), dtype= np.float32)

    #g = np.concatenate((a, b, c ,d), axis=0)
    return a1,b1,c1,d1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3,a4,b4,c4,d4,e4,f4,a5,b5,c5,d5,a6,b6,c6,d6

def download_vgg():
    if os.path.isfile('vgg.zip') ==0:
        url = "https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip"
        # Streaming, so we can iterate over the response.
        r = requests.get(url, stream=True)
        # Total size in bytes.
        total_size = int(r.headers.get('content-length', 0));
        block_size = 1024*10
        wrote = 0
        with open('vgg.zip', 'wb') as f:
            for data in tqdm(r.iter_content(block_size), total=math.ceil(total_size//block_size) , unit='MB', unit_scale=True):
                wrote = wrote  + len(data)
                f.write(data)
        if total_size != 0 and wrote != total_size:
            print("ERROR, something went wrong in Vgg16 download")
    if (os.path.isfile('vgg.zip') ==1 and os.path.isdir('vgg')==0):
        print('Extracting vgg16.zip...')
        with zipfile.ZipFile("vgg.zip","r") as zip_ref:
            zip_ref.extractall()

def download_dataset(url,dataset):
    if os.path.isfile(dataset) ==0:
        #url = "https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip"
        # Streaming, so we can iterate over the response.
        r = requests.get(url, stream=True)
        # Total size in bytes.
        total_size = int(r.headers.get('content-length', 0));
        block_size = 1024*10
        wrote = 0
        with open(dataset, 'wb') as f:
            for data in tqdm(r.iter_content(block_size), total=math.ceil(total_size//block_size) , unit='MB', unit_scale=True):
                wrote = wrote  + len(data)
                f.write(data)
        if total_size != 0 and wrote != total_size:
            print("ERROR, something went wrong in " + dataset + " download")

    fname = dataset
    if os.path.isfile(dataset) ==1 :
        print('Extracting '+ dataset + '...')
        if (fname.endswith("tar.gz")):
            tar = tarfile.open(fname, "r:gz")
            tar.extractall()
            tar.close()
        elif (fname.endswith("tar")):
            tar = tarfile.open(fname, "r:")
            tar.extractall()
            tar.close()

def data_gen(list2,batch,cat):

    feed_arr1 = np.ones((batch,8732,10), dtype=np.int32 )
    loc_feed_arr1 = np.ones((batch,8732,4),dtype = np.float32)
    image_feed = np.ones((batch,300,300,3),dtype = np.uint8)

    for data in range(batch):
        image = list2[data][0]
        image = cv2.imread(image)
        print(image.shape)

        img_width = int(list2[data][1])
        img_height = int(list2[data][2])
        gt_array = np.zeros((len(list2[data][3]),6))
        for i in range(len(list2[data][3])):
            gt_array[i,0] = label2id(int(float(list2[data][3][i])))
            #print(type(gt_array[i,0]))
            gt_array[i,1] = float(list2[data][4][i])/img_width
            gt_array[i,2] = float(list2[data][5][i])/img_height
            gt_array[i,3] = float(list2[data][6][i])/img_width
            gt_array[i,4] = float(list2[data][7][i])/img_height
            gt_array[i,5] = data+1
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
                print('salt_papper added')
            else:
                image = image
                gt_array = gt_array
                print('salt_papper not added')

            ## Brightness
            prob_bri = random.uniform(0, 1)
            if prob_bri > 0.3:
                image,gt_array = brightness(image,gt_array)
                print('brightness added')
            else:
                image = image
                gt_array = gt_array
                print('brightness not added')

            ### Contrast
            prob_cont = random.uniform(0, 1)
            if prob_cont > 0.3:
                image,gt_array = contrast(image,gt_array)
                print('Contrast added')
            else:
                image = image
                gt_array = gt_array
                print('Contrast not added')

            #### Hue
            prob_hue = random.uniform(0, 1)
            if prob_hue > 0.3:
                image,gt_array = hue(image,gt_array)
                print('hue added')
            else:
                image = image
                gt_array = gt_array
                print('hue not added')

            ####Saturation
            prob_sat = random.uniform(0, 1)
            if prob_sat > 0.3:
                image,gt_array = saturation(image,gt_array)
                print('saturation added')
            else:
                image = image
                gt_array = gt_array
                print('saturation not added')

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
                image,gt_array = screw(image,gt_array)
                print('screw added')
            else:
                image = image
                gt_array = gt_array
                print('screw not added')
            #### rotation
            prob_rot = random.uniform(0,1)
            if prob_rot > 0:
                image,gt_array = rotation(image,gt_array)
                print('rotation added')
            else:
                image = image
                gt_array = gt_array
                print('rotation not added')
            ### Channel
            prob_Channel = random.uniform(0, 1)
            if prob_Channel > 0:
                image,gt_array = channel(image,gt_array)
                print('Channel added')
            else:
                image = image
                gt_array = gt_array
                print('Channel not added')
            #### randomplacement
            prob_place = random.uniform(0, 1)
            if prob_place > 0:
                image,gt_array = randomplacement(image,gt_array)
                print('placement added')
            else:
                image = image
                gt_array = gt_array
                print('placement not added')
             ###Resize
            image,gt_array = resize(image,gt_array)
            print(gt_array)
        else:
        ###Resize
            image,gt_array = randomplacement(image,gt_array)
            image,gt_array = resize(image,gt_array)
            print(gt_array)

        #for i in range(len(gt_array)):
        #    cv2.rectangle(image,(int(gt_array[i,1]*300),int(gt_array[i,2]*300)),(int(gt_array[i,3]*300),int(gt_array[i,4]*300)),(0,0,255),2)

        #cv2.imwrite('result_train_data_den'+str(data)+'.png', image)
        

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
        a1,b1,c1,d1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3,a4,b4,c4,d4,e4,f4,a5,b5,c5,d5,a6,b6,c6,d6 = grid_maker()
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
        data = data 

        feed_arr1[data,:,:] = np.concatenate((clas_feed_4_3_1, clas_feed_4_3_2, clas_feed_4_3_3 ,clas_feed_4_3_4, clas_feed_7_1_1, clas_feed_7_1_2, clas_feed_7_1_3, clas_feed_7_1_4, clas_feed_7_1_5, clas_feed_7_1_6, clas_feed_8_2_1, clas_feed_8_2_2, clas_feed_8_2_3, clas_feed_8_2_4, clas_feed_8_2_5, clas_feed_8_2_6, clas_feed_9_2_1, clas_feed_9_2_2, clas_feed_9_2_3, clas_feed_9_2_4, clas_feed_9_2_5, clas_feed_9_2_6, clas_feed_10_2_1, clas_feed_10_2_2, clas_feed_10_2_3, clas_feed_10_2_4, clas_feed_11_2_1, clas_feed_11_2_2, clas_feed_11_2_3, clas_feed_11_2_4), axis=0)
        loc_feed_arr1[data,:,:] = np.concatenate((local_feed_4_3_1, local_feed_4_3_2, local_feed_4_3_3 ,local_feed_4_3_4, local_feed_7_1_1, local_feed_7_1_2, local_feed_7_1_3, local_feed_7_1_4, local_feed_7_1_5, local_feed_7_1_6, local_feed_8_2_1, local_feed_8_2_2, local_feed_8_2_3, local_feed_8_2_4, local_feed_8_2_5, local_feed_8_2_6, local_feed_9_2_1, local_feed_9_2_2, local_feed_9_2_3, local_feed_9_2_4, local_feed_9_2_5, local_feed_9_2_6, local_feed_10_2_1, local_feed_10_2_2, local_feed_10_2_3, local_feed_10_2_4, local_feed_11_2_1, local_feed_11_2_2, local_feed_11_2_3, local_feed_11_2_4), axis=0)
        image_feed[data,:,:,:] = image
        if data ==0:
            gt_array_feed = gt_array
        if data >0 :
            gt_array_feed= np.concatenate((gt_array_feed,gt_array),axis = 0)

    for i in range(len(gt_array_feed)):
        gt_array_feed[i,1] = 300 * gt_array_feed[i,1]
        gt_array_feed[i,2] = 300 * gt_array_feed[i,2]
        gt_array_feed[i,3] = 300 * gt_array_feed[i,3]
        gt_array_feed[i,4] = 300 * gt_array_feed[i,4]

    anc_array_feed = np.concatenate((a1,b1,c1,d1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3,a4,b4,c4,d4,e4,f4,a5,b5,c5,d5,a6,b6,c6,d6),axis = 0)
    for i in range(8732):
        anc_array_feed[i,0] = 300 * anc_array_feed[i,0]
        anc_array_feed[i,1] = 300 * anc_array_feed[i,1]
        anc_array_feed[i,2] = 300 * anc_array_feed[i,2]
        anc_array_feed[i,3] = 300 * anc_array_feed[i,3]
        
    feed = np.concatenate((feed_arr1,loc_feed_arr1), axis=2)
    return feed, image_feed, gt_array_feed, anc_array_feed

#t1,t2,t3,t4 = image_data_gen()

def mp_data_gen():

    batch = 8

    list2 = file_input()
    a1,b1,c1,d1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3,a4,b4,c4,d4,e4,f4,a5,b5,c5,d5,a6,b6,c6,d6 = grid_maker()
    data = np.random.randint(1,100, size=8, dtype=np.int32)



    feed = np.ones( (batch,8732,24), dtype=np.float32 )
    #loc_feed_arr1 = np.ones((batch,8732,4),dtype = np.float32)
    image_feed = np.ones((batch,300,300,3),dtype = np.uint8)

    #mp.set_start_method('spawn')
    q1 = mp.Queue()
    q2 = mp.Queue()
    q3 = mp.Queue()
    q4 = mp.Queue()
    q5 = mp.Queue()
    q6 = mp.Queue()
    q7 = mp.Queue()
    q8 = mp.Queue()

    #data=0
    p1 = mp.Process(target=image_data_gen, args=(q1,data[0],list2,a1,b1,c1,d1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3,a4,b4,c4,d4,e4,f4,a5,b5,c5,d5,a6,b6,c6,d6))
    #data=1
    p2 = mp.Process(target=image_data_gen, args=(q2,data[1],list2,a1,b1,c1,d1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3,a4,b4,c4,d4,e4,f4,a5,b5,c5,d5,a6,b6,c6,d6))
    #data=2
    p3 = mp.Process(target=image_data_gen, args=(q3,data[2],list2,a1,b1,c1,d1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3,a4,b4,c4,d4,e4,f4,a5,b5,c5,d5,a6,b6,c6,d6))
    #data=3
    p4 = mp.Process(target=image_data_gen, args=(q4,data[3],list2,a1,b1,c1,d1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3,a4,b4,c4,d4,e4,f4,a5,b5,c5,d5,a6,b6,c6,d6))
    #data=4
    p5 = mp.Process(target=image_data_gen, args=(q5,data[4],list2,a1,b1,c1,d1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3,a4,b4,c4,d4,e4,f4,a5,b5,c5,d5,a6,b6,c6,d6))
    #data=5
    p6 = mp.Process(target=image_data_gen, args=(q6,data[5],list2,a1,b1,c1,d1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3,a4,b4,c4,d4,e4,f4,a5,b5,c5,d5,a6,b6,c6,d6))
    #data=6
    p7 = mp.Process(target=image_data_gen, args=(q7,data[6],list2,a1,b1,c1,d1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3,a4,b4,c4,d4,e4,f4,a5,b5,c5,d5,a6,b6,c6,d6))
    #data=7
    p8 = mp.Process(target=image_data_gen, args=(q8,data[7],list2,a1,b1,c1,d1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3,a4,b4,c4,d4,e4,f4,a5,b5,c5,d5,a6,b6,c6,d6))

    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()
    p6.start()
    p7.start()
    p8.start()

    feed[0,:,:] =q1.get()
    feed[1,:,:] =q2.get()
    feed[2,:,:] =q3.get()
    feed[3,:,:] =q4.get()
    feed[4,:,:] =q5.get()
    feed[5,:,:] =q6.get()
    feed[6,:,:] =q7.get()
    feed[7,:,:] =q8.get()

    image_feed[0,:,:,:] = q1.get()
    image_feed[1,:,:,:] = q2.get()
    image_feed[2,:,:,:] = q3.get()
    image_feed[3,:,:,:] = q4.get()
    image_feed[4,:,:,:] = q5.get()
    image_feed[5,:,:,:] = q6.get()
    image_feed[6,:,:,:] = q7.get()
    image_feed[7,:,:,:] = q8.get()

    z1 = q1.get()
    z2 = q2.get()
    z3 = q3.get()
    z4 = q4.get()
    z5 = q5.get()
    z6 = q6.get()
    z7 = q7.get()
    z8 = q8.get()

    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()
    p6.join()
    p7.join()
    p8.join()

    for i in range(len(z1)):
        z1[i,5] = 1
    for i in range(len(z2)):
        z2[i,5] = 2
    for i in range(len(z3)):
        z3[i,5] = 3
    for i in range(len(z4)):
        z4[i,5] = 4
    for i in range(len(z5)):
        z5[i,5] = 5
    for i in range(len(z6)):
        z6[i,5] = 6
    for i in range(len(z7)):
        z7[i,5] = 7
    for i in range(len(z8)):
        z8[i,5] = 8

    gt_array_feed = np.concatenate((z1,z2,z3,z4,z5,z6,z7,z8),axis = 0)
    print(gt_array_feed)

    anc_array_feed = np.concatenate((a1,b1,c1,d1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3,a4,b4,c4,d4,e4,f4,a5,b5,c5,d5,a6,b6,c6,d6),axis = 0)

    for i in range(8732):

        anc_array_feed[i,0] = 300 * anc_array_feed[i,0]
        anc_array_feed[i,1] = 300 * anc_array_feed[i,1]
        anc_array_feed[i,2] = 300 * anc_array_feed[i,2]
        anc_array_feed[i,3] = 300 * anc_array_feed[i,3]
    

    return feed,image_feed,gt_array_feed, anc_array_feed

if __name__ == '__main__':

    list_pass=[]
    cat =10
    batch =1
    ther = 0.7
    list_training_data = testMain()
    #r_no = np.random.randint(0,len(list_training_data), size=8, dtype=np.int32)
    #print(r_no)
    #r_no = 1
    r_no =[10583]
    for i in range(batch):
        random_image = list_training_data[r_no[i]]
        list_pass.append(random_image)
    #list_pass = list_training_data[r_no]
    print(list_pass)
    count=0
    count2=0
    count3 =0
    count4 =0

    feed,image_feed, gt_array_feed, anc_array_feed = data_gen(list_pass,batch,cat)
    feed1 = feed[:,:,:10]
    feed2 = feed[:,:,10:]
    for k in range(batch):
        for i in range(8732):
            for j in range(9):
                if (feed1[k,i,j] ==1):
                    count =count+1
                    #print(i)
                    #print(j)
        print(count)
        count = 0
    for k in range(batch):
        for i in range(8732):
            for j in range(4):
                if feed2[k,i,j] !=0:
                    count4 =count4+1
                    #print(i)
                    #print(j)

        print(count4)
        count4 = 0
    
    #print(count)
    #print(count2)
    #print(count3)
    
    #print(feed2[0,u,:])
                          

    print(gt_array_feed)
    mean, dict5, dict6  = A_P(gt_array_feed,feed, anc_array_feed,batch,cat)
    mean = float(mean) 
    print(mean)
    
    image_tb = image_for_tb(image_feed,dict5, dict6,batch,ther)
  

#    main()
#    for i in range(10):
#        t1, t2, t3, t4 = mp_data_gen()

#        print(t1.shape)
#        print(t2.shape)
#        print(t3.shape)
#        print(t4.shape)

