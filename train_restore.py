import numpy as np
import tensorflow as tf
import multiprocessing as mp
from ap import A_P,covert, image_for_tb
from list_train import image_data_gen, image_data_gen_valid
from train_read import testMain
from train_data_gen import grid_maker

##### To run Train_restore.py
##### Update latest metadata file name
##### Update step value to same as metadata file name.


def multi_gen(list_pass,batch,cat):

    a1,b1,c1,d1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3,a4,b4,c4,d4,e4,f4,a5,b5,c5,d5,a6,b6,c6,d6 = grid_maker()
    feed = np.ones( (batch,8732,cat+4), dtype=np.float32 )
    image_feed = np.ones((batch,300,300,3),dtype = np.uint8)

    q1 = mp.Queue()
    q2 = mp.Queue()
    q3 = mp.Queue()
    q4 = mp.Queue()
    q5 = mp.Queue()
    q6 = mp.Queue()
    q7 = mp.Queue()
    q8 = mp.Queue()
    q9 = mp.Queue()
    q10 = mp.Queue()
    q11 = mp.Queue()
    q12 = mp.Queue()
    q13 = mp.Queue()
    q14 = mp.Queue()
    q15 = mp.Queue()
    q16 = mp.Queue()
    q17 = mp.Queue()
    q18 = mp.Queue()
    q19 = mp.Queue()
    q20 = mp.Queue()
    q21 = mp.Queue()
    q22 = mp.Queue()
    q23 = mp.Queue()
    q24 = mp.Queue()
    q25 = mp.Queue()
    q26 = mp.Queue()
    q27 = mp.Queue()
    q28 = mp.Queue()
    q29 = mp.Queue()
    q30 = mp.Queue()
    q31 = mp.Queue()
    q32 = mp.Queue()


    p1 = mp.Process(target=image_data_gen, args=(q1,list_pass[0],cat,a1,b1,c1,d1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3,a4,b4,c4,d4,e4,f4,a5,b5,c5,d5,a6,b6,c6,d6))
    p2 = mp.Process(target=image_data_gen, args=(q2,list_pass[1],cat,a1,b1,c1,d1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3,a4,b4,c4,d4,e4,f4,a5,b5,c5,d5,a6,b6,c6,d6))
    p3 = mp.Process(target=image_data_gen, args=(q3,list_pass[2],cat,a1,b1,c1,d1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3,a4,b4,c4,d4,e4,f4,a5,b5,c5,d5,a6,b6,c6,d6))
    p4 = mp.Process(target=image_data_gen, args=(q4,list_pass[3],cat,a1,b1,c1,d1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3,a4,b4,c4,d4,e4,f4,a5,b5,c5,d5,a6,b6,c6,d6))
    p5 = mp.Process(target=image_data_gen, args=(q5,list_pass[4],cat,a1,b1,c1,d1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3,a4,b4,c4,d4,e4,f4,a5,b5,c5,d5,a6,b6,c6,d6))
    p6 = mp.Process(target=image_data_gen, args=(q6,list_pass[5],cat,a1,b1,c1,d1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3,a4,b4,c4,d4,e4,f4,a5,b5,c5,d5,a6,b6,c6,d6))
    p7 = mp.Process(target=image_data_gen, args=(q7,list_pass[6],cat,a1,b1,c1,d1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3,a4,b4,c4,d4,e4,f4,a5,b5,c5,d5,a6,b6,c6,d6))
    p8 = mp.Process(target=image_data_gen, args=(q8,list_pass[7],cat,a1,b1,c1,d1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3,a4,b4,c4,d4,e4,f4,a5,b5,c5,d5,a6,b6,c6,d6))
    p9 = mp.Process(target=image_data_gen, args=(q9,list_pass[8],cat,a1,b1,c1,d1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3,a4,b4,c4,d4,e4,f4,a5,b5,c5,d5,a6,b6,c6,d6))
    p10 = mp.Process(target=image_data_gen, args=(q10,list_pass[9],cat,a1,b1,c1,d1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3,a4,b4,c4,d4,e4,f4,a5,b5,c5,d5,a6,b6,c6,d6))
    p11 = mp.Process(target=image_data_gen, args=(q11,list_pass[10],cat,a1,b1,c1,d1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3,a4,b4,c4,d4,e4,f4,a5,b5,c5,d5,a6,b6,c6,d6))
    p12 = mp.Process(target=image_data_gen, args=(q12,list_pass[11],cat,a1,b1,c1,d1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3,a4,b4,c4,d4,e4,f4,a5,b5,c5,d5,a6,b6,c6,d6))
    p13 = mp.Process(target=image_data_gen, args=(q13,list_pass[12],cat,a1,b1,c1,d1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3,a4,b4,c4,d4,e4,f4,a5,b5,c5,d5,a6,b6,c6,d6))
    p14 = mp.Process(target=image_data_gen, args=(q14,list_pass[13],cat,a1,b1,c1,d1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3,a4,b4,c4,d4,e4,f4,a5,b5,c5,d5,a6,b6,c6,d6))
    p15 = mp.Process(target=image_data_gen, args=(q15,list_pass[14],cat,a1,b1,c1,d1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3,a4,b4,c4,d4,e4,f4,a5,b5,c5,d5,a6,b6,c6,d6))
    p16 = mp.Process(target=image_data_gen, args=(q16,list_pass[15],cat,a1,b1,c1,d1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3,a4,b4,c4,d4,e4,f4,a5,b5,c5,d5,a6,b6,c6,d6))
    p17 = mp.Process(target=image_data_gen, args=(q17,list_pass[16],cat,a1,b1,c1,d1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3,a4,b4,c4,d4,e4,f4,a5,b5,c5,d5,a6,b6,c6,d6))
    p18 = mp.Process(target=image_data_gen, args=(q18,list_pass[17],cat,a1,b1,c1,d1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3,a4,b4,c4,d4,e4,f4,a5,b5,c5,d5,a6,b6,c6,d6))
    p19 = mp.Process(target=image_data_gen, args=(q19,list_pass[18],cat,a1,b1,c1,d1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3,a4,b4,c4,d4,e4,f4,a5,b5,c5,d5,a6,b6,c6,d6))
    p20 = mp.Process(target=image_data_gen, args=(q20,list_pass[19],cat,a1,b1,c1,d1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3,a4,b4,c4,d4,e4,f4,a5,b5,c5,d5,a6,b6,c6,d6))
    p21 = mp.Process(target=image_data_gen, args=(q21,list_pass[20],cat,a1,b1,c1,d1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3,a4,b4,c4,d4,e4,f4,a5,b5,c5,d5,a6,b6,c6,d6))
    p22 = mp.Process(target=image_data_gen, args=(q22,list_pass[21],cat,a1,b1,c1,d1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3,a4,b4,c4,d4,e4,f4,a5,b5,c5,d5,a6,b6,c6,d6))
    p23 = mp.Process(target=image_data_gen, args=(q23,list_pass[22],cat,a1,b1,c1,d1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3,a4,b4,c4,d4,e4,f4,a5,b5,c5,d5,a6,b6,c6,d6))
    p24 = mp.Process(target=image_data_gen, args=(q24,list_pass[23],cat,a1,b1,c1,d1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3,a4,b4,c4,d4,e4,f4,a5,b5,c5,d5,a6,b6,c6,d6))
    p25 = mp.Process(target=image_data_gen, args=(q25,list_pass[24],cat,a1,b1,c1,d1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3,a4,b4,c4,d4,e4,f4,a5,b5,c5,d5,a6,b6,c6,d6))
    p26 = mp.Process(target=image_data_gen, args=(q26,list_pass[25],cat,a1,b1,c1,d1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3,a4,b4,c4,d4,e4,f4,a5,b5,c5,d5,a6,b6,c6,d6))
    p27 = mp.Process(target=image_data_gen, args=(q27,list_pass[26],cat,a1,b1,c1,d1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3,a4,b4,c4,d4,e4,f4,a5,b5,c5,d5,a6,b6,c6,d6))
    p28 = mp.Process(target=image_data_gen, args=(q28,list_pass[27],cat,a1,b1,c1,d1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3,a4,b4,c4,d4,e4,f4,a5,b5,c5,d5,a6,b6,c6,d6))
    p29 = mp.Process(target=image_data_gen, args=(q29,list_pass[28],cat,a1,b1,c1,d1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3,a4,b4,c4,d4,e4,f4,a5,b5,c5,d5,a6,b6,c6,d6))
    p30 = mp.Process(target=image_data_gen, args=(q30,list_pass[29],cat,a1,b1,c1,d1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3,a4,b4,c4,d4,e4,f4,a5,b5,c5,d5,a6,b6,c6,d6))
    p31 = mp.Process(target=image_data_gen, args=(q31,list_pass[30],cat,a1,b1,c1,d1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3,a4,b4,c4,d4,e4,f4,a5,b5,c5,d5,a6,b6,c6,d6))
    p32 = mp.Process(target=image_data_gen, args=(q32,list_pass[31],cat,a1,b1,c1,d1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3,a4,b4,c4,d4,e4,f4,a5,b5,c5,d5,a6,b6,c6,d6))

    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()
    p6.start()
    p7.start()
    p8.start()
    p9.start()
    p10.start()
    p11.start()
    p12.start()
    p13.start()
    p14.start()
    p15.start()
    p16.start()
    p17.start()
    p18.start()
    p19.start()
    p20.start()
    p21.start()
    p22.start()
    p23.start()
    p24.start()
    p25.start()
    p26.start()
    p27.start()
    p28.start()
    p29.start()
    p30.start()
    p31.start()
    p32.start()

    
    feed[0,:,:] =q1.get()
    feed[1,:,:] =q2.get()
    feed[2,:,:] =q3.get()
    feed[3,:,:] =q4.get()
    feed[4,:,:] =q5.get()
    feed[5,:,:] =q6.get()
    feed[6,:,:] =q7.get()
    feed[7,:,:] =q8.get()
    feed[8,:,:] =q9.get()
    feed[9,:,:] =q10.get()
    feed[10,:,:] =q11.get()
    feed[11,:,:] =q12.get()
    feed[12,:,:] =q13.get()
    feed[13,:,:] =q14.get()
    feed[14,:,:] =q15.get()
    feed[15,:,:] =q16.get()
    feed[16,:,:] =q17.get()
    feed[17,:,:] =q18.get()
    feed[18,:,:] =q19.get()
    feed[19,:,:] =q20.get()
    feed[20,:,:] =q21.get()
    feed[21,:,:] =q22.get()
    feed[22,:,:] =q23.get()
    feed[23,:,:] =q24.get()
    feed[24,:,:] =q25.get()
    feed[25,:,:] =q26.get()
    feed[26,:,:] =q27.get()
    feed[27,:,:] =q28.get()
    feed[28,:,:] =q29.get()
    feed[29,:,:] =q30.get()
    feed[30,:,:] =q31.get()
    feed[31,:,:] =q32.get()

    image_feed[0,:,:,:] = q1.get()
    image_feed[1,:,:,:] = q2.get()
    image_feed[2,:,:,:] = q3.get()
    image_feed[3,:,:,:] = q4.get()
    image_feed[4,:,:,:] = q5.get()
    image_feed[5,:,:,:] = q6.get()
    image_feed[6,:,:,:] = q7.get()
    image_feed[7,:,:,:] = q8.get()
    image_feed[8,:,:,:] = q9.get()
    image_feed[9,:,:,:] = q10.get()
    image_feed[10,:,:,:] = q11.get()
    image_feed[11,:,:,:] = q12.get()
    image_feed[12,:,:,:] = q13.get()
    image_feed[13,:,:,:] = q14.get()
    image_feed[14,:,:,:] = q15.get()
    image_feed[15,:,:,:] = q16.get()
    image_feed[16,:,:,:] = q17.get()
    image_feed[17,:,:,:] = q18.get()
    image_feed[18,:,:,:] = q19.get()
    image_feed[19,:,:,:] = q20.get()
    image_feed[20,:,:,:] = q21.get()
    image_feed[21,:,:,:] = q22.get()
    image_feed[22,:,:,:] = q23.get()
    image_feed[23,:,:,:] = q24.get()
    image_feed[24,:,:,:] = q25.get()
    image_feed[25,:,:,:] = q26.get()
    image_feed[26,:,:,:] = q27.get()
    image_feed[27,:,:,:] = q28.get()
    image_feed[28,:,:,:] = q29.get()
    image_feed[29,:,:,:] = q30.get()
    image_feed[30,:,:,:] = q31.get()
    image_feed[31,:,:,:] = q32.get()

    z1 = q1.get()
    z2 = q2.get()
    z3 = q3.get()
    z4 = q4.get()
    z5 = q5.get()
    z6 = q6.get()
    z7 = q7.get()
    z8 = q8.get()
    z9 = q9.get()
    z10 = q10.get()
    z11 = q11.get()
    z12 = q12.get()
    z13 = q13.get()
    z14 = q14.get()
    z15 = q15.get()
    z16 = q16.get()
    z17 = q17.get()
    z18 = q18.get()
    z19 = q19.get()
    z20 = q20.get()
    z21 = q21.get()
    z22 = q22.get()
    z23 = q23.get()
    z24 = q24.get()
    z25 = q25.get()
    z26 = q26.get()
    z27 = q27.get()
    z28 = q28.get()
    z29 = q29.get()
    z30 = q30.get()
    z31 = q31.get()
    z32 = q32.get()

    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()
    p6.join()
    p7.join()
    p8.join()
    p9.join()
    p10.join()
    p11.join()
    p12.join()
    p13.join()
    p14.join()
    p15.join()
    p16.join()
    p17.join()
    p18.join()
    p19.join()
    p20.join()
    p21.join()
    p22.join()
    p23.join()
    p24.join()
    p25.join()
    p26.join()
    p27.join()
    p28.join()
    p29.join()
    p30.join()
    p31.join()
    p32.join()

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
    for i in range(len(z9)):
        z9[i,5] = 9
    for i in range(len(z10)):
        z10[i,5] = 10
    for i in range(len(z11)):
        z11[i,5] = 11
    for i in range(len(z12)):
        z12[i,5] = 12
    for i in range(len(z13)):
        z13[i,5] = 13
    for i in range(len(z14)):
        z14[i,5] = 14
    for i in range(len(z15)):
        z15[i,5] = 15
    for i in range(len(z16)):
        z16[i,5] = 16
    for i in range(len(z17)):
        z17[i,5] = 17
    for i in range(len(z18)):
        z18[i,5] = 18
    for i in range(len(z19)):
        z19[i,5] = 19
    for i in range(len(z20)):
        z20[i,5] = 20
    for i in range(len(z21)):
        z21[i,5] = 21
    for i in range(len(z22)):
        z22[i,5] = 22
    for i in range(len(z23)):
        z23[i,5] = 23
    for i in range(len(z24)):
        z24[i,5] = 24
    for i in range(len(z25)):
        z25[i,5] = 25
    for i in range(len(z26)):
        z26[i,5] = 26
    for i in range(len(z27)):
        z27[i,5] = 27
    for i in range(len(z28)):
        z28[i,5] = 28
    for i in range(len(z29)):
        z29[i,5] = 29
    for i in range(len(z30)):
        z30[i,5] = 30
    for i in range(len(z31)):
        z31[i,5] = 31
    for i in range(len(z32)):
        z32[i,5] = 32

    gt_array_feed = np.concatenate((z1,z2,z3,z4,z5,z6,z7,z8,z9,z10,z11,z12,z13,z14,z15,z16,z17,z18,z19,z20,z21,z22,z23,z24,z25,z26,z27,z28,z29,z30,z31,z32),axis = 0)
    #print(gt_array_feed)

    anc_array_feed = np.concatenate((a1,b1,c1,d1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3,a4,b4,c4,d4,e4,f4,a5,b5,c5,d5,a6,b6,c6,d6),axis = 0)

    for i in range(8732):

        anc_array_feed[i,0] = 300 * anc_array_feed[i,0]
        anc_array_feed[i,1] = 300 * anc_array_feed[i,1]
        anc_array_feed[i,2] = 300 * anc_array_feed[i,2]
        anc_array_feed[i,3] = 300 * anc_array_feed[i,3]

    return feed,image_feed, gt_array_feed, anc_array_feed

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

    print('Reading training dataset...')
    dataset_1 = 'train'
    list_training_data = testMain(dataset_1)
    print('Reading extra training dataset...')
    dataset_2 = 'extra'
    list_extra_data = testMain(dataset_2)
    list_training_data = list_training_data + list_extra_data
    print('Reading test dataset...')
    dataset_3 = 'test'
    list_valid_data = testMain(dataset_3)
    epochs = 10
    batch = 32
    batch_valid = batch

    sess = tf.Session()
    new_saver = tf.train.import_meta_graph('checkpoint/model.ckpt-40.meta')  #### Update metadata file name
    new_saver.restore(sess, tf.train.latest_checkpoint('./checkpoint'))
    #op = sess.graph.get_operations()
    #[print(m.values()) for m in op]
    #print(m.value())

    #for m in op:
        #print (m.name)

    image_input = sess.graph.get_tensor_by_name('image_input:0')
    label_array = sess.graph.get_tensor_by_name('label_array:0')
    global_step = sess.graph.get_tensor_by_name('global_step:0')
    mAP_placeholder = sess.graph.get_tensor_by_name('mAP_placeholder:0')
    total_loss_placeholder = sess.graph.get_tensor_by_name('total_loss_placeholder:0')
    binary_loss_placeholder = sess.graph.get_tensor_by_name('binary_loss_placeholder:0')
    local_loss_placeholder = sess.graph.get_tensor_by_name('local_loss_placeholder:0')
    l2_reg_loss_placeholder  = sess.graph.get_tensor_by_name('l2_reg_loss_placeholder:0')
    image_placeholder = sess.graph.get_tensor_by_name('image_placeholder:0')
    image_placeholder2 = sess.graph.get_tensor_by_name('image_placeholder2:0')
    output_softmax = sess.graph.get_tensor_by_name('output_softmax:0')
    

    binary_loss = sess.graph.get_tensor_by_name('binary_loss:0')
    local_loss = sess.graph.get_tensor_by_name('local_loss:0')
    l2_reg_loss = sess.graph.get_tensor_by_name('l2_reg_loss:0')

    loss_total = sess.graph.get_tensor_by_name('loss:0')
    Loss_min = sess.graph.get_operation_by_name('Optimizer')

    first_summary = sess.graph.get_tensor_by_name('mAPs:0')
    loss_total_summary = sess.graph.get_tensor_by_name('Total_Loss_summary:0')
    l2_reg_summary = sess.graph.get_tensor_by_name('L2_Loss_summary:0')
    local_loss_summary = sess.graph.get_tensor_by_name('Local_Loss_summary:0')
    binary_loss_summary = sess.graph.get_tensor_by_name('Binary_Loss_summary:0')
    image_summary1 = sess.graph.get_tensor_by_name('image_summary_training:0')
    image_summary2 = sess.graph.get_tensor_by_name('image_summary_Validation:0')

    #l2_reg_loss_placeholder  = sess.graph.get_tensor_by_name('l2_reg_loss_placeholder:0')
    #image_placeholder = sess.graph.get_tensor_by_name('image_placeholder:0')
    #image_placeholder2 = sess.graph.get_tensor_by_name('image_placeholder2:0')
    #output_softmax = sess.graph.get_tensor_by_name('output_softmax:0')
    

 
    saver= tf.train.Saver(max_to_keep=20)
    save_path="checkpoint/"
    model_save = save_path+"model.ckpt"

    #init=initialize_uninitialized_variables()


    #init = tf.global_variables_initializer()
    #sess = tf.Session()
    #sess.run(init)

    ###### writers
    writer1 = tf.summary.FileWriter('./graphs/log1', sess.graph)
    writer2 = tf.summary.FileWriter('./graphs/log2',sess.graph)
    
    step = 40  ##### Update Value same as of metadata file number
    cat = 10
    ther = 0.50

    for m in range(step+1,6000000,1):
        list_pass=[]
        #feed,image_feed = image_data_gen()
        r_no = np.random.randint(0,len(list_training_data), size =batch , dtype=np.int32)
        #r_no = 0
        for j in range(batch):
            random_image_train = list_training_data[r_no[j]]
            list_pass.append(random_image_train)
        #list_pass = list_training_data[r_no]

        feed,image_feed, gt_array_feed, anc_array_feed = multi_gen(list_pass,batch,cat)
        #sess.run(Loss_min, feed_dict = {image_input : image_feed, label_array:feed })
        _,boxes, final_loss, binary_train,local_train,l2_reg_train = sess.run([Loss_min,output_softmax,loss_total,binary_loss,local_loss,l2_reg_loss], feed_dict = {image_input : image_feed, label_array:feed, global_step: m })

        if m%20==0:
            mean, dict5, dict6  = A_P(gt_array_feed,boxes, anc_array_feed,batch,cat)
            mean = float(mean)
            #print(mean)

            summary1= sess.run(first_summary,feed_dict = {mAP_placeholder : mean})
            writer1.add_summary(summary1,global_step=m)

            summary2= sess.run(loss_total_summary,feed_dict = {total_loss_placeholder : final_loss})
            writer1.add_summary(summary2,global_step=m)

            summary3 = sess.run(l2_reg_summary,feed_dict = {l2_reg_loss_placeholder : l2_reg_train})
            writer1.add_summary(summary3,global_step=m)

            summary4 = sess.run(local_loss_summary,feed_dict = {local_loss_placeholder : local_train})
            writer1.add_summary(summary4,global_step=m)

            summary5 = sess.run(binary_loss_summary,feed_dict = {binary_loss_placeholder : binary_train})
            writer1.add_summary(summary5,global_step=m)

        


        if m%100 ==0:
            image_tb = image_for_tb(image_feed,dict5, dict6,batch,ther)

            summary6 = sess.run(image_summary1,feed_dict = {image_placeholder : image_tb })
            writer1.add_summary(summary6,global_step=m)

        if  m%20==0:
            saver.save(sess,model_save,global_step=m)

        writer1.flush()

        print('Total Training Loss during training at step {:5d} is {:g} '.format(m, final_loss)) 

        if m%100==0:

            list_pass_valid = []
            r_no_v = np.random.randint(0,len(list_valid_data),size = batch, dtype=np.int32)
            for k in range(batch):
                random_image = list_valid_data[r_no_v[k]]
                list_pass_valid.append(random_image)

        #list_pass_valid = list_valid_data[r_no_v]
    
            feed_valid,image_feed_valid, gt_array_feed_valid, anc_array_feed_valid = multi_gen_valid(list_pass_valid,batch_valid,cat)
            boxes_valid,final_loss_valid, binary_train_valid,local_train_valid,l2_reg_train_valid = sess.run([output_softmax,loss_total,binary_loss,local_loss,l2_reg_loss], feed_dict = {image_input : image_feed_valid, label_array:feed_valid, global_step: m })
            mean_valid, dict5_valid, dict6_valid = A_P(gt_array_feed_valid,boxes_valid, anc_array_feed_valid,batch_valid,cat)
    
            mean_valid = float(mean_valid)
            #print(mean_valid)
    
            #image_tb_valid = image_for_tb(image_feed_valid,dict5_valid, dict6_valid,batch_valid,ther)
    
            summary1= sess.run(first_summary,feed_dict = {mAP_placeholder : mean_valid})
            writer2.add_summary(summary1,global_step=m)
  
            summary2= sess.run(loss_total_summary,feed_dict = {total_loss_placeholder : final_loss_valid})
            writer2.add_summary(summary2,global_step=m)
  
            summary3 = sess.run(l2_reg_summary,feed_dict = {l2_reg_loss_placeholder : l2_reg_train_valid })
            writer2.add_summary(summary3,global_step=m)
  
            summary4 = sess.run(local_loss_summary,feed_dict = {local_loss_placeholder : local_train_valid })
            writer2.add_summary(summary4,global_step=m)
  
            summary5 = sess.run(binary_loss_summary,feed_dict = {binary_loss_placeholder : binary_train_valid})
            writer2.add_summary(summary5,global_step=m)
  
            #summary6 = sess.run(image_summary2,feed_dict = {image_placeholder2 : image_tb_valid })
            #writer2.add_summary(summary6,global_step=i)

        if m%100==0:
            image_tb_valid = image_for_tb(image_feed_valid,dict5_valid, dict6_valid,batch_valid,ther)

            summary6 = sess.run(image_summary2,feed_dict = {image_placeholder2 : image_tb_valid })
            writer2.add_summary(summary6,global_step=m)

        if m%100==0:
            writer2.flush()
            print('Total Validation Loss during training at step {:5d} is {:g} '.format(m, final_loss_valid)) 
        #print (i)
        #print(sess.run(loss_total))
