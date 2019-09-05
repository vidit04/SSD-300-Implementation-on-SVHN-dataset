from tqdm import tqdm
import h5py
import numpy as np
import cv2
import math

class BBox:
    def __init__(self):
        self.label = ""     # Digit
        self.left = 0
        self.top = 0
        self.width = 0
        self.height = 0

class DigitStruct:
    def __init__(self):
        self.name = None    # Image file name
        self.bboxList = None # List of BBox structs

# Function for debugging
def printHDFObj(theObj, theObjName):
    isFile = isinstance(theObj, h5py.File)
    isGroup = isinstance(theObj, h5py.Group)
    isDataSet = isinstance(theObj, h5py.Dataset)
    isReference = isinstance(theObj, h5py.Reference)
    print("{}".format(theObjName))
    print("    type(): {}".format(type(theObj)))
    if isFile or isGroup or isDataSet:
        # if theObj.name != None:
        #    print "    name: {}".format(theObj.name)
        print("    id: {}".format(theObj.id))
    if isFile or isGroup:
        print("    keys: {}".format(theObj.keys()))
    if not isReference:
        print("    Len: {}".format(len(theObj)))

    if not (isFile or isGroup or isDataSet or isReference):
        print(theObj)

def readDigitStructGroup(dsFile):
    dsGroup = dsFile["digitStruct"]
    return dsGroup

#
# Reads a string from the file using its reference
#
def readString(strRef, dsFile):
    strObj = dsFile[strRef]
    str = ''.join(chr(i) for i in strObj)
    return str

#
# Reads an integer value from the file
#
def readInt(intArray, dsFile):
    intRef = intArray[0]
    isReference = isinstance(intRef, h5py.Reference)
    intVal = 0
    if isReference:
        intObj = dsFile[intRef]
        intVal = int(intObj[0])
    else: # Assuming value type
        intVal = int(intRef)
    return intVal

def yieldNextInt(intDataset, dsFile):
    for intData in intDataset:
        intVal = readInt(intData, dsFile)
        yield intVal 

def yieldNextBBox(bboxDataset, dsFile):
    for bboxArray in bboxDataset:
        bboxGroupRef = bboxArray[0]
        bboxGroup = dsFile[bboxGroupRef]
        labelDataset = bboxGroup["label"]
        leftDataset = bboxGroup["left"]
        topDataset = bboxGroup["top"]
        widthDataset = bboxGroup["width"]
        heightDataset = bboxGroup["height"]

        left = yieldNextInt(leftDataset, dsFile)
        top = yieldNextInt(topDataset, dsFile)
        width = yieldNextInt(widthDataset, dsFile)
        height = yieldNextInt(heightDataset, dsFile)

        bboxList = []

        for label in yieldNextInt(labelDataset, dsFile):
            bbox = BBox()
            bbox.label = label
            bbox.left = next(left)
            bbox.top = next(top)
            bbox.width = next(width)
            bbox.height = next(height)
            bboxList.append(bbox)

        yield bboxList

def yieldNextFileName(nameDataset, dsFile):
    for nameArray in nameDataset:
        nameRef = nameArray[0]
        name = readString(nameRef, dsFile)
        yield name

# dsFile = h5py.File('../data/gsvhn/train/digitStruct.mat', 'r')
def yieldNextDigitStruct(dsFileName):
    dsFile = h5py.File(dsFileName, 'r')
    dsGroup = readDigitStructGroup(dsFile)
    nameDataset = dsGroup["name"]
    bboxDataset = dsGroup["bbox"]

    bboxListIter = yieldNextBBox(bboxDataset, dsFile)
    for name in yieldNextFileName(nameDataset, dsFile):
        bboxList = next(bboxListIter)
        obj = DigitStruct()
        obj.name = name
        obj.bboxList = bboxList
        yield obj

def testMain(dataset):

    dsFileName = dataset + '/digitStruct.mat'
    testCounter = 0
    list2 = []
    label_name=[]
    x_cen_box=[]
    y_cen_box=[]
    width_box=[]
    height_box=[]
    mat_data = h5py.File(dataset + '/digitStruct.mat')
    size = mat_data['/digitStruct/name'].size
    total_size = size
    block_size = 1
    for dsObj in tqdm(yieldNextDigitStruct(dsFileName), total=math.ceil(total_size//block_size) , unit=' Images', unit_scale=True):
        # testCounter += 1
        #print(dsObj.name)
        file_n = dataset + '/' + dsObj.name
        image = cv2.imread (file_n)
        img_height = image.shape[0]
        #print(img_height)
        img_width = image.shape[1]
        #print(img_width)

        for bbox in dsObj.bboxList:

            x = bbox.left + (bbox.width)/2
            y = bbox.top + (bbox.height)/2
            w = bbox.width
            h = bbox.height
            label_n= bbox.label
            
            #print("    {}:{},{},{},{}".format(
#                bbox.label, bbox.left, bbox.top, bbox.width, bbox.height))

            label_name.append(label_n)
            x_cen_box.append(x)
            y_cen_box.append(y)
            width_box.append(w)
            height_box.append(h)
            
            Data_img = [file_n,img_width, img_height , label_name, x_cen_box, y_cen_box, width_box, height_box]

        list2.append(Data_img)
        label_name=[]
        x_cen_box=[]
        y_cen_box=[]
        width_box=[]
        height_box=[]
        if testCounter >= 5:
            break
    #print(list2)
    #print(list2)
    #print(list2[0][0])
    #print(list2[0][1])
    #print(list2[0][2])
    #print(list2[0][3][0])
    #print(list2[0][4][0])
    #print(type(list2[0][0]))
    #print(type(list2[0][1]))
    #print(type(list2[0][2]))
    #print(type(list2[0][3][0]))
    #print(type(list2[0][4][0]))

    return list2

if __name__ == "__main__":
    dataset = 'train'

    list2 = testMain(dataset)
    
    print(list2[1])
    r_no = np.random.randint(0,len(list2), dtype=np.int32)
    print(r_no)
    #list_pass = list_training_data[r_no]
    image = cv2.imread(list2[r_no][0])
    print(image.shape)
    for i in range(len(list2[r_no][3])):
        print(list2[r_no][6][i])
        print(list2[r_no][7][i])

        xmin = list2[r_no][4][i] - list2[r_no][6][i]/2
        ymin = list2[r_no][5][i] - list2[r_no][7][i]/2
        xmax = list2[r_no][4][i] + list2[r_no][6][i]/2
        ymax = list2[r_no][5][i] + list2[r_no][7][i]/2
        
        cv2.rectangle(image,(int(xmin),int(ymin)),(int(xmax),int(ymax)),(0,0,255),1)
    cv2.imwrite('result_placement.png', image)
    print(list2[r_no])
    
  



