
import numpy as np
import pandas as pd   
import os
from pathlib import Path
import glob
import json
import tensorflow as tf
import tensorflow_datasets as tfds
import os
import nltk

import cv2
import matplotlib.pyplot as plt
import random
import  json

lenghtData = 0
posTrain = []
posQues = []
trainList=[]


def decodificador()
    
    with open('C:\\Users\\Admin\\Downloads\\CLEVR_v1.0\\CLEVR_v1.0_1\\questions\\CLEVR_train_questions.json') as f:
    data = json.load(f)
    if lenghtData == 0 : 
        lenghtData = len(data['questions'])
    for k in range(lenghtData):
        pos = 0
        if (lenghtData != len(data['questions'])): 
            pos = random.randrange(0, len(data['questions']) -1)
        else :
            pos = k 
        posTrain.append(pos)
        i = data['questions'][pos]
        temp=[]
       
        for path in glob.glob('C:\\Users\\Admin\\Downloads\\CLEVR_v1.0\\CLEVR_v1.0_1\\images\\train\\'+i['image_filename']): 
            temp.append(path)
        temp.append(i['question'])
        temp.append(i['answer'])
        trainList.append(temp)
    f.close()
    labels=['Path','Question','Answer']
    train_dataframe = pd.DataFrame.from_records(trainList, columns=labels)#training Dataframe 
    del(data)
    del(trainList)
    print('K')
    print(k)
    print(pos)

    k=0
    pos=0
    valList=[]
    lenghtData = 0
    with open('C:\\Users\\Admin\\Downloads\\CLEVR_v1.0\\CLEVR_v1.0_1\\questions\\CLEVR_val_questions.json') as f:
        data = json.load(f)
        if lenghtData == 0 : 
            lenghtData = len(data['questions'])
        for k in range(lenghtData):
            pos = 0
            if (lenghtData != len(data['questions'])): 
                pos = random.randrange(0, len(data['questions'])-1)
            else :
                pos = k   
        
        
            posQues.append(pos)
            i = data['questions'][pos]
            temp=[]
            for path in glob.glob('C:\\Users\\Admin\\Downloads\\CLEVR_v1.0\\CLEVR_v1.0_1\\images\\val\\'+i['image_filename']): 
                temp.append(path)
            temp.append(i['question'])
            temp.append(i['answer'])
            valList.append(temp)
    


strDiretorio = ''
with open(strDiretorio+'trainQues_2.json', 'r') as myfile:
    data=myfile.read()
obj = json.loads(data)

obj.len()

