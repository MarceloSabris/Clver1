
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
k=0
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

print('K')
print(k)
print(pos)
f.close()
val_dataframe = pd.DataFrame.from_records(valList, columns=labels)#validation Dataframe
del(data)
del(valList)
val_dataframe.head()


vocab_set=set()#set object used to store the vocabulary

tokenizer = tfds.deprecated.text.Tokenizer()
for i in val_dataframe['Question']:
    vocab_set.update(tokenizer.tokenize(i))
for i in train_dataframe['Question']:
    vocab_set.update(tokenizer.tokenize(i))
for i in val_dataframe['Answer']:
    vocab_set.update(tokenizer.tokenize(i))
for i in train_dataframe['Answer']:
    vocab_set.update(tokenizer.tokenize(i))

BATCH_SIZE=8
IMG_SIZE=(200,200)
with open('vocab_set_'+str(lenghtData) + '.json', 'w', encoding='utf-8') as f:
    json.dump(list(vocab_set), f, ensure_ascii=False, indent=4)


encoder=tfds.deprecated.text.TokenTextEncoder(vocab_set)
index=1
print("Testing the Encoder with sample questions - \n ")
example_text=encoder.encode(train_dataframe['Question'][index])
print("Original Text = "+train_dataframe['Question'][index])
print("After Encoding = "+str(example_text))



CNN_Input=tf.keras.layers.Input(shape=(200,200,3),name='image_input')

mobilenetv2=tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=(200,200,3), alpha=1.0, include_top=False,
                                                      weights='imagenet', input_tensor=CNN_Input)

CNN_model=tf.keras.models.Sequential()
CNN_model.add(CNN_Input)
CNN_model.add(mobilenetv2)
CNN_model.add(tf.keras.layers.GlobalAveragePooling2D())






#Creating the RNN model for text processing
RNN_model=tf.keras.models.Sequential()
RNN_Input=tf.keras.layers.Input(shape=(50),name='text_input')
RNN_model.add(RNN_Input)
RNN_model.add(tf.keras.layers.Embedding (101,256))
RNN_model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256,stateful=False,return_sequences=True,recurrent_initializer='glorot_uniform')))
RNN_model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256,stateful=False,return_sequences=True,recurrent_initializer='glorot_uniform')))
RNN_model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(512,stateful=False,return_sequences=False,recurrent_initializer='glorot_uniform')))


concat=tf.keras.layers.concatenate([CNN_model.output,RNN_model.output])
dense_out=tf.keras.layers.Dense(100,activation='softmax',name='output')(concat)

model = tf.keras.Model(inputs=[CNN_Input,RNN_Input],
                    outputs=dense_out)
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
model.summary()

filepath='weights.01.ckpt'

model.load_weights(filepath)







