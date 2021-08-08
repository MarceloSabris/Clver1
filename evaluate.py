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
from keras.preprocessing import text
import os
import nltk

import cv2
import matplotlib.pyplot as plt
import random
grupo = 0
grupo = grupo + 1


from os.path import isfile, join

def GravarArquivo ( data_dict,fname):
    print("gravar arquivo" + str(len(data)))
    if os.path.isfile(fname):
    # File exists
     with open(fname, 'a+') as outfile:
        outfile.seek(outfile.tell() - 2, os.SEEK_SET)
        outfile.truncate()
        outfile.write(',')
        outfile.write(json.dumps(data_dict,ensure_ascii=False, indent=4)[1:-1])
        outfile.write(']')
        outfile.close() 
    else: 
    # Create file
     with open(fname, 'w') as outfile:
        json.dump(data_dict, outfile, ensure_ascii=False, indent=4) 
        outfile.close()

print('oi')
file_to_search = 'C:\\result\\'
folders = dirs = [d for d in os.listdir(file_to_search) if os.path.isdir(os.path.join(file_to_search, d))]
print(folders)
for folder in folders:
    if 'percentage_05'  in folder:
        
        qtd = 0 
     
        vocab_set=set()#set object used to store the vocabulary
        partSplit = folder.split('_')
        perc = partSplit[1]
        FolderSource = file_to_search + '//' + folder 
        Files = [f for f in os.listdir(FolderSource) if isfile(join(FolderSource, f))]
        posQues =[]
        valList=[]
        trainList=[]
        memorys = []
        print(Files)
        for File in Files:

                if 'vocab' in File : 
                    
                    with open(FolderSource + '//' + File) as f:

                         vocabs = json.load(f)
                         for vocab in vocabs :
                            
                            vocab_set.add(vocab)
                    print('fim - vocab')

                elif 'ques' in File:     
                    with open('/home/kaggle/input/clevr-dataset/CLEVR_v1.0/questions/CLEVR_val_questions.json') as que:
                        print("qes - inicio")
                        data = json.load(que)
                        with open(FolderSource + '//' + File) as f:
                             ques = json.load(f)
                             for que in ques :
                                pos = que    
                                i = data['questions'][pos]
                                temp=[]
                                
                                for path in glob.glob('/home/kaggle/input/clevr-dataset/CLEVR_v1.0/images/val/'+i['image_filename']): 
                                    temp.append(path)
                                temp.append(i['question'])
                                temp.append(i['answer'])
                                temp.append(pos)
                                valList.append(temp)
                        f.close()
                        print("qes - fim")

                elif 'train' in File:
                    print('train')
                    temp=[]    
                    with open('/home/kaggle/input/clevr-dataset/CLEVR_v1.0/questions/CLEVR_train_questions.json') as train:
                        data = json.load(train)
                       
                        with open(FolderSource + '//' + File) as f:
                            Ftrain = json.load(f)
                            for tr in Ftrain:
                                pos = tr
                              
                                i = data['questions'][pos]
                              
                                temp=[]
                                for path in glob.glob('/home/kaggle/input/clevr-dataset/CLEVR_v1.0/images/train/'+i['image_filename']): 
                                    temp.append(path)
                                temp.append(i['question'])
                                temp.append(i['answer'])
                              
                                trainList.append(temp)
                        f.close()
                        
                    print('train -fim')
                    train.close()
                elif 'ckpt' in File and 'index' in File:
                    obj=[]
                    obj.append(perc)
                    obj.append(File[:-6])
                    memorys.append(obj)
       
        encoder=tfds.features.text.TokenTextEncoder(vocab_set)
        print("Testing the Encoder with sample questions - \n ")
        example_text=encoder.encode(trainList[1][1])
        print("Original Text = "+trainList[1][1])

        print("After Encoding = "+str(example_text))
        Traninlist=[]
       
        for memory in memorys :
            if 'weights-improvement-30' in memory:
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
                RNN_model.add(tf.keras.layers.Embedding (len(vocab_set)+1,256))
                RNN_model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256,stateful=False,return_sequences=True,recurrent_initializer='glorot_uniform')))
                RNN_model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256,stateful=False,return_sequences=True,recurrent_initializer='glorot_uniform')))
                RNN_model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(512,stateful=False,return_sequences=False,recurrent_initializer='glorot_uniform')))


                concat=tf.keras.layers.concatenate([CNN_model.output,RNN_model.output])
                dense_out=tf.keras.layers.Dense(len(vocab_set)+1,activation='softmax',name='output')(concat)

                model = tf.keras.Model(inputs=[CNN_Input,RNN_Input],
                                outputs=dense_out)
                model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',
                        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
                model.summary()

                filepath=FolderSource + '//' + memory[1]  
                print(filepath)
                model.load_weights(filepath)
                print(' ************ carregou *************')
                #generate dates to test 

                index=1
                #0 - path image 
                #1 - address 
                ListasAcertasTrainlist = []
                for contador in range(len(trainList)) :

                    im=cv2.imread(trainList[contador][0])
                    im=cv2.resize(im,(200,200))
                    q=trainList[contador][1]  
                    q=encoder.encode(q)
                    paddings = [[0, 50-tf.shape(q)[0]]]
                    q=tf.pad(q, paddings, 'CONSTANT', constant_values=0)
                    q=np.array(q)
                    im.resize(1,200,200,3)
                    q.resize(1,50)
                    ans=model.predict([im,q]) 
                    decodAns = encoder.decode([np.argmax(ans)])
                    if trainList[contador][2] != decodAns :
                        ListasAcertasTrainlist.append(0)

                    else:     
                        ListasAcertasTrainlist.append(1)
                    if (contador%2000) == 0 : 
                        GravarArquivo (ListasAcertasTrainlist,FolderSource + '/ResultPredication_Train_' +memory[1][:-5] +'.json')   
                        print("grupo" + str(grupo ) )
                        grupo = grupo + 1
                        ListasAcertasTrainlist=[]
                if (contador%2000) >= 0 :
                    GravarArquivo (ListasAcertasTrainlist,FolderSource + '/ResultPredication_Train_' +memory[1][:-5] +'.json')   
                ListasValidacaolist = []
                for contador in   range(len(valList)) :

                    im=cv2.imread(valList[contador][0])
                    im=cv2.resize(im,(200,200))
                    q=valList[contador][1]
                    q=encoder.encode(q)
                    paddings = [[0, 50-tf.shape(q)[0]]]
                    q=tf.pad(q, paddings, 'CONSTANT', constant_values=0)
                    q=np.array(q)
                    im.resize(1,200,200,3)
                    q.resize(1,50)

                    ans=model.predict([im,q]) 
                    decodAns = encoder.decode([np.argmax(ans)])
                    if valList[contador][2] != decodAns :
                        ListasValidacaolist.append(0)

                    else:     
                        ListasValidacaolist.append(1)
                    if (contador%2000) == 0 : 
                        GravarArquivo (ListasValidacaolist,FolderSource + '/ResultPredication_Val_' +memory[1][:-5] +'.json')   
                      
                        grupo = grupo +1
                        ListasValidacaolist=[]     
                if (contador%2000) >= 0 :      
                     GravarArquivo (ListasValidacaolist,FolderSource + '/ResultPredication_Val_' +memory[1][:-5] +'.json')   

    
    
print('fim')