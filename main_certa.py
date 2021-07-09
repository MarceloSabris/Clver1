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


physical_devices = tf.config.list_physical_devices('GPU') 
print("Number of GPUs :", len(physical_devices)) 
print("Tensorflow GPU :",tf.test.is_built_with_cuda())


if len(physical_devices)>0:
    device="/GPU:0"
else:
    device="/CPU:0"
listlenghData = [1,40,80,100]
for LenghCoguiniccao in listlenghData:

    valList=[]
    trainList=[]
    data=[]
    lista=[]

    #train_dataframe and val_dataframe stores the path to the images and respective questions and answers
    trainList=[]
    with open('C:\\Users\\Admin\\Downloads\\CLEVR_v1.0\\CLEVR_v1.0_1\\questions\\CLEVR_train_questions.json') as f:
        data = json.load(f)
        qtdData = len(data['questions'])
        lenghtData = int(qtdData * (LenghCoguiniccao/100))

        for k in range(lenghtData):
            q = 0 
            while  (q in lista):
               q= random.randrange(1, qtdData-1, 3)
            lista.append(q)
           
            
            i = data['questions'][q]
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

    valList=[] 


    with open('C:\\Users\\Admin\\Downloads\\CLEVR_v1.0\\CLEVR_v1.0_1\\questions\\CLEVR_val_questions.json') as f:
        data = json.load(f)
        for k in range(lenghtData):
            i = data['questions'][random.randrange(20, 5000, 3)]
            temp=[]
            for path in glob.glob('C:\\Users\\Admin\\Downloads\\CLEVR_v1.0\\CLEVR_v1.0_1\\images\\val\\'+i['image_filename']): 
                temp.append(path)
            temp.append(i['question'])
            temp.append(i['answer'])
            valList.append(temp)

    f.close()
    val_dataframe = pd.DataFrame.from_records(valList, columns=labels)#validation Dataframe
    del(data)
    del(valList)
    val_dataframe.head()





    vocab_set=set()#set object used to store the vocabulary

    tokenizer = tfds.deprecated.text.Tokenizer()
    for val in val_dataframe['Question']:
        vocab_set.update(tokenizer.tokenize(val))
    for val in train_dataframe['Question']:
        vocab_set.update(tokenizer.tokenize(val))
    for val in val_dataframe['Answer']:
        vocab_set.update(tokenizer.tokenize(val))
    for val in train_dataframe['Answer']:
        vocab_set.update(tokenizer.tokenize(val))

    encoder=tfds.deprecated.text.TokenTextEncoder(vocab_set)
    index=14
    print("Testing the Encoder with sample questions - \n ")
    example_text=encoder.encode(train_dataframe['Question'][index])
    print("Original Text = "+train_dataframe['Question'][index])
    print("After Encoding = "+str(example_text))

    #ver melhor 
    BATCH_SIZE=8
    IMG_SIZE=(200,200)
    #Function that uses the encoder created to encode the input question and answer string
    def encode_fn(text):
        return np.array(encoder.encode(text.numpy()))
    #Function to load and decode the image from the file paths in the dataframe and use the encoder function
    def preprocess(ip,ans):
        img,ques=ip#ip is a list containing image paths and questions
        img=tf.io.read_file(img)
        img=tf.image.decode_jpeg(img,channels=3)
        img=tf.image.resize(img,IMG_SIZE)
        img=tf.math.divide(img, 255)#The image has been loaded , decoded and resized 
        
        #The question string is converted to encoded list with fixed size of 50 with padding with 0 value
        ques=tf.py_function(encode_fn,inp=[ques],Tout=tf.int32)
        paddings = [[0, 50-tf.shape(ques)[0]]]
        ques = tf.pad(ques, paddings, 'CONSTANT', constant_values=0)
        ques.set_shape([50])#Explicit shape must be defined in order to create the Input pipeline
        
        #The Answer is also encoded 
        ans=tf.py_function(encode_fn,inp=[ans],Tout=tf.int32)
        ans.set_shape([1])
        
        return (img,ques),ans
        
    def create_pipeline(dataframe):
        raw_df=tf.data.Dataset.from_tensor_slices(((dataframe['Path'],dataframe['Question']),dataframe['Answer']))
        df=raw_df.map(preprocess)#Preprocessing function is applied to the dataset
        df=df.batch(BATCH_SIZE)#The dataset is batched
        return df

    #Creating the CNN model for image processing


    CNN_Input=tf.keras.layers.Input(shape=(200,200,3),name='image_input')

    mobilenetv2=tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=(200,200,3), alpha=1.0, include_top=False,
                                                        weights='imagenet', input_tensor=CNN_Input)

    #sequencial 
    CNN_model=tf.keras.models.Sequential()
    CNN_model.add(CNN_Input)
    CNN_model.add(mobilenetv2)
    #ver melhor a rede o por que 
    CNN_model.add(tf.keras.layers.GlobalAveragePooling2D())


    #The training and validation Dataset objects are created
    train_dataset=create_pipeline(train_dataframe)
    validation_dataset=create_pipeline(val_dataframe)

    #Creating the RNN model for text processing
    RNN_model=tf.keras.models.Sequential()
    RNN_Input=tf.keras.layers.Input(shape=(50),name='text_input')
    RNN_model.add(RNN_Input)
    RNN_model.add(tf.keras.layers.Embedding (len(vocab_set)+1,256))
    RNN_model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256,stateful=False,return_sequences=True,recurrent_initializer='glorot_uniform')))
    RNN_model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256,stateful=False,return_sequences=True,recurrent_initializer='glorot_uniform')))
    RNN_model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(512,stateful=False,return_sequences=False,recurrent_initializer='glorot_uniform')))

    #ver a camada 
    concat=tf.keras.layers.concatenate([CNN_model.output,RNN_model.output])
    dense_out=tf.keras.layers.Dense(len(vocab_set),activation='softmax',name='output')(concat)

    #ver model 
    model = tf.keras.Model(inputs=[CNN_Input,RNN_Input],
                        outputs=dense_out)
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    model.summary()
    def scheduler(epoch):
        if epoch < 1:
            return 0.001
        else:
            return 0.001 * tf.math.exp(0.1 * (1 - epoch))





    # Create a callback that saves the model's weights
    #cp_callback = tf.keras.callbacks.ModelCheckpoint(
    #  filepath='weights.{epoch:02d}.ckpt', 
    #    verbose=1, 
    #    save_weights_only=True,
    #    save_freq=1000*BATCH_SIZE)
    # ver melhor 
    LRS = tf.keras.callbacks.LearningRateScheduler(scheduler)
    csv_callback=tf.keras.callbacks.CSVLogger(
        "Training Parameters.csv", separator=',', append=False
    )
    filepath="weights-improvement-{epoch:02d}-{val_loss:.2f}.ckpt"
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1,save_weights_only=True)
    callbacks_list = [checkpoint]

    checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
    model.save_weights(checkpoint_path.format(epoch=0,val_loss=0))

    with tf.device(device):
        model.fit(train_dataset,
                validation_data=validation_dataset,
                callbacks=[csv_callback,LRS,callbacks_list],
                epochs=2)

    print("teste")
    for i in range(5):
        index=1
        fig,axis=plt.subplots(1,2,figsize=(25, 8))
        im=cv2.imread(val_dataframe.iloc[index]['Path'])
        im=cv2.resize(im,(200,200))
        q=val_dataframe.iloc[index]['Question']
        q=encoder.encode(q)
        paddings = [[0, 50-tf.shape(q)[0]]]
        q=tf.pad(q, paddings, 'CONSTANT', constant_values=0)
        q=np.array(q)
        ans=model.predict([[im],[q]])
        question=""


        flag=0
        for i,j in enumerate(val_dataframe.iloc[index]['Question']):
            if (flag==1) and (j==' '):
                question+='\n'
                flag=0
            question+=j
            if (i%40==0)and (i!=0):
                flag=1
        axis[0].imshow(im)
        axis[0].axis('off')
        axis[0].set_title('Image', fontsize=30)
        axis[1].text(0.05,0.5,
                "Question = {}\n\nPredicted Answer = {}\n\nActual Answer ={}".format(question,encoder.decode([np.argmax(ans)]),val_dataframe.iloc[index]['Answer']),
                transform=plt.gca().transAxes,fontsize=19)
        axis[1].axis('on')
        axis[1].set_title('Question And Answers', fontsize=30)

