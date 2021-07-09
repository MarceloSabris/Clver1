
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

lenghtData = 2
physical_devices = tf.config.list_physical_devices('GPU') 
print("Number of GPUs :", len(physical_devices)) 
print("Tensorflow GPU :",tf.test.is_built_with_cuda())
if len(physical_devices)>0:
    device="/GPU:0"
else:
    device="/CPU:0"

trainList=[]


def to_xml(df, filename=None, mode='w'):
    def row_to_xml(row):
        xml = ['<item>']
        for field in row.index:
            xml.append('  <field name="{0}">{1}</field>'.format(field, row[field]))
            xml.append('</item>')
        return '\n'.join(xml)   
    res = '\n'.join(df.apply(row_to_xml, axis=1))
    
    
    
    #def row_to_xml(row):
    #    xml = ['<item>']
    #    for i, col_name in enumerate(row.index):
    #        xml.append('  <field name="{0}">{1}</field>'.format(col_name, row.iloc[i]))
    #    xml.append('</item>')
    #    return '\n'.join(xml)
    #res = '\n'.join(df.apply(row_to_xml, axis=1))

    if filename is None:
        return res
    with open(filename, mode) as f:
        f.write(res)
posTrain = []
posQues = []

with open('C:\\Users\\Admin\\Downloads\\CLEVR_v1.0\\CLEVR_v1.0_1\\questions\\CLEVR_train_questions.json') as f:
    data = json.load(f)
    
    for k in range(lenghtData):
        pos = random.randrange(20, 5000, 3)
        i = data['questions'][pos]
        posTrain.append(pos)
        temp=[]
        for path in glob.glob('C:\\Users\\Admin\\Downloads\\CLEVR_v1.0\\CLEVR_v1.0_1\\images\\train\\'+i['image_filename']): 
            temp.append(path)
        temp.append(i['question'])
        temp.append(i['answer'])
        trainList.append(temp)
f.close()
labels=['Path','Question','Answer']
pd.DataFrame.to_xml = to_xml
train_dataframe = pd.DataFrame.from_records(trainList, columns=labels)#training Dataframe 
train_dataframe.to_json(orient='records')

del(data)
del(trainList)



valList=[]
with open('C:\\Users\\Admin\\Downloads\\CLEVR_v1.0\\CLEVR_v1.0_1\\questions\\CLEVR_val_questions.json') as f:
    data = json.load(f)
    for k in range(lenghtData):
        item =random.randrange(20, 5000, 3)
        posQues.append(item)
        i = data['questions'][item]
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

encoder=tfds.deprecated.text.TokenTextEncoder(vocab_set)
index=1
print("Testing the Encoder with sample questions - \n ")
example_text=encoder.encode(train_dataframe['Question'][index])
print("Original Text = "+train_dataframe['Question'][index])
print("After Encoding = "+str(example_text))


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

#The training and validation Dataset objects are created
train_dataset=create_pipeline(train_dataframe)
validation_dataset=create_pipeline(val_dataframe)



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
dense_out=tf.keras.layers.Dense(len(vocab_set),activation='softmax',name='output')(concat)

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

LRS = tf.keras.callbacks.LearningRateScheduler(scheduler)
csv_callback=tf.keras.callbacks.CSVLogger(
    "Training Parameters.csv", separator=',', append=False
)
with tf.device(device):
    history = model.fit(train_dataset,
              validation_data=validation_dataset,
              callbacks=[csv_callback,LRS],
              epochs=4)

       
  
print("Predictions Are as follows = ")
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
history.history
hist.tail()
def plot_loss(history):
  plt.plot(history.history['loss'], label='training loss')
  plt.plot(history.history['val_loss'], label=' validation loss')
  plt.ylim([0, 10])
  plt.xlabel('Epoch')
  plt.ylabel('Error')
  plt.legend()
  plt.grid(True)
  plt.show()
  plt.savefig('error.png')
  
def plot_acc(history):  
  plt.plot(history.history['sparse_categorical_accuracy'])
  plt.plot(history.history['val_sparse_categorical_accuracy'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'val'], loc='upper left')
  plt.grid(True)
  plt.savefig('accuracy.png')
  plt.show()
   

plot_loss(history)
plot_acc(history)

#def plot_ACC(history):
#  plt.plot(history.history['acc'], label='training ACURACY')
#  plt.plot(history.history['val_acc'], label=' validation ACURACY')
#  plt.ylim([0, 10])
#  plt.xlabel('Epoch')
#  plt.ylabel('Error [MPG]')
#  plt.legend()
#  plt.grid(True)
#  plt.show()


#plot_ACC(history)

    #The Answer is also encoded 
  


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
    im.resize(1,200,200,3)
    
    .resize(1,50)
    ans=model.predict([[im],[q]])
    question=""
    flag=0
    im.resize(200,200,3)
    q.resize(50)
    for i,j in enumerate(val_dataframe.iloc[index]['Question']):
        if (flag==1) and (j==' '):
            question+='\n'
            flag=0
        question+=j
        if (i%40==0)and (i!=0):
            flag=1
    axis[0].imshow(im)
   
    axis[0].set_title('Image', fontsize=30)
    axis[1].text(0.05,0.5,
             "Question  = {}\n\nPredicted Answer = {}\n\nActual Answer ={}".format(question,encoder.decode([np.argmax(ans)]),val_dataframe.iloc[index]['Answer']),
             transform=plt.gca().transAxes,fontsize=19)
  
    axis[1].set_title('Question And Answers', fontsize=30)
    plt.show()


with open('trainQues_'+str(lenghtData) + '.json', 'w', encoding='utf-8') as f:
    json.dump(posTrain, f, ensure_ascii=False, indent=4)
with open('ValidQues_'+str(lenghtData) + '.json', 'w', encoding='utf-8') as f:
    json.dump(posQues,f, ensure_ascii=False, indent=4)