  
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


from sqlalchemy import create_engine

from sqlalchemy import create_engine

db_name = 'postgres'
db_user = 'postgres'
db_pass = 'mudar123'
db_host = 'localhost'
db_port = '5432'

db_string = 'postgres://{}:{}@{}:{}/{}'.format(db_user, db_pass, db_host, db_port, db_name)
db = create_engine(db_string)
def gravarresposta(resposta ):
    try:
       query = "INSERT INTO mestrado.RespostaTipo (resposta) VALUES('%s')" %( resposta)
       db.execute(query) 
    except:
        print('erro -- ao executar')
        print(query)

def inserirBanco(linha,qtd,coguinitive,tipo,result):
    try:
        
        query ="INSERT INTO mestrado.predicaoresposta (questao, perc, tipo, resposta, predicado) VALUES(%s, '%s', '%s', '%s', '%s')"%( linha,qtd,coguinitive,tipo,result);
        db.execute(query) 
    except:
        print('erro -- ao executar')
        print(query)

from os.path import isfile, join


def GravarArquivo ( data_dict,fname):
    fname = fname  + '.json'
    print("gravar arquivo: " + fname + " qtd: " +  str(len(data_dict)))
    os.makedirs('tipoquestao' , exist_ok=True)
    fname = 'tipoquestao' + "/" + fname
    # Create file
    with open(fname, 'w') as outfile:
        json.dump(data_dict, outfile, ensure_ascii=False, indent=4) 
        outfile.close()


valList=[]
with open('C:\\Users\\Admin\\Downloads\\CLEVR_v1.0\\CLEVR_v1.0_1\\questions\CLEVR_val_questions.json') as f:
    data = json.load(f)
    
    with open('C:\\Users\\Admin\\Documents\\val_questao_material.json') as mat: 
       mat1 = json.load(mat)
       for m in mat1 :
        print(int(str(m).split(':')[1].replace('''''','').replace('}',''))) 
  # int(str(mat1[1]).split(':')[1].replace('''''','').replace('}',''))
        
    mat.close()
    
    #lenghtDataVal = int(len(data['questions']))
    #for K in range(lenghtDataVal):
    
       
    #    i = data['questions'][K]
        
    #    temp=[]
    #    temp.append(i['question'])
    #    temp.append(i['answer'])
    #    valList.append(temp)
f.close() 


trainList =[]
with open('C:\\Users\\Admin\\Downloads\\CLEVR_v1.0\\CLEVR_v1.0_1\\questions\\CLEVR_train_questions.json') as f:
    with open('C:\\Users\\Admin\\Documents\\val_questao_material.json') as mat:
        mat1 = json.load(mat)
        for m in mat1 : 
            print(m)
            print('oi')
            #data = json.load(f)
            #lenghtDataTrain = int(len(data['questions']))
            #print(lenghtDataTrain)
            #for K in range(lenghtDataTrain):
                #i = data['questions'][K]
        
                #temp=[]
                    
                #temp.append(i['question'])
                #temp.append(i['answer'])
                #trainList.append(temp)
f.close()




trainList =[]
with open('C:\\Users\\Admin\\Downloads\\CLEVR_v1.0\\CLEVR_v1.0_1\\questions\\CLEVR_train_questions.json') as f:
    
    print('oi')
    data = json.load(f)
    lenghtDataTrain = int(len(data['questions']))
    print(lenghtDataTrain)
    for K in range(lenghtDataTrain):
        i = data['questions'][K]
       
        temp=[]
                  
        temp.append(i['question'])
        temp.append(i['answer'])
        trainList.append(temp)
f.close()


valList=[]
with open('C:\\Users\\Admin\\Downloads\\CLEVR_v1.0\\CLEVR_v1.0_1\\questions\CLEVR_val_questions.json') as f:
    data = json.load(f)
    lenghtDataVal = int(len(data['questions']))
    for K in range(lenghtDataVal):
    
       
        i = data['questions'][K]
        
        temp=[]
        temp.append(i['question'])
        temp.append(i['answer'])
        valList.append(temp)
f.close() 



       



for contador in range(len(trainList)) : 
    inserirBanco( contador,'101', 'train', trainList[contador][1] ,trainList[contador][1] )     
  


for contador in range(len(valList)) : 
    inserirBanco( contador,'101', 'val', valList[contador][1] ,valList[contador][1] ) 

for contador in range(len(trainList)) : 
    gravarresposta(trainList[contador][1]  )     
  


for contador in range(len(valList)) : 
      gravarresposta(valList[contador][1]  )  

questaoContador = []
for contador in range(len(trainList)) : 
    if 'are there any' in trainList[contador][0].lower(): 
        questaoContador.append(contador)
GravarArquivo(questaoContador,'questaoExistsTrains')     
  

questaoContador = []
for contador in range(len(valList)) : 
    if 'are there any' in valList[contador][0].lower(): 
        questaoContador.append(contador)
GravarArquivo(questaoContador,'questaoExistsVal')     
  


questaoContador = []
for contador in range(len(trainList)) : 
    if 'what number' in trainList[contador][0].lower() : 
        questaoContador.append(contador)
    if 'how many' in trainList[contador][0].lower() : 
        questaoContador.append(contador)
GravarArquivo(questaoContador,'questaoContarVal')  

questaoContador = []
for contador in range(len(valList)) : 
   if 'what number' in trainList[contador][0].lower() : 
        questaoContador.append(contador)
   if 'how many' in trainList[contador][0].lower() : 
        questaoContador.append(contador)
GravarArquivo(questaoContador,'questaoContarTrain') 