
import numpy as np
import pandas as pd   
import os
from pathlib import Path
import glob
import json

import os
import nltk

import cv2
import matplotlib.pyplot as plt
import random


from os.path import isfile, join
from sqlalchemy import create_engine

from sqlalchemy import create_engine

db_name = 'postgres'
db_user = 'postgres'
db_pass = 'mudar123'
db_host = 'localhost'
db_port = '5432'

db_string = 'postgres://{}:{}@{}:{}/{}'.format(db_user, db_pass, db_host, db_port, db_name)
db = create_engine(db_string)


def add_new_row(linha,qtd,coguinitive,tipo,result):
    try:
       query = "INSERT INTO mestrado.processamento (linha, qtd, coguinitive,tipo, result) VALUES(%s,'%s','%s','%s','%s')" %( linha,qtd,coguinitive,tipo,result)
       db.execute(query) 
    except:
        print('erro -- ao executar')
        print(query)



file_to_search = 'C:\\result'
folders = dirs = [d for d in os.listdir(file_to_search) if os.path.isdir(os.path.join(file_to_search, d))]
folders.sort()
for folder in folders:
    if 'Percentage_3'  in folder: 
        qtd = 0 
    
        vocab_set=set()#set object used to store the vocabulary
        partSplit = folder.split('_')
        perc = partSplit[1]
        FolderSource = file_to_search + '\\' + folder 
        Files = [f for f in os.listdir(FolderSource) if isfile(join(FolderSource, f))]
        Files.sort()
        posQues =[]
        valList=[]
        trainList=[]
        memorys = []
        resultpredication = []
        Files.sort()
        for File in Files:
            
            if '.json' in File:

              '''  if 'ques' in File:     
                    with open(FolderSource + '\\' + File) as f:
                            ques = json.load(f)
          
                            for que in ques :
                                valList.append(que)
                elif 'train' in File:
                    with open(FolderSource + '\\' + File) as f:
                            Ftrain = json.load(f)
                            for tr in Ftrain:
                                trainList.append(tr)
               
                el''' 
              if 'Resul' in File :
                    with open(FolderSource + '\\' + File) as f:
                            fresultpredication = json.load(f)
                            i =0
                            result = File 
                            resultUltimo = File[:-5].replace('-','_').replace('.','_')
                            split = resultUltimo.split('_')
                            arquivo =  split[1] + "_"+ split[2]
                            '''arquivo = folder.replace('.','_')
                            if 'Train' in File : 
                                arquivo  = arquivo + "_Train_" 
                            else: 
                                arquivo  = arquivo + "_Val_" 
                            arquivo =  arquivo + "_" + resultUltimo[len(resultUltimo) -7:len(resultUltimo)] 
                             linha,qtd,coguinitive,tipo,result
                            '''
                           
                            for tr in fresultpredication:
                                add_new_row(i,perc,split[2], split[1],tr)
                                i=i+1
                           
               
                           
  

