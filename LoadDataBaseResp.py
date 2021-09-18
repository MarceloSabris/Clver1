
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


def inserirBanco(linha,qtd,coguinitive,tipo,result):
    try:
        
        query ="INSERT INTO mestrado.predicaoresposta (questao, perc, tipo, resposta, predicado) VALUES(%s, '%s', '%s', '%s', '%s')"%( linha,qtd,coguinitive,tipo,result);
        db.execute(query) 
    except:
        print('erro -- ao executar')
        print(query)


file_to_search = 'C:\\result\\Percentage_3\\'
folders = dirs = [d for d in os.listdir(file_to_search) if os.path.isdir(os.path.join(file_to_search, d))]
folders.sort()
for folder in folders:
    if 'ResultPredication'  in folder: 
       
    
        
        FolderSource = file_to_search + '\\' + folder 
        Files = [f for f in os.listdir(FolderSource) if isfile(join(FolderSource, f))]
        Files.sort()
      
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
                    if 'form' in File:
                        tipo=""
                        if 'Val' in File: 
                            tipo = "val"
                        else:
                            tipo = "train"
                        with open(FolderSource + '\\' + File) as f:
                                fresultpredication = json.load(f)
                                i =0
                                while i < (len(fresultpredication) -1) :
                                    
                                    questao =  fresultpredication[i] 
                                    predicado = fresultpredication[i+2] 
                                    resp = predicado.split('-')
                                    inserirBanco(questao,'3',tipo, resp[0],resp[1] )
                                  
                                    i = i+3

                                    '''result = File 
                                    resultUltimo = File[:-5].replace('-','_').replace('.','_')
                                    split = resultUltimo.split('_')
                                    arquivo =  split[1] + "_"+ split[2]
                                    arquivo = folder.replace('.','_')
                                    if 'Train' in File : 
                                        arquivo  = arquivo + "_Train_" 
                                    else: 
                                        arquivo  = arquivo + "_Val_" 
                                    arquivo =  arquivo + "_" + resultUltimo[len(resultUltimo) -7:len(resultUltimo)] 
                                    linha,qtd,coguinitive,tipo,result
                                    '''
                                
                           
               
                           
  

