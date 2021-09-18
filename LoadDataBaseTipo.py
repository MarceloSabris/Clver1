
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
def add_new_row(nrquestao,tipodatabase,tipoquestao):
    try:
       query = "INSERT INTO mestrado.QuestaoPorTipo (nrquestao, tipodatabase, tipoquestao) VALUES(%s,'%s','%s')" %( nrquestao,tipodatabase,tipoquestao)
       db.execute(query) 
    except:
        print('erro -- ao executar')
        print(query)


file_to_search = 'C:\\result\\tipoquestao'

Files = [f for f in os.listdir(file_to_search) if isfile(join(file_to_search, f))]
Files.sort()
for File in Files:
    ler=[]
    tipo="" 
    tipoquestao =""
    if '.json' in File:
        if 'Train' in File:      
              tipo = "Train"
        else:
              tipo = "Val"
        if 'Exists' in File: 
            tipoquestao = "Exist"
        elif 'Contar' in  File: 
            tipoquestao = "Contar"
        with open(file_to_search + '\\' + File) as f:
                ques = json.load(f)
                for que in ques :
                    add_new_row(que,tipo,tipoquestao)

                           
  


