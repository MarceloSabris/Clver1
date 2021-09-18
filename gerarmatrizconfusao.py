
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


import psycopg2
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

filename = 'material'
tipo = 'val'
def cm_analysis(y_true, y_pred, labels, ymap=None, figsize=(1000,1000),filename=None,tipo=None):
    if ymap is not None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=annot, fmt='', ax=ax)
   # plt.savefig('Grafico_'+filename +'_'+tipo )
    plt.show()



db_name = 'postgres'
db_user = 'postgres'
db_pass = 'mudar123'

db_host = 'localhost'
db_port = '5432'

connection = psycopg2.connect(user=db_user,
                                  password=db_pass,
                                  host=db_host,
                                  port=db_port,
                                  database="postgres")







cursor = connection.cursor()
postgreSQL_select_Query = (" select * from mestrado.respostatipo r where tipo= '{0}' ").format(filename)

cursor.execute(postgreSQL_select_Query)
print("select * from mestrado.respostatipo r")
predicado_records = cursor.fetchall()
label=[]
for row in predicado_records:
    label.append(row[0])

label.sort()

cursor = connection.cursor()
postgreSQL_select_Query = ("select * from mestrado.predicaoresposta where tipo = '{0}'").format(tipo)

cursor.execute(postgreSQL_select_Query)
print("Selecting from predicado resposta")
predicado_records = cursor.fetchall()
Resp =[]
Pred=[]
for row in predicado_records:
    for cont in label:  
        if row[3] == cont or row[4] == cont :
             Resp.append(row[3])
             Pred.append(row[4])

cm_analysis(Resp, Pred, label, ymap=None, figsize=(len(label),len(label)),filename=filename, tipo=tipo)
print("fim")


