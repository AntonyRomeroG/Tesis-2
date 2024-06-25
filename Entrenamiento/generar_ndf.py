import os
import pandas as pd
from math import sqrt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
import pickle

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

carpeta = r"C:\Users\Admin\Desktop\DATA\Scripts\"

archivo = 'iq1_final_l.csv'

with open(carpeta + 'mb_iq1_final_l.pkl', 'rb') as file:
  model1 = pickle.load(file)
with open(carpeta + 'mb_iq2_final_l.pkl', 'rb') as file:
  model2 = pickle.load(file)
with open(carpeta + 'mb_iq3_final_l.pkl', 'rb') as file:
  model3 = pickle.load(file)
  
with open(carpeta + 'mb_p1_final_l.pkl', 'rb') as file:
  model4 = pickle.load(file)
with open(carpeta + 'mb_p2_final_l.pkl', 'rb') as file:
  model5 = pickle.load(file)
with open(carpeta + 'mb_p3_final_l.pkl', 'rb') as file:
  model6 = pickle.load(file)
with open(carpeta + 'mb_p4_final_l.pkl', 'rb') as file:
  model7 = pickle.load(file)

with open(carpeta + 'mb_q1_final_l.pkl', 'rb') as file:
  model8 = pickle.load(file)
with open(carpeta + 'mb_q2_final_l.pkl', 'rb') as file:
  model9 = pickle.load(file)
with open(carpeta + 'mb_q3_final_l.pkl', 'rb') as file:
  model10 = pickle.load(file)
with open(carpeta + 'mb_q4_final_l.pkl', 'rb') as file:
  model11 = pickle.load(file)
with open(carpeta + 'mb_q4_final_l.pkl', 'rb') as file:
  model12 = pickle.load(file)

def procesar_archivo_csv(archivo):
    try:
        # Leer el archivo CSV
        df = pd.read_csv(archivo,sep=',')
        
        print(f'Procesando {archivo}')
        
        array = df.values
        X = array[:,2:6]
        y = array[:,1]
        
        y1_predicted = model1.predict(X)
        y2_predicted = model2.predict(X)
        y3_predicted = model3.predict(X)
        y4_predicted = model4.predict(X)
        y5_predicted = model5.predict(X)
        y6_predicted = model6.predict(X)
        y7_predicted = model7.predict(X)
        y8_predicted = model8.predict(X)
        y9_predicted = model9.predict(X)
        y10_predicted = model10.predict(X)
        y11_predicted = model11.predict(X)
        y12_predicted = model12.predict(X)
        
        y_promedio_total = (y1_predicted + y2_predicted + y3_predicted + y4_predicted + y5_predicted
                      + y6_predicted + y7_predicted + y8_predicted + y9_predicted
                      + y10_predicted + y11_predicted + y12_predicted)/12
        
        y_promedio_total_2d = y_promedio_total.reshape(-1, 1)
        
        y_promedio_iq = (y1_predicted + y2_predicted + y3_predicted)/3
        
        y_promedio_iq_2d = y_promedio_iq.reshape(-1, 1)
        
        y_promedio_p = (y4_predicted + y5_predicted + y6_predicted + y7_predicted)/4
        
        y_promedio_p_2d = y_promedio_p.reshape(-1, 1)
        
        y_promedio_q = (y8_predicted + y9_predicted + y10_predicted
                        + y11_predicted + y12_predicted)/5
        
        y_promedio_q_2d = y_promedio_q.reshape(-1, 1)
        
        #Caso 1
        
        df1 = df.copy(deep=True)
        df1['sp'] = y_promedio_iq_2d
        
        df1.to_csv(archivo[:archivo.rfind('.')] + "_sp" + ".csv", sep=',')
        
        #Caso 2
        
        df2 = df.copy(deep=True)
        df2['ap'] = y_promedio_total_2d
        
        df2.to_csv(archivo[:archivo.rfind('.')] + "_ap" + ".csv", sep=',')
        
        print(f'{archivo} procesado con éxito.\n')
    except Exception as e:
        print(f'Error procesando {archivo}: {e}')


procesar_archivo_csv(archivo)