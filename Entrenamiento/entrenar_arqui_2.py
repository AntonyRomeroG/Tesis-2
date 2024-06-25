import os
import pandas as pd
from math import sqrt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

# Ruta a la carpeta que contiene los archivos CSV
carpeta = r"C:\Users\Admin\Desktop\ML"

def concatenar_mismo_sensor(arch):
    
    prefix = arch.split('_')[0]
    
    archivos = os.listdir(carpeta)
    archivos_filtrados = [archivo for archivo in archivos if archivo.startswith(prefix)]
    archivos_union = [archivo for archivo in archivos_filtrados if archivo != arch]
    
    dfs = []
    for archivo in archivos_union:
        df = pd.read_csv(os.path.join(carpeta, archivo),sep=',')
        dfs.append(df)
        
    #print(archivos_union)
    
    dataset_union = pd.concat(dfs, ignore_index=True)
    
    return dataset_union

def concatenar_mismo_proveedor(arch):
    prefix = arch.split('_')[0][:-1]
    suffix = arch.split('_')[2][0]
    
    archivos = os.listdir(carpeta)
    archivos_union = [archivo for archivo in archivos if archivo.startswith(prefix) and archivo.split('_')[-1][0] != suffix]
    
    dfs = []
    for archivo in archivos_union:
        df = pd.read_csv(os.path.join(carpeta, archivo),sep=',')
        dfs.append(df)
        
    #print(archivos_union)
        
    dataset_union = pd.concat(dfs, ignore_index=True)
    
    return dataset_union
    
def entrenar_modelo(X_train,y_train,X_valid,y_valid):
    # define los valores que seran probados de los hiperparametros max_depth y n_estimators
    parameter_grid = dict(max_depth=np.array([5,10,20,30,40,50,100,None]),
                      n_estimators=np.array([50,100,150, 500]))

    # define los folds para la evaluacion de cada valor del hiperparametro n_estimators
    seed = 7
    kfold = KFold(n_splits=10, random_state=seed, shuffle= True)  # especifica el particionador de datos a 10-folds CV

    # define el tipo de modelo con el cual se testara diferentes hyperparametros
    model = ExtraTreesRegressor(random_state=seed)

    # define el buscador grid en crosvalidacion
    print("...")
    grid = GridSearchCV(estimator=model, param_grid=parameter_grid, scoring='neg_mean_squared_error', cv=kfold)
    grid_result = grid.fit(X_train, y_train)

    # muestra resultados de la busqueda grid
    #print("Mejor neg_mean_squared_error: %f con hyperparametro %s" % (grid_result.best_score_, grid_result.best_params_))
    
    # Reentrena modelo con ExtraTrees con todos los datos de entrenamiento y lo prueba en el conjunto de validación
    model = ExtraTreesRegressor(max_depth=grid_result.best_params_['max_depth'], n_estimators=grid_result.best_params_['n_estimators'])
    model.fit(X_train, y_train)

    # predice el target en el conjunto de validacion
    y_predicted = model.predict(X_valid)
    
    print("Root Mean squared error:", sqrt(mean_squared_error(y_valid, y_predicted)))
    print("R2 score:", r2_score(y_valid, y_predicted)**2)
    
# Función para procesar cada archivo CSV
def procesar_archivo_csv(ruta_archivo,archivo):
    try:
        # Leer el archivo CSV
        df = pd.read_csv(ruta_archivo,sep=',')
        
        # Realizar operaciones con el DataFrame `df`
        # Ejemplo: Mostrar las primeras 5 filas
        print(f'Procesando {ruta_archivo}')
        
        array = df.values
        X = array[:,2:6]
        y = array[:,1]
        
        test_size = 672/df.shape[0]
        seed = 7
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=test_size, random_state=seed)
        
        
        #concatenar solo del mismo sensor en otro tiempo
        print("Caso mismo sensor:")
        df_concatenar = concatenar_mismo_sensor(archivo)
        
        array = df_concatenar.values
        X = array[:,2:6]
        y = array[:,1]
        
        X_train = np.concatenate((X_train, X), axis=0)
        y_train = np.concatenate((y_train, y), axis=0)
        
        entrenar_modelo(X_train,y_train,X_valid,y_valid)
        
        #concatenar todos los sensores del mismo proveedor en otro tiempo
        print("Caso mismo proveedor:")
        df_concatenar = concatenar_mismo_proveedor(archivo)
        
        array = df_concatenar.values
        X = array[:,2:6]
        y = array[:,1]
        
        X_train = np.concatenate((X_train, X), axis=0)
        y_train = np.concatenate((y_train, y), axis=0)
        
        entrenar_modelo(X_train,y_train,X_valid,y_valid)
        
        
        print(f'{ruta_archivo} procesado con éxito.\n')
    except Exception as e:
        print(f'Error procesando {ruta_archivo}: {e}')

# Recorrer todos los archivos en la carpeta
for archivo in os.listdir(carpeta):
    if archivo.endswith('.csv'):
        ruta_archivo = os.path.join(carpeta, archivo)
        procesar_archivo_csv(ruta_archivo,archivo)