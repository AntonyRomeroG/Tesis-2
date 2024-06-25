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

carpeta = r"C:\Users\Admin\Desktop\ML"

def entrenar_modelo(X_train,y_train,X_valid,y_valid,archivo):
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
    
    #nombre_modelo = "mb_" + archivo[:archivo.rfind('.')] + ".pkl"
    #with open(nombre_modelo, 'wb') as file:
    #    pickle.dump(model, file)
    
    print("Root Mean squared error:", sqrt(mean_squared_error(y_valid, y_predicted)))
    print("R2 score:", r2_score(y_valid, y_predicted)**2)
    
def procesar_archivo_csv(ruta_archivo,archivo):
    try:
        # Leer el archivo CSV
        df = pd.read_csv(ruta_archivo,sep=',')
        
        print(f'Procesando {ruta_archivo}')
        
        array = df.values
        X = array[:,3:8]
        #print(X)
        y = array[:,2]
        #print(y)
        
        test_size = 672/df.shape[0]
        seed = 7
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=test_size, random_state=seed)
        
        entrenar_modelo(X_train,y_train,X_valid,y_valid,archivo)
        
        print(f'{ruta_archivo} procesado con éxito.\n')
    except Exception as e:
        print(f'Error procesando {ruta_archivo}: {e}')

for archivo in os.listdir(carpeta):
    if archivo.endswith('.csv'):
        ruta_archivo = os.path.join(carpeta, archivo)
        procesar_archivo_csv(ruta_archivo,archivo)