import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import ( recall_score)
# from sklearn.metrics import make_scorer
import warnings
import tensorflow as tf
from tensorflow.keras.models import load_model
import sys
warnings.filterwarnings("ignore")


def filter_and_add_coordinates(df):
    """
    Filtra un DataFrame para incluir solo ciudades seleccionadas y añade coordenadas fijas.
    Parámetros:
    - df (pd.DataFrame): DataFrame de entrada con la columna 'Location'.
    Retorna:
    - pd.DataFrame: DataFrame filtrado con columnas 'Latitude' y 'Longitude' añadidas, y sin la columna 'Location'.
    """
    # Lista de ciudades a filtrar
    cities = ['Dartmoor', 'Nuriootpa', 'PerthAirport', 'Uluru', 'Cobar', 'CoffsHarbour', 
              'Walpole', 'Cairns', 'AliceSprings', 'GoldCoast']
    # Diccionario de coordenadas de las ciudades
    coordinates = {
        'Dartmoor': {'Latitud': -37.9167, 'Longitud': 141.2833},
        'Nuriootpa': {'Latitud': -34.4700, 'Longitud': 138.9960},
        'PerthAirport': {'Latitud': -31.9403, 'Longitud': 115.9668},
        'Uluru': {'Latitud': -25.3444, 'Longitud': 131.0369},
        'Cobar': {'Latitud': -31.4983, 'Longitud': 145.8389},
        'CoffsHarbour': {'Latitud': -30.2963, 'Longitud': 153.1157},
        'Walpole': {'Latitud': -34.9780, 'Longitud': 116.7330},
        'Cairns': {'Latitud': -16.9203, 'Longitud': 145.7700},
        'AliceSprings': {'Latitud': -23.6980, 'Longitud': 133.8807},
        'GoldCoast': {'Latitud': -28.0167, 'Longitud': 153.4000}
    }
    # Filtrar el DataFrame para incluir solo las ciudades seleccionadas
    df_weather = df[df['Location'].isin(cities)].copy()  # Usamos .copy() para evitar la advertencia de pandas    
    # Añadir las coordenadas de latitud y longitud
    df_weather['Latitude'] = df_weather['Location'].map(lambda loc: coordinates.get(loc, {}).get('Latitud', None))
    df_weather['Longitude'] = df_weather['Location'].map(lambda loc: coordinates.get(loc, {}).get('Longitud', None))    
    # Eliminar la columna 'Location'
    df_weather = df_weather.drop(columns=['Location'])    
    return df_weather

def drop_target(df, target_column='RainTomorrow'):
    """
    Elimina filas con valores nulos en la columna objetivo.
    """
    return df.drop(target_column, axis=1)

def add_month(df):
    # En relación al tiempo se utilizan las features Date(cuantitativa) y Month(cualitativa)
    df['Date']=pd.to_datetime(df['Date'],format='mixed')
    # df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Month'] = df['Date'].dt.month
    meses = ['ene','feb','mar','abr','may','jun','jul','ago','sep','oct','nov','dic']
    month_dict = {}
    for i,mes in enumerate(meses):
        month_dict[i+1] = mes
    df['Month'] = df['Month'].map(month_dict)
    return df

def impute_media_segmentada_por_mes(df, variables, mes_col='Month'):
    """
    Imputa valores faltantes en las columnas especificadas usando la media segmentada por mes.    
    Parámetros:
    - df (pd.DataFrame): DataFrame de entrada.
    - variables (list): Lista de columnas a imputar.
    - mes_col (str): Nombre de la columna que contiene los meses (default='Month').    
    Retorna:
    - pd.DataFrame: DataFrame con valores imputados en las columnas seleccionadas.
    """
    # Hacer una copia para evitar modificar el DataFrame original
    df_imputed = df.copy()    
    # Calcular la media por mes para cada variable
    medias_por_mes = {
        var: df_imputed.groupby(mes_col)[var].mean() for var in variables
    }   
    # Imputar valores faltantes usando la media segmentada por mes
    for var in variables:
        df_imputed[var] = df_imputed[var].fillna(df_imputed[mes_col].map(medias_por_mes[var]))    
    return df_imputed

def impute_wind_directions(df):
    """
    Imputa valores faltantes en las columnas relacionadas con las direcciones de viento.    
    Parámetros:
    - df (pd.DataFrame): DataFrame que contiene las columnas 'WindDir9am', 'WindDir3pm' y 'WindGustDir'.    
    Retorna:
    - pd.DataFrame: DataFrame con los valores imputados en las columnas de direcciones de viento.
    """
    # Crear una copia del DataFrame para evitar modificar el original
    df = df.copy()    
    # Imputar 'WindDir9am' y 'WindDir3pm' en función de 'WindGustDir'
    df['WindDir9am'] = df.apply(
        lambda row: row['WindGustDir'] if pd.isna(row['WindDir9am']) and pd.notna(row['WindGustDir']) else row['WindDir9am'],
        axis=1
    )
    df['WindDir3pm'] = df.apply(
        lambda row: row['WindGustDir'] if pd.isna(row['WindDir3pm']) and pd.notna(row['WindGustDir']) else row['WindDir3pm'],
        axis=1
    )
    # Imputar 'WindGustDir' en función de 'WindDir9am' y 'WindDir3pm'
    df['WindGustDir'] = df.apply(
        lambda row: row['WindDir9am'] if pd.isna(row['WindGustDir']) and pd.notna(row['WindDir9am']) else row['WindGustDir'],
        axis=1
    )
    df['WindGustDir'] = df.apply(
        lambda row: row['WindDir3pm'] if pd.isna(row['WindGustDir']) and pd.notna(row['WindDir3pm']) else row['WindGustDir'],
        axis=1
    )
    return df

def imputar_moda(df, variables):
    """
    Imputa valores faltantes en las columnas especificadas usando la moda.    
    Parámetros:
    - df (pd.DataFrame): DataFrame de entrada.
    - variables (list): Lista de columnas a imputar.    
    Retorna:
    - pd.DataFrame: DataFrame con los valores imputados en las columnas especificadas.
    """
    # Crear una copia del DataFrame para no modificar el original
    df = df.copy()    
    # Calcular la moda de cada columna y realizar la imputación
    for var in variables:
        moda = df[var].mode()[0]  # Obtener la moda de la columna
        df[var] = df[var].fillna(moda)  # Imputar valores faltantes con la moda    
    return df

def map_wind_directions(df, features_cuali, direction_dict):
    """
    Mapea las direcciones del viento en un DataFrame según un diccionario de mapeo.
    Parámetros:
    - df (pd.DataFrame): DataFrame de entrada que contiene las columnas de direcciones de viento.
    - features_cuali (list): Lista de columnas que contienen las direcciones de viento.
    - direction_dict (dict): Diccionario que define el mapeo de direcciones de viento.
    Retorna:
    - pd.DataFrame: DataFrame con las direcciones mapeadas según el diccionario.
    """
    # Aplicar el mapeo de direcciones en las columnas especificadas
    for feature_cuali in features_cuali:  # Limitar a las primeras 
        df[feature_cuali] = df[feature_cuali].map(direction_dict)    
    return df

def impute_with_median(df, columns_to_impute):
    """
    Imputa valores faltantes en un DataFrame usando la mediana de las columnas especificadas.    
    Parámetros:
    - df (pd.DataFrame): DataFrame de entrada.
    - columns_to_impute (list): Lista de columnas a imputar utilizando la mediana.    
    Retorna:
    - pd.DataFrame: DataFrame con los valores imputados en las columnas especificadas.
    """
    # Calcular la mediana de las columnas seleccionadas
    medians = {col: df[col].median() for col in columns_to_impute}
    # Imputar las medianas en el DataFrame
    df_imputed = df.fillna(medians)    
    return df_imputed

def impute_with_mean(df, columns_to_impute_simetricas):
    """
    Imputa valores faltantes en un DataFrame usando la media de las columnas especificadas.    
    Parámetros:
    - df (pd.DataFrame): DataFrame de entrada.
    - columns_to_impute (list): Lista de columnas a imputar utilizando la media.    
    Retorna:
    - pd.DataFrame: DataFrame con los valores imputados en las columnas especificadas.
    """
    # Calcular la media de las columnas seleccionadas
    means = {col: df[col].mean() for col in columns_to_impute_simetricas}
    # Imputar las medias en el DataFrame
    df_imputed = df.fillna(means)    
    return df_imputed

def encode_categorical_features(data_set, col_catego):
    """
    Realiza la codificación one-hot de columnas categóricas en un DataFrame.
    Parámetros:
    - data_set (pd.DataFrame): DataFrame original con las columnas categóricas.
    - col_catego (list): Lista de nombres de columnas categóricas a codificar.
    Retorna:
    - pd.DataFrame: DataFrame con las columnas categóricas codificadas.
    """
    # Inicializar el codificador OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    # Ajustar y transformar las columnas categóricas
    one_hot_encoded = encoder.fit_transform(data_set[col_catego])
    # Crear un DataFrame con las columnas codificadas
    one_hot_df = pd.DataFrame(one_hot_encoded, 
                              columns=encoder.get_feature_names_out(col_catego), 
                              index=data_set.index)
    # Eliminar las columnas categóricas originales y unir las codificadas
    df_encoded = pd.concat([data_set.drop(columns=col_catego), one_hot_df], axis=1)
    return df_encoded

def scale_numeric_features(df, numeric_columns):
    """
    Escala las características numéricas en un único DataFrame.
    Parámetros:
    - df (pd.DataFrame): DataFrame de entrada.
    - numeric_columns (list): Lista de columnas numéricas a escalar.
    Retorna:
    - pd.DataFrame: DataFrame con las columnas numéricas escaladas.
    """
    # Inicializar el escalador
    scaler = StandardScaler()
    # Escalar las columnas numéricas
    df_scaled = df.copy()
    df_scaled[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    return df_scaled

def drop_date_column(df):
    """
    Elimina la columna 'Date' de un DataFrame.
    Parámetros:
    - df (pd.DataFrame): DataFrame de entrada.
    Retorna:
    - pd.DataFrame: DataFrame sin la columna 'Date'.
    """
    if 'Date' in df.columns:
        df_cleaned = df.drop(columns=['Date'])
    else:
        raise ValueError("La columna 'Date' no existe en el DataFrame.")
    return df_cleaned

def pondered_accuracy(y_true, y_pred):
    recall_scorer_0 = recall_score(y_true, y_pred, pos_label=0)
    recall_scorer_1 = recall_score(y_true, y_pred, pos_label=1)
    return (2 * recall_scorer_1 + recall_scorer_0) / 3

def pondered_accuracy_rn(y_true, y_pred):
    # Convertimos predicciones a clases binarias (umbral 0.5)
    y_pred_binary = tf.cast(y_pred > 0.5, tf.int32)
    y_true = tf.cast(y_true, tf.int32)
    # Calculamos recall por clase
    recall_0 = tf.reduce_sum(tf.cast(tf.logical_and(y_true == 0, y_pred_binary == 0), tf.float32)) / (tf.reduce_sum(tf.cast(y_true == 0, tf.float32) + tf.keras.backend.epsilon()))
    recall_1 = tf.reduce_sum(tf.cast(tf.logical_and(y_true == 1, y_pred_binary == 1), tf.float32)) / (tf.reduce_sum(tf.cast(y_true == 1, tf.float32) + tf.keras.backend.epsilon()))
    # Ponderación
    return (2 * recall_1 + recall_0) / 3


if __name__ == '__main__':
    # Leer datos de entrada desde la línea de comandos
    if len(sys.argv) < 2:
        print("Error: Debes proporcionar un archivo .csv como argumento.")
        sys.exit(1)

    datos = sys.argv[1]
    print(f"Procesando el archivo: {datos}")
    try:
        input = pd.read_csv(datos)

        paso1 = filter_and_add_coordinates(input)
        paso2 =  add_month(paso1)
        # Imputaciones
        variables = ['Sunshine', 'Cloud9am', 'Cloud3pm', 'Evaporation']
        paso3 = impute_media_segmentada_por_mes(paso2, variables, mes_col='Month')
        paso4 = impute_wind_directions(paso3)
        variables = ['WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']
        paso5 = imputar_moda(paso4, variables)
        features_cuali = ['WindGustDir', 'WindDir9am', 'WindDir3pm']
        direction_dict = {
            'SSW':'S',
            'S':'S',
            'SE':'S',
            'NNE':'N',
            'WNW':'W',
            'N':'N',
            'ENE':'E',
            'NE':'N',
            'E':'E',
            'SW':'S',
            'W':'W',
            'WSW':'W',
            'NNW':'N',
            'ESE':'E',
            'SSE':'S',
            'NW':'N'
        }
        paso6 = map_wind_directions(paso5, features_cuali, direction_dict)
        columns_to_impute = ['WindGustSpeed', 'Humidity9am', 'Humidity3pm', 'Rainfall',
                            'WindSpeed9am', 'WindSpeed3pm', 'Pressure9am', 'Pressure3pm']
        paso7 = impute_with_median(paso6, columns_to_impute)
        columns_to_impute_simetricas = ['MinTemp', 'MaxTemp', 'Temp9am', 'Temp3pm']
        paso8 = impute_with_mean(paso7, columns_to_impute_simetricas)
        features_cuali = ['WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday', 'Month']
        # paso9 = codificar_features_cuali(paso8, features_cuali)
        paso9 = encode_categorical_features(paso8, features_cuali)
        # Escalado
        features_cuanti = ['MinTemp','MaxTemp','Rainfall','Evaporation','Sunshine','WindGustSpeed','WindSpeed9am','WindSpeed3pm',
        'Humidity9am','Humidity3pm','Pressure9am','Pressure3pm','Cloud9am','Cloud3pm','Temp9am','Temp3pm','Latitude','Longitude']
        paso10 = scale_numeric_features(paso9, features_cuanti)
        # Elimina columna date
        paso11 = drop_date_column(paso10)

        modelo_cargado = load_model("modelos/modelo_rn_best.h5", custom_objects={'pondered_accuracy_rn': pondered_accuracy_rn})
        # Evaluar el modelo cargado
        predict = modelo_cargado.predict(paso11)
        pd.DataFrame(predict, columns=['RainTomorrow']).to_csv('output.csv', index = False)#Ruta dentro del contenedor

    except ValueError:
        print('Archivo con formato incorrecto o no se encontró el archivo')
