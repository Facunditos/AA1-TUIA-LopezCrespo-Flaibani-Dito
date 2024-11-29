
#if __name__ == "__main__":
#    main()

import pandas as pd
import numpy as np
import warnings
#import tensorflow as tf
warnings.filterwarnings("ignore")

#df = pd.read_csv(r"C:\Users\Usuario\Documents\TECNICATURA EN IA\AA1_TP_daniela\weatherAUS.csv")

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

#df_weather = df_weather.dropna(subset=['RainTomorrow'])

def drop_missing_target(df, target_column='RainTomorrow'):
    """
    Elimina filas con valores nulos en la columna objetivo.
    """
    df= filter_and_add_coordinates(df)
    return df.dropna(subset=[target_column])

def transform_date_column(df, date_column='Date', month_column='Month'):
    """
    Transforma una columna de fecha para extraer el mes como categórico.
    Retorna: pd.DataFrame: DataFrame con la columna de fechas convertida y una nueva columna con el mes categórico.
    """
    # Diccionario para convertir números de mes a nombres
    month_dict = {
        1: 'ene', 2: 'feb', 3: 'mar', 4: 'abr',
        5: 'may', 6: 'jun', 7: 'jul', 8: 'ago',
        9: 'sep', 10: 'oct', 11: 'nov', 12: 'dic'
    }
    
    # Crear una copia del DataFrame para evitar modificar el original
    df_weather= drop_missing_target(df).copy()
    
    # Convertir la columna de fechas a datetime
    df_weather[date_column] = pd.to_datetime(df_weather[date_column], format='mixed')
    #df_weather['Date']=pd.to_datetime(df_weather['Date'],format='mixed')
    
    # Extraer el mes y mapearlo a nombres de mes
    df_weather[month_column] = df_weather[date_column].dt.month.map(month_dict)
    
    return df_weather

