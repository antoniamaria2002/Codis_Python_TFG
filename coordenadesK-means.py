# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 13:56:31 2025

@author: Moi
"""


import cv2
import numpy as np
import os
import pandas as pd

# Funció per convertir coordenades de píxels a latitud i longitud
def pixel_to_latlon(px_x, px_y, lat_min, lat_max, lon_min, lon_max, width, height):
    """
    Converteix les coordenades d’un píxel (x, y) a latitud i longitud.
    
    Args:
        px_x (int): Posició x del píxel (horitzontal)
        px_y (int): Posició y del píxel (vertical)
        lat_min (float): Latitud mínima (part inferior)
        lat_max (float): Latitud màxima (part superior)
        lon_min (float): Longitud mínima (esquerra)
        lon_max (float): Longitud màxima (dreta)
        width (int): Amplada de la imatge en píxels
        height (int): Alçada de la imatge en píxels

    Returns:
        (float, float): (latitud, longitud)
    """
    lon = lon_min + (px_x / width) * (lon_max - lon_min)
    lat = lat_max - (px_y / height) * (lat_max - lat_min)  
    return lat, lon

# Ruta de la imatge amb la màscara de la posidònia (meitat inferior)
posidonia_image_path = r"C:\Users\Moi\Desktop\tfg\videoimatgesvilanova\10juriol2018\imatges\K-means_Posidonia_Area.png"

# Comprovar si la imatge existeix a la ruta indicada
if os.path.exists(posidonia_image_path):
    posidonia_mask = cv2.imread(posidonia_image_path, cv2.IMREAD_GRAYSCALE)  # Carregar en escala de grisos
    
    if posidonia_mask is None:
        print("Error: No s'ha pogut carregar la màscara de *Posidònia*. Verifica la ruta.")
    else:
        print(" Màscara de *Posidònia* carregada correctament.")

        # Coordenades adaptades per a la meitat inferior de la imatge 
        lat_min, lat_max = 41.1683645522291, 41.19261949901548  
        lon_min, lon_max = 1.64385338, 1.8365719  
        height, width = posidonia_mask.shape[:2]

        latitudes = []
        longitudes = []

        # Recórrer cada píxel negre (valor 0) de la màscara
        for y in range(height):
            for x in range(width):
                if posidonia_mask[y, x] == 0:
                    lat, lon = pixel_to_latlon(x, y, lat_min, lat_max, lon_min, lon_max, width, height)
                    latitudes.append(f"{lat:.8f}")
                    longitudes.append(f"{lon:.8f}")

        # Crear DataFrame amb coordenades en format (Latitud, Longitud)
        df_lat_lon = pd.DataFrame({'Latitud': latitudes, 'Longitud': longitudes})

        # Guardar com a fitxer CSV (Latitud, Longitud)
        output_directory = os.path.dirname(posidonia_image_path)
        output_csv_path_lat_lon = os.path.join(output_directory, "Posidonia_Lat_Lon.csv")
        df_lat_lon.to_csv(output_csv_path_lat_lon, index=False)

        print(f"Coordenades (Latitud, Longitud) desades a: '{output_csv_path_lat_lon}'.")

        # Crear DataFrame en format (Longitud, Latitud)
        df_lon_lat = pd.DataFrame({'Longitud': longitudes, 'Latitud': latitudes})

        # Guardar com a fitxer CSV (Longitud, Latitud)
        output_csv_path_lon_lat = os.path.join(output_directory, "Posidonia_Lon_Lat.csv")
        df_lon_lat.to_csv(output_csv_path_lon_lat, index=False)

        print(f" Coordenades (Longitud, Latitud) desades a: '{output_csv_path_lon_lat}'.")

else:
    print(f" Error: La màscara de *Posidònia* no existeix a la ruta indicada: {posidonia_image_path}")
