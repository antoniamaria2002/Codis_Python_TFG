# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 15:33:03 2025

@author: Moi
"""

import cv2 
import numpy as np

# Ruta de la imagen en escala de grises
ruta_imagen = r'C:\Users\Moi\Desktop\tfg\videoimatgesvilanova\posidoniavilanova3juliol20234bandas\codis\Blue-Nir_resta2023\Blue_Corrected2_cropped.png'

# Ruta para guardar la imagen procesada
ruta_guardar_imagen = r'C:\Users\Moi\Desktop\tfg\videoimatgesvilanova\posidoniavilanova3juliol20234bandas\codis\imatge_punts_vermells.png'

# Cargar la imagen en escala de grises
imagen_gray = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)

# Convertir la imagen en escala de grises a formato RGB
imagen_rgb = cv2.cvtColor(imagen_gray, cv2.COLOR_GRAY2BGR)

# Identificar los píxeles en el rango de 100 a 180
mask = (imagen_gray >= 25) & (imagen_gray <= 45)

# Cambiar esos píxeles a color rojo (en formato BGR, rojo es [0, 0, 255])
imagen_rgb[mask] = [0, 0, 255]

# Guardar la imagen procesada
cv2.imwrite(ruta_guardar_imagen, imagen_rgb)

print(f"Imagen procesada guardada en: {ruta_guardar_imagen}")