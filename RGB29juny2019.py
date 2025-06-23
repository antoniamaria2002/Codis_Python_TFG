# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 14:19:17 2025

@author: Moi
"""
import rasterio    
import numpy as np
import cv2
import os

# Rutes de les imatges d'entrada
path_1 = r"C:\Users\Moi\Desktop\tfg\videoimatgesvilanova\29juny2019\PSScene\20190629_101918_1001_3B_AnalyticMS_SR_clip.tif"
path_2 = r"C:\Users\Moi\Desktop\tfg\videoimatgesvilanova\29juny2019\PSScene\20190629_101919_1001_3B_AnalyticMS_SR_clip2.tif"

# Directoris de sortida
directory_1 = r"C:\Users\Moi\Desktop\tfg\videoimatgesvilanova\29juny2019\imatges\tif1"
directory_2 = r"C:\Users\Moi\Desktop\tfg\videoimatgesvilanova\29juny2019\imatges\tif2"

# Crear els directoris si no existeixen
os.makedirs(directory_1, exist_ok=True)
os.makedirs(directory_2, exist_ok=True)

# Funció per llegir i retornar les bandes RGB d'una imatge
def llegir_i_normalitzar_rgb(path):
    with rasterio.open(path) as image:
        red = image.read(3).astype(np.float32)
        green = image.read(2).astype(np.float32)
        blue = image.read(1).astype(np.float32)
    return red, green, blue

# Funció per normalitzar una banda a 8 bits (0-255)
def normalitzar_banda(banda):
    b_min, b_max = np.percentile(banda, (2, 98))
    return np.clip((banda - b_min) / (b_max - b_min) * 255, 0, 255).astype(np.uint8)

# Processar i guardar només la composició RGB de la primera imatge
r1, g1, b1 = llegir_i_normalitzar_rgb(path_1)
rgb1 = np.stack([normalitzar_banda(r1), normalitzar_banda(g1), normalitzar_banda(b1)], axis=-1)
cv2.imwrite(os.path.join(directory_1, "Composicio_RGB_imatge1.png"), cv2.cvtColor(rgb1, cv2.COLOR_RGB2BGR))
print("Composició RGB de la imatge 1 desada correctament.")

# Processar i guardar la composició RGB de la segona imatge
r2, g2, b2 = llegir_i_normalitzar_rgb(path_2)
rgb2 = np.stack([normalitzar_banda(r2), normalitzar_banda(g2), normalitzar_banda(b2)], axis=-1)
cv2.imwrite(os.path.join(directory_2, "Composicio_RGB_imatge2.png"), cv2.cvtColor(rgb2, cv2.COLOR_RGB2BGR))
print(" Composició RGB de la imatge 2 desada correctament.")
