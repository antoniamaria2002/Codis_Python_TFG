# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 12:17:23 2025

@author: Moi
"""

from PIL import Image
import numpy as np

# Carregar les imatges
image_path_1 = r'C:\Users\Moi\Desktop\tfg\videoimatgesvilanova\posidoniavilanova3juliol20234bandas\codis\output.png'  
image_path_2 = r'C:\Users\Moi\Desktop\tfg\videoimatgesvilanova\posidoniavilanova3juliol20234bandas\codis\Blue-Nir_resta2023\Blue_Corrected2_cropped.png'  

# Obrir les imatges
img1 = Image.open(image_path_1).convert('L')  
img2 = Image.open(image_path_2).convert('L')  

# Convertir les imatges a arrays de numpy
arr1 = np.array(img1)
arr2 = np.array(img2)

# Comprovar que les dimensions de les imatges són iguals
if arr1.shape != arr2.shape:
    raise ValueError("Les dimensions de les imatges no coincideixen!")

# Crear una màscara on els píxels de la primera imatge siguin 255 
mask = (arr1 == 255)

# Comprovació de valors: verificar que hi ha píxels amb valor 255 a la màscara
if np.sum(mask) == 0:
    print("No hi ha píxels amb valor 255 a la màscara.")

# Substituir els píxels de la segona imatge amb valor 255 (blanc) allà on la màscara sigui True
arr2[mask] = 255  # Píxels de la segona imatge es faran blancs on la primera imatge sigui 255

# Ajustar la intensitat de la imatge per fer que es vegin millor els valors
arr2 = np.clip(arr2, 0, 255)  

# Convertir l'array modificat a imatge
result_img = Image.fromarray(arr2)

# Desa la nova imatge
result_img.save(r'C:\Users\Moi\Desktop\tfg\videoimatgesvilanova\posidoniavilanova3juliol20234bandas\codis\resultat_visible.png')

# Mostrar la nova imatge 
result_img.show()
