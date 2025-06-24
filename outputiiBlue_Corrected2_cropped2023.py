# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 09:52:44 2025

@author: Moi
"""

from PIL import Image
import numpy as np

# Carregar les imatges
mask_path = r"C:\Users\Moi\Desktop\tfg\videoimatgesvilanova\posidoniavilanova3juliol20234bandas\codis\output.png"
grayscale_path = r"C:\Users\Moi\Desktop\tfg\videoimatgesvilanova\posidoniavilanova3juliol20234bandas\codis\Blue-Nir_resta2023\Blue_Corrected2_cropped.png"
output_path = r"C:\Users\Moi\Desktop\tfg\videoimatgesvilanova\posidoniavilanova3juliol20234bandas\codis\resultat0.png"

mask = Image.open(mask_path).convert("L")  
grayscale = Image.open(grayscale_path).convert("L")

# Convertir a arrays numpy
mask_array = np.array(mask)
grayscale_array = np.array(grayscale)

# Aplicar la condició: si mask és 0 → 0, si és 255 → valor de la imatge en escala de grisos
output_array = np.where(mask_array == 0, 0, grayscale_array)

# Convertir a imatge
output_image = Image.fromarray(output_array.astype(np.uint8))

# Guardar la imatge
output_image.save(output_path)

print(f"Imatge generada i guardada a {output_path}")
