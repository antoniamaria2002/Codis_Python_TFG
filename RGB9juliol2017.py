# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 18:01:27 2025

@author: Moi
"""

import rasterio
import numpy as np
import cv2
import os

# Ruta del fitxer d’entrada
input_tif = r"C:\Users\Moi\Desktop\tfg\videoimatgesvilanova\9juliol20174bandas\PSScene\20170709_095358_1021_3B_AnalyticMS_SR_clip.tif"
# Carpeta de sortida
output_dir = r"C:\Users\Moi\Desktop\tfg\videoimatgesvilanova\9juliol20174bandas\PSScene\imatges"
os.makedirs(output_dir, exist_ok=True)

# Funció per normalitzar una banda a 8 bits (0–255)
def normalitzar_banda(banda):
    b_min, b_max = np.percentile(banda, (2, 98))
    return np.clip((banda - b_min) / (b_max - b_min) * 255, 0, 255).astype(np.uint8)

# Obrir el TIFF i llegir les bandes 1 (Blue), 2 (Green) i 3 (Red)
with rasterio.open(input_tif) as src:
    blue = src.read(1).astype(np.float32)
    green = src.read(2).astype(np.float32)
    red = src.read(3).astype(np.float32)

# Normalitzar les bandes
blue_8bit = normalitzar_banda(blue)
green_8bit = normalitzar_banda(green)
red_8bit = normalitzar_banda(red)

# Crear composició RGB (ordre: Red, Green, Blue)
rgb_image = np.stack([red_8bit, green_8bit, blue_8bit], axis=-1)

# Guardar la composició RGB
output_rgb = os.path.join(output_dir, "Composicioo_RGB.png")
cv2.imwrite(output_rgb, cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
print(f" Composició RGB desada com: {output_rgb}")
