# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 18:07:07 2025

@author: Moi
"""

import rasterio 
import numpy as np
import cv2
import os

# Ruta del fitxer TIFF
path = r"C:\Users\Moi\Desktop\tfg\videoimatgesvilanova\5agost20194bandas\PSScene\20190805_103537_29_105d_3B_AnalyticMS_SR_clip.tif"

# Carpeta de sortida 
directory = os.path.dirname(path)

# Funció per normalitzar una banda (0–255)
def normalitzar_banda(banda):
    b_min, b_max = np.percentile(banda, (2, 98))
    return np.clip((banda - b_min) / (b_max - b_min) * 255, 0, 255).astype(np.uint8)

# Llegir les bandes Blue (1), Green (2) i Red (3)
with rasterio.open(path) as imatge:
    blue = imatge.read(1).astype(np.float32)
    green = imatge.read(2).astype(np.float32)
    red = imatge.read(3).astype(np.float32)

# Normalitzar bandes
blue_8bit = normalitzar_banda(blue)
green_8bit = normalitzar_banda(green)
red_8bit = normalitzar_banda(red)

# Crear composició RGB (ordre: Red, Green, Blue)
rgb_8bit = np.stack([red_8bit, green_8bit, blue_8bit], axis=-1)

# Desar la imatge RGB
sortida_rgb = os.path.join(directory, "Composicioo_RGB.png")
cv2.imwrite(sortida_rgb, cv2.cvtColor(rgb_8bit, cv2.COLOR_RGB2BGR))
print(f" Composició RGB desada com: {sortida_rgb}")
