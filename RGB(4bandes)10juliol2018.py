# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 13:27:13 2025

@author: Moi
"""
import rasterio  
import numpy as np
import cv2
import os

# Ruta de la imatge TIFF
path = r"C:\Users\Moi\Desktop\tfg\videoimatgesvilanova\10juriol2018\PSScene\20180710_100256_0f49_3B_AnalyticMS_SR_clip.tif"

# Directori de sortida
directory = r"C:\Users\Moi\Desktop\tfg\videoimatgesvilanova\10juriol2018\imatges"
os.makedirs(directory, exist_ok=True)

# Obrir la imatge i llegir les bandes
with rasterio.open(path) as image:
    bands = [image.read(i + 1).astype(np.float32) for i in range(image.count)]  # 4 bandes: Blue, Green, Red, NIR

# Funció per normalitzar una banda (a 8-bit)
def normalize_band(band):
    band_min, band_max = np.percentile(band, (2, 98))
    return np.clip((band - band_min) / (band_max - band_min) * 255, 0, 255).astype(np.uint8)

# Crear la composició RGB (Red, Green, Blue → bandes 3, 2, 1)
rgb_8bit = np.stack([
    normalize_band(bands[2]),
    normalize_band(bands[1]),
    normalize_band(bands[0])   
], axis=-1)

# Guardar la composició RGB
rgb_path = os.path.join(directory, "RGB_Composite.png")
cv2.imwrite(rgb_path, cv2.cvtColor(rgb_8bit, cv2.COLOR_RGB2BGR))
print(f" Imatge RGB guardada com: {rgb_path}")
