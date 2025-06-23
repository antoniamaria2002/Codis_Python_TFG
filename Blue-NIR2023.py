# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 20:36:22 2025

@author: Moi
"""

import cv2
import os
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

# Funció per retallar vores negres
def crop_black_borders(image):
    gray = image.copy()
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        return image[y:y+h, x:x+w]
    else:
        return image

# Rutes
input_tif = r"C:\Users\Moi\Desktop\tfg\videoimatgesvilanova\posidoniavilanova3juliol20234bandas\PSScene\20230703_095132_47_24c9_3B_AnalyticMS_SR_clip.tif"
output_dir = r"C:\Users\Moi\Desktop\tfg\videoimatgesvilanova\posidoniavilanova3juliol20234bandas\codis\Blue-Nir_resta2023"
os.makedirs(output_dir, exist_ok=True)

# Reprojectar a EPSG:4326 i llegir les bandes
with rasterio.open(input_tif) as src:
    transform, width, height = calculate_default_transform(
        src.crs, "EPSG:4326", src.width, src.height, *src.bounds)
    kwargs = src.meta.copy()
    kwargs.update({"crs": "EPSG:4326", "transform": transform, "width": width, "height": height})
    output_tif = os.path.join(output_dir, "temp_reprojected.tif")
    with rasterio.open(output_tif, "w", **kwargs) as dst:
        for i in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, i),
                destination=rasterio.band(dst, i),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs="EPSG:4326",
                resampling=Resampling.nearest
            )

# Processar només Blue i NIR
with rasterio.open(output_tif) as image:
    blue = image.read(1).astype(np.float32)  # Band 1 = Blue
    nir = image.read(4).astype(np.float32)   # Band 4 = NIR

# Normalitzar
def normalize_band(band):
    band_min, band_max = np.percentile(band, (2, 98))
    return np.clip((band - band_min) / (band_max - band_min) * 255, 0, 255).astype(np.uint8)

blue_norm = normalize_band(blue)
nir_norm = normalize_band(nir)

# Correcció gamma a NIR
gamma = 1.5
invGamma = 1.0 / gamma
table = np.array([(i / 255.0) ** invGamma * 255 for i in np.arange(0, 256)]).astype("uint8")
nir_gamma_corrected = cv2.LUT(nir_norm, table)

# Resta ponderada: Blue - NIR
blue_corrected = cv2.addWeighted(blue_norm, 1.0, nir_gamma_corrected, -1.0, 0)

# Retallar vores negres
blue_cropped = crop_black_borders(blue_corrected)

# Guardar la imatge final
output_image_path = os.path.join(output_dir, "Blue_Corrected2_cropped.png")
cv2.imwrite(output_image_path, blue_cropped)
print(f"Imatge final guardada: {output_image_path}")
