# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 18:22:42 2025

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
input_tif = r"C:\Users\Moi\Desktop\tfg\videoimatgesvilanova\10juriol2018\PSScene\20180710_100256_0f49_3B_AnalyticMS_SR_clip.tif"
output_dir = r"C:\Users\Moi\Desktop\tfg\videoimatgesvilanova\10juriol2018\imatges\Blue_NIR_resta"
os.makedirs(output_dir, exist_ok=True)

# Reprojectar a EPSG:4326
with rasterio.open(input_tif) as src:
    transform, width, height = calculate_default_transform(
        src.crs, "EPSG:4326", src.width, src.height, *src.bounds)
    kwargs = src.meta.copy()
    kwargs.update({"crs": "EPSG:4326", "transform": transform, "width": width, "height": height})
    output_tif = os.path.join(output_dir, "reprojected.tif")
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

print(" Imatge reproyectada i guardada.")

# Llegir les bandes
with rasterio.open(output_tif) as img:
    blue = img.read(1).astype(np.float32)
    nir = img.read(4).astype(np.float32)

# Normalitzar
def normalize_band(band):
    band_min, band_max = np.percentile(band, (2, 98))
    return np.clip((band - band_min) / (band_max - band_min), 0, 1) * 255

blue_norm = normalize_band(blue).astype(np.uint8)
nir_norm = normalize_band(nir).astype(np.uint8)

# Correcció gamma a NIR
gamma = 1.5
invGamma = 1.0 / gamma
table = np.array([(i / 255.0) ** invGamma * 255 for i in np.arange(256)]).astype("uint8")
nir_gamma = cv2.LUT(nir_norm, table)

# Resta NIR a Blue (ponderada)
alpha = 1
blue_corrected = cv2.addWeighted(blue_norm, 1.0, nir_gamma, -alpha, 0)

# Retallar i guardar
blue_cropped = crop_black_borders(blue_corrected)
output_path = os.path.join(output_dir, "Blue_Corrected2_cropped.png")
cv2.imwrite(output_path, blue_cropped)

print(f"Imatge final guardada: {output_path}")
