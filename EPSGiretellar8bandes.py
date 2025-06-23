# -*- coding: utf-8 -*-
"""
Created on Mon May 26 15:32:26 2025

@author: Moi
"""

import os 
import numpy as np
import cv2
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

# Rutes
input_tif = r"C:\Users\Moi\Desktop\tfg\videoimatgesvilanova\posidoniavilanova3juliol2023_8bandas\PSScene\20230703_095132_47_24c9_3B_AnalyticMS_SR_8b_clip.tif"
output_dir = r"C:\Users\Moi\Desktop\tfg\videoimatgesvilanova\posidoniavilanova3juliol2023_8bandas\PSScene\EPSG_Cortar\nova"
os.makedirs(output_dir, exist_ok=True)

# Reprojectar la imatge a EPSG:4326
with rasterio.open(input_tif) as src:
    transform, width, height = calculate_default_transform(
        src.crs, "EPSG:4326", src.width, src.height, *src.bounds)
    kwargs = src.meta.copy()
    kwargs.update({"crs": "EPSG:4326", "transform": transform, "width": width, "height": height})

    output_tif = os.path.join(output_dir, "reprojected_image_4326.tif")
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
print(f"Imatge reproyectada guardada a: {output_tif}")

# Llegir les bandes reproyectades
with rasterio.open(output_tif) as src:
    img = src.read()  # (8, height, width)
    img = np.moveaxis(img, 0, -1)  # (height, width, 8)

# Noms de les bandes
band_names = [
    "CoastalBlue",  # Band 1
    "Blue",         # Band 2
    "GreenI",       # Band 3
    "Green",        # Band 4
    "Yellow",       # Band 5
    "Red",          # Band 6
    "RedEdge",      # Band 7
    "NIR"           # Band 8
]

# Funció per normalitzar
def normalize_band(band):
    band_min, band_max = np.percentile(band, (2, 98))
    return np.clip((band - band_min) / (band_max - band_min) * 255, 0, 255).astype(np.uint8)

# Funció per retallar vores negres
def crop_black_borders(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        return image[y:y+h, x:x+w]
    else:
        return image

# Processar i guardar només imatges retallades
for i, band_name in enumerate(band_names):
    norm_band = normalize_band(img[:, :, i])
    norm_bgr = cv2.cvtColor(norm_band, cv2.COLOR_GRAY2BGR)
    cropped = crop_black_borders(norm_bgr)
    cropped_path = os.path.join(output_dir, f"{band_name}_cropped.png")
    cv2.imwrite(cropped_path, cropped)
    print(f"Banda '{band_name}' recortada guardada com: {cropped_path}")

# Composició RGB (Red = 6, Green = 4, Blue = 2)
rgb_image = np.stack([
    normalize_band(img[:, :, 5]),  # Red (Band 6)
    normalize_band(img[:, :, 3]),  # Green (Band 4)
    normalize_band(img[:, :, 1])   # Blue (Band 2)
], axis=-1)

# Retallar i guardar la composició RGB
rgb_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
rgb_cropped = crop_black_borders(rgb_bgr)
output_rgb_cropped = os.path.join(output_dir, "RGB_Composite_cropped.png")
cv2.imwrite(output_rgb_cropped, rgb_cropped)
print(f"RGB recortada guardada com: {output_rgb_cropped}")

print("Totes les bandes i la imatge RGB han estat processades i guardades només en versió retallada.")
