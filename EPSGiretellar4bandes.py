# -*- coding: utf-8 -*-
"""
Created on Sun May 25 23:00:13 2025

@author: Moi
"""
import os
import cv2
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
        return image[y:y + h, x:x + w]
    else:
        return image

# Funció per normalitzar amb ajust de percentils i gamma
def normalize_band(band, low=2, high=98, gamma=1.0):
    band_min, band_max = np.percentile(band, (low, high))
    normalized = np.clip((band - band_min) / (band_max - band_min), 0, 1)
    if gamma != 1.0:
        normalized = np.power(normalized, gamma)
    return (normalized * 255).astype(np.uint8)

# Rutes d’entrada i sortida
input_tif = r"C:\Users\Moi\Desktop\tfg\videoimatgesvilanova\posidoniavilanova3juliol20234bandas\PSScene\20230703_095132_47_24c9_3B_AnalyticMS_SR_clip.tif"
output_dir = r"C:\Users\Moi\Desktop\tfg\videoimatgesvilanova\posidoniavilanova3juliol20234bandas\codis\nous"
os.makedirs(output_dir, exist_ok=True)

# Reprojectar la imatge a EPSG:4326
with rasterio.open(input_tif) as src:
    transform, width, height = calculate_default_transform(
        src.crs, "EPSG:4326", src.width, src.height, *src.bounds)
    kwargs = src.meta.copy()
    kwargs.update({
        "crs": "EPSG:4326",
        "transform": transform,
        "width": width,
        "height": height,
        "count": 4  # Blue, Green, Red, NIR
    })
    output_tif = os.path.join(output_dir, "reprojected_image_4326.tif")
    with rasterio.open(output_tif, "w", **kwargs) as dst:
        for i in range(1, 5):
            reproject(
                source=rasterio.band(src, i),
                destination=rasterio.band(dst, i),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs="EPSG:4326",
                resampling=Resampling.nearest
            )
print(f" Imatge reproyectada i guardada a: {output_tif}")

# Llegir les bandes
with rasterio.open(output_tif) as image:
    bands = [image.read(i + 1).astype(np.float32) for i in range(4)]  

band_names = ["Blue", "Green", "Red", "NIR"]
cropped_bands = []

# Normalitzar i retallar 
for i, name in enumerate(band_names):
    # Ajusts específics per NIR
    if name == "NIR":
        norm = normalize_band(bands[i], low=0.5, high=92, gamma=0.8)
    else:
        norm = normalize_band(bands[i])
    
    # Retallar vores negres
    cropped = crop_black_borders(norm)
    cropped_path = os.path.join(output_dir, f"{name}_cropped.png")
    cv2.imwrite(cropped_path, cropped)
    print(f"Banda retallada guardada: {cropped_path}")
    cropped_bands.append(cropped)

# Igualar dimensions entre imatges retallades
min_height = min(b.shape[0] for b in cropped_bands)
min_width = min(b.shape[1] for b in cropped_bands)
resized_bands = [cv2.resize(b, (min_width, min_height)) for b in cropped_bands]

# Crear composició RGB
rgb_image = np.stack([resized_bands[2], resized_bands[1], resized_bands[0]], axis=-1)
rgb_path = os.path.join(output_dir, "RGB_Composite_cropped.png")
cv2.imwrite(rgb_path, cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
print(f"Composició RGB guardada com: {rgb_path}")
