# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 13:17:02 2025

@author: Moi
"""
import rasterio
import numpy as np
import cv2
import os
from rasterio.warp import calculate_default_transform, reproject, Resampling

# Rutes de les imatges TIFF
path1 = r"C:\Users\Moi\Desktop\tfg\videoimatgesvilanova\19agost20184bandas\PSScene\20180819_095714_1054_3B_AnalyticMS_SR_clip.tif"
path2 = r"C:\Users\Moi\Desktop\tfg\videoimatgesvilanova\19agost20184bandas\PSScene\20180819_101225_103a_3B_AnalyticMS_SR_clip.tif"

# Directori de sortida
directori_sortida = os.path.dirname(path1)

# Funció per llegir bandes
def llegir_bandes(path):
    with rasterio.open(path) as imatge:
        bandes = [imatge.read(i + 1).astype(np.float32) for i in range(imatge.count)]
        transform = imatge.transform
        crs = imatge.crs
    return bandes, transform, crs

# Funció per normalitzar
def normalitzar_banda(banda):
    b_min, b_max = np.percentile(banda, (2, 98))
    return np.clip((banda - b_min) / (b_max - b_min) * 255, 0, 255).astype(np.uint8)

# Funció per combinar dues bandes (valor màxim)
def combinar_bandes(b1, b2):
    return np.maximum(b1, b2)

# Llegir les bandes i metadades
bandes1, transform1, crs1 = llegir_bandes(path1)
bandes2, transform2, crs2 = llegir_bandes(path2)

# Reprojectar si cal
if crs1 != crs2:
    with rasterio.open(path2) as src2:
        transform2, ample2, alt2 = calculate_default_transform(
            src2.crs, crs1, src2.width, src2.height, *src2.bounds)
        bandes2_reproj = np.zeros((len(bandes2), alt2, ample2), dtype=np.float32)
        for i in range(len(bandes2)):
            reproject(
                source=rasterio.band(src2, i + 1),
                destination=bandes2_reproj[i],
                src_transform=src2.transform,
                src_crs=src2.crs,
                dst_transform=transform2,
                dst_crs=crs1,
                resampling=Resampling.nearest
            )
else:
    bandes2_reproj = bandes2

# Fer coincidir mides
alt = max(bandes1[0].shape[0], bandes2_reproj[0].shape[0])
ample = max(bandes1[0].shape[1], bandes2_reproj[0].shape[1])
bandes1_resize = [cv2.resize(b, (ample, alt)) for b in bandes1]
bandes2_resize = [cv2.resize(b, (ample, alt)) for b in bandes2_reproj]

# Bandes: Band 1 = Blue, Band 2 = Green, Band 3 = Red, Band 4 = NIR (no usada)

# Combinar bandes RGB
banda_blue = combinar_bandes(bandes1_resize[0], bandes2_resize[0])   # Band 1 → Blue
banda_green = combinar_bandes(bandes1_resize[1], bandes2_resize[1])  # Band 2 → Green
banda_red = combinar_bandes(bandes1_resize[2], bandes2_resize[2])    # Band 3 → Red

# Normalitzar i apilar RGB
rgb_comb = np.stack([
    normalitzar_banda(banda_red),
    normalitzar_banda(banda_green),
    normalitzar_banda(banda_blue)
], axis=-1)

# Desar composició RGB superposada
sortida_rgb = os.path.join(directori_sortida, "Composicio_RGB_Superposadaaa.png")
cv2.imwrite(sortida_rgb, cv2.cvtColor(rgb_comb, cv2.COLOR_RGB2BGR))
print(f"✅ Composició RGB superposada desada com: {sortida_rgb}")

