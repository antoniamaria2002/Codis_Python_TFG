# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 16:28:04 2025

@author: Moi
"""
import rasterio
import numpy as np
import cv2
import os
from rasterio.warp import calculate_default_transform, reproject, Resampling

# Rutes de les imatges d'entrada
path_1 = r"C:\Users\Moi\Desktop\tfg\videoimatgesvilanova\29juny2019\PSScene\20190629_101918_1001_3B_AnalyticMS_SR_clip.tif"
path_2 = r"C:\Users\Moi\Desktop\tfg\videoimatgesvilanova\29juny2019\PSScene\20190629_101919_1001_3B_AnalyticMS_SR_clip2.tif"

# Directori de sortida
directory_sortida = r"C:\Users\Moi\Desktop\tfg\videoimatgesvilanova\29juny2019\imatges\superposed2"
os.makedirs(directory_sortida, exist_ok=True)

# Funció per llegir les bandes d'una imatge
def llegir_bandes(path):
    with rasterio.open(path) as imatge:
        bandes = [imatge.read(i + 1).astype(np.float32) for i in range(imatge.count)]
        transform = imatge.transform
        crs = imatge.crs
    return bandes, transform, crs

# Funció per normalitzar una banda (a 8 bits)
def normalitzar_banda(banda):
    b_min, b_max = np.percentile(banda, (2, 98))
    return np.clip((banda - b_min) / (b_max - b_min) * 255, 0, 255).astype(np.uint8)

# Funció per combinar dues bandes mantenint el valor màxim
def combinar_bandes(b1, b2):
    return np.maximum(b1, b2)

# Llegir bandes i metadades
bandes1, transform1, crs1 = llegir_bandes(path_1)
bandes2, transform2, crs2 = llegir_bandes(path_2)

# Reprojectar si tenen CRS diferents
if crs1 != crs2:
    print("Les imatges tenen sistemes de coordenades diferents. Reprojectant la segona imatge...")
    with rasterio.open(path_2) as src2:
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

# Ajustar les dimensions
alt = max(bandes1[0].shape[0], bandes2_reproj[0].shape[0])
ample = max(bandes1[0].shape[1], bandes2_reproj[0].shape[1])
bandes1_resize = [cv2.resize(b, (ample, alt)) for b in bandes1]
bandes2_resize = [cv2.resize(b, (ample, alt)) for b in bandes2_reproj]

# Combinar bandes
bandes_comb = [combinar_bandes(bandes1_resize[i], bandes2_resize[i]) for i in range(len(bandes1))]

# Guardar TIFF amb les bandes combinades
sortida_tif = os.path.join(directory_sortida, "bandes_combinades.tif")
with rasterio.open(
    sortida_tif, 'w',
    driver='GTiff',
    count=len(bandes_comb),
    dtype=bandes_comb[0].dtype,
    width=ample,
    height=alt,
    crs=crs1,
    transform=transform1
) as dst:
    for i, banda in enumerate(bandes_comb):
        dst.write(banda, i + 1)

print(f" Arxiu TIFF amb bandes combinades desat a: {sortida_tif}")

# Crear i desar la composició RGB
rgb_comb = np.stack([
    normalitzar_banda(bandes_comb[2]),  
    normalitzar_banda(bandes_comb[1]),
    normalitzar_banda(bandes_comb[0])   
], axis=-1)

sortida_rgb = os.path.join(directory_sortida, "Composicio_RGB_combinada.png")
cv2.imwrite(sortida_rgb, cv2.cvtColor(rgb_comb, cv2.COLOR_RGB2BGR))
print(f"Composició RGB desada com: {sortida_rgb}")

print("Procés complet: imatge combinada i composició RGB guardades correctament.")
