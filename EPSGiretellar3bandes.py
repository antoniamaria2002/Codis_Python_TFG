# -*- coding: utf-8 -*-
import rasterio 
import numpy as np
import cv2
import os
from rasterio.warp import calculate_default_transform, reproject, Resampling

# Ruta de entrada
input_tif = r"C:\Users\Moi\Desktop\tfg\videoimatgesvilanova\3juliol2023_psscene_visual\PSScene\20230703_095132_47_24c9_3B_Visual_clip.tif"
# Directorio de salida
output_dir = r"C:\Users\Moi\Desktop\tfg\videoimatgesvilanova\3juliol2023_psscene_visual\PSScene\nova"
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
        "count": 3 
    })

    output_tif = os.path.join(output_dir, "reprojected_image_RGB.tif")
    with rasterio.open(output_tif, "w", **kwargs) as dst:
        for i in range(1, 4):  
            reproject(
                source=rasterio.band(src, i),
                destination=rasterio.band(dst, i),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs="EPSG:4326",
                resampling=Resampling.nearest
            )

print(f" Imatge reproyectada a EPSG:4326 i guardada a {output_tif}")

# Llegir les 3 bandes (Red, Green, Blue)
with rasterio.open(output_tif) as src:
    img = src.read([1, 2, 3])  # Red, Green, Blue
    img = np.moveaxis(img, 0, -1)

# Funció per normalitzar una banda amb ajust opcional de percentils
def normalize_band(band, low=2, high=98, gamma=1.0):
    band_min, band_max = np.percentile(band, (low, high))
    normalized = np.clip((band - band_min) / (band_max - band_min), 0, 1)
    if gamma != 1.0:
        normalized = np.power(normalized, gamma)
    return (normalized * 255).astype(np.uint8)

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

# Noms de bandes
band_names = ['Red', 'Green', 'Blue']

# Processar i guardar només les imatges retallades
normalized_bands = []
for i, band_name in enumerate(band_names):
    norm_band = normalize_band(img[:, :, i], low=1, high=97, gamma=0.8 if band_name == 'Red' else 1.0)
    band_image = cv2.cvtColor(norm_band, cv2.COLOR_GRAY2BGR)
    cropped_band = crop_black_borders(band_image)
    output_png = os.path.join(output_dir, f"{band_name}_cropped.png")
    cv2.imwrite(output_png, cropped_band)
    print(f" Banda '{band_name}' recortada guardada com '{output_png}'")
    normalized_bands.append(norm_band)

# Crear composició RGB i retallar-la
rgb_image = np.stack(normalized_bands, axis=-1)
rgb_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
rgb_cropped = crop_black_borders(rgb_bgr)
output_png_rgb_cropped = os.path.join(output_dir, "RGB_Composite_cropped.png")
cv2.imwrite(output_png_rgb_cropped, rgb_cropped)
print(f" RGB recortada guardada com '{output_png_rgb_cropped}'")

print(" Totes les imatges retallades han estat guardades correctament.")
