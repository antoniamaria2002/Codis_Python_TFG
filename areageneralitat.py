# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 11:36:22 2025

@author: Moi
"""
import geopandas as gpd
import numpy as np
from rasterio.features import rasterize
from rasterio.transform import from_origin
from PIL import Image
import math

# Rutes d'entrada i sortida
shapefile_path = r"C:\Users\Moi\Desktop\tfg\videoimatgesvilanova\posidoniavilanova3juliol20234bandas\codis\codis\polignsgeneralitat.shp"
image_path = r"C:\Users\Moi\Desktop\tfg\videoimatgesvilanova\posidoniavilanova3juliol20234bandas\codis\Blue-Nir_resta2023\Blue_Corrected2_cropped.png"
output_png_path = r"C:\Users\Moi\Desktop\tfg\videoimatgesvilanova\posidoniavilanova3juliol20234bandas\codis\output.png"

# Límits de l'àrea de la imatge
bounds = [(1.64385338, 41.15793979), (1.8365719, 41.15793979),
          (1.8365719, 41.21405976), (1.64385338, 41.21405976)]
minx, miny = bounds[0]
maxx, maxy = bounds[2]

# Carregar la imatge original per obtenir mida en píxels
with Image.open(image_path) as img:
    width, height = img.size  

# Calcular la resolució en graus per píxel basant-se en els límits
res_x = (maxx - minx) / width
res_y = (maxy - miny) / height

# Calcular la resolució en metres per píxel
# Resolució a la latitud (graus a metres)
meters_per_degree_lat = 111320  
meters_per_degree_long = 111320  

# La resolució de longitud varia segons la latitud
latitude = 41.186  
meters_per_degree_long_at_lat = meters_per_degree_long * math.cos(math.radians(latitude))

# Resoldre la resolució de píxel en metres
res_x_meters = res_x * meters_per_degree_long_at_lat  
res_y_meters = res_y * meters_per_degree_lat  

# Mostrar les coordenades dels extrems, la resolució en graus i la resolució en metres
print(f"Coordenades mínimes (esquerra inferior): ({minx}, {miny})")
print(f"Coordenades màximes (dreta superior): ({maxx}, {maxy})")
print(f"Resoldre en X (graus per píxel): {res_x}")
print(f"Resoldre en Y (graus per píxel): {res_y}")
print(f"Resoldre en X (metres per píxel): {res_x_meters} metres")
print(f"Resoldre en Y (metres per píxel): {res_y_meters} metres")

# Carregar el shapefile
gdf = gpd.read_file(shapefile_path)

# Eliminar la coordenada Z i convertir a POLYGON 2D
gdf["geometry"] = gdf["geometry"].apply(lambda geom: 
    type(geom)([(x, y) for x, y, _ in geom.exterior.coords]) if geom.has_z else geom)

# Assegurar que el shapefile està en EPSG:4326 (coordenades geogràfiques)
gdf = gdf.to_crs("EPSG:4326")

# Definir la transformació geoespacial per al raster 
transform = from_origin(minx, maxy, res_x, res_y)

# Rasteritzar els polígons amb valors 1 (polígon) i 0 (fons)
raster = rasterize(
    [(geom, 1) for geom in gdf.geometry],
    out_shape=(height, width),
    transform=transform,  
    fill=0,
    dtype=np.uint8
)

# Convertir la matriu binària en una imatge PNG (0 = negre, 1 = blanc)
img_out = Image.fromarray(raster * 255)  
img_out.save(output_png_path)

print(f"Imatge PNG guardada en {output_png_path}")

