# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 12:06:30 2025

@author: Moi
"""

import geopandas as gpd

# Ruta del fitxer de polígons (Shapefile)
shapefile_path = r'C:\Users\Moi\Desktop\tfg\videoimatgesvilanova\posidoniavilanova3juliol20234bandas\codis\codis\polignsgeneralitat.shp'

# Carregar el fitxer de polígons (Shapefile)
gdf_poligons = gpd.read_file(shapefile_path)

# Mostrar les primeres files per veure les dades
print("Primeres files del Shapefile:")
print(gdf_poligons.head())

# Mostrar les columnes disponibles (camps del Shapefile)
print("\nColumnes del Shapefile:")
print(gdf_poligons.columns)

# Mostrar la geometria de les primeres files
print("\nGeometria dels primers polígons:")
print(gdf_poligons.geometry.head())

# Mostrar informació sobre el sistema de coordenades (CRS)
print("\nSistema de coordenades del Shapefile (CRS):")
print(gdf_poligons.crs)
