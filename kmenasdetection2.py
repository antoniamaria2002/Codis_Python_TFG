# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 18:15:45 2025

@author: Moi
"""

import cv2
import numpy as np

# Carregar la imatge en escala de grisos
image_path = r'C:\Users\Moi\Desktop\tfg\videoimatgesvilanova\10juriol2018\imatges\resultat0_2018.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 1. Millorar el contrast utilitzant CLAHE 
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
image_contrast = clahe.apply(image)

# 2. Aplicar K-means per segmentar la imatge (per segmentar en 2 regions: fons i objecte)
# Reestructurarem la imatge en un vector de píxels
Z = image_contrast.reshape((-1, 1))

# Convertir a float32 per al K-means
Z = np.float32(Z)

# Definir criteris i aplicar K-means
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 800, 0.5)
K = 2  # Dividir la imatge en 2 grups 
_, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Convertir els resultats de K-means a la imatge original
centers = np.uint8(centers)
segmented_image = centers[labels.flatten()]
segmented_image = segmented_image.reshape(image_contrast.shape)

# 3. Guardar la imatge segmentada
output_dir = r'C:\Users\Moi\Desktop\tfg\videoimatgesvilanova\10juriol2018\imatges\detection2'
output_path_segmented = f'{output_dir}\\segmented_image.png'

cv2.imwrite(output_path_segmented, segmented_image)

print(f"Imatge segmentada guardada a: {output_path_segmented}")