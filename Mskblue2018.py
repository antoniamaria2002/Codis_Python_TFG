# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 13:59:51 2025

@author: Moi
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Ruta de la imatge
image_path = r"C:\Users\Moi\Desktop\tfg\videoimatgesvilanova\10juriol2018\imatges\resultat0_2018.png"

# Carregar la imatge de la banda blava amb màscara aplicada (fora de la cartografia = 0)
blue_masked = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  

# Verificació de càrrega
if blue_masked is None:
    raise ValueError(f"Image not loaded. Check path: {image_path}")

# Pas 1: Extreure només els píxels diferents de zero (dins la màscara)
blue_values = blue_masked[blue_masked > 0]

# Pas 2: Representar l’histograma per veure el contrast entre aigua i Posidònia
plt.hist(blue_values, bins=40, color='blue', alpha=0.7)
plt.title("Blue Band Histogram Inside Posidonia Mask")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# Pas 3: Aplicar un llindar
threshold = 30  
posidonia_binary = np.zeros_like(blue_masked)
posidonia_binary[(blue_masked > 0) & (blue_masked < threshold)] = 255  

# Pas 4: Guardar i visualitzar el resultat
output_image_path = r"C:\Users\Moi\Desktop\tfg\videoimatgesvilanova\10juriol2018\imatges\Detected_Posidonia_Only_Blue_2018.png"
cv2.imwrite(output_image_path, posidonia_binary)

# Mostrar la imatge original i la binària resultant
fig, ax = plt.subplots(2, 1, figsize=(24, 12))
ax[0].imshow(blue_masked, cmap="gray")
ax[0].set_title("Enhanced Image")
ax[0].axis("off")

ax[1].imshow(posidonia_binary, cmap="gray")
ax[1].set_title("Detected *Posidonia* Area")
ax[1].axis("off")

plt.tight_layout()
plt.show()
