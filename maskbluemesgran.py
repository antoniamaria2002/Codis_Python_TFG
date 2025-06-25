# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 14:01:07 2025

@author: Moi
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Ruta de la imatge
image_path = r"C:\Users\Moi\Desktop\tfg\videoimatgesvilanova\posidoniavilanova3juliol20234bandas\codis\resultat0_2023_mesgran.png"

# Carregar la imatge de la banda blava amb la màscara aplicada 
blue_masked = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Comprovació bàsica per verificar si s'ha carregat la imatge correctament
if blue_masked is None:
    raise ValueError(f"No s'ha pogut carregar la imatge. Comprova la ruta: {image_path}")

# Pas 1: Extreure només els píxels diferents de 0 
blue_values = blue_masked[blue_masked > 0]

# Pas 2: Crear un histograma per observar el contrast entre aigua i Posidònia
plt.hist(blue_values, bins=40, color='blue', alpha=0.7)
plt.title("Histograma de la Banda Blava dins la Màscara de Posidònia")
plt.xlabel("Intensitat del píxel")
plt.ylabel("Freqüència")
plt.grid(True)
plt.show()

# Pas 3: Aplicar un llindar (threshold) per detectar Posidònia
threshold = 55
posidonia_binary = np.zeros_like(blue_masked)
posidonia_binary[(blue_masked > 0) & (blue_masked < threshold)] = 255  


# Pas 4: Guardar i visualitzar el resultat
output_image_path = r"C:\Users\Moi\Desktop\tfg\videoimatgesvilanova\posidoniavilanova3juliol20234bandas\codis\Detected_Posidonia_2023_mesgran.png"
cv2.imwrite(output_image_path, posidonia_binary)

# Mostrar les imatges: original millorada i resultat de la detecció
fig, ax = plt.subplots(2, 1, figsize=(24, 12))
ax[0].imshow(blue_masked, cmap="gray")
ax[0].set_title("Imatge Millorada")
ax[0].axis("off")

ax[1].imshow(posidonia_binary, cmap="gray")
ax[1].set_title("Àrea Detectada de *Posidònia*")
ax[1].axis("off")

plt.tight_layout()
plt.show()
