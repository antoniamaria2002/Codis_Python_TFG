# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 22:49:37 2025

@author: Moi
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Paràmetres del mètode Region Growing
SEED_POINT = (100, 100)       
THRESHOLD = 15                
CONNECTIVITY = 8              

# Ruta de la imatge d'entrada
image_path = r"C:\Users\Moi\Desktop\tfg\videoimatgesvilanova\posidoniavilanova3juliol20234bandas\codis\Blue-Nir_resta2023\Blue_Corrected2_cropped.png"

# Ruta de sortida on es guardarà la imatge segmentada
output_path = r"C:\Users\Moi\Desktop\tfg\videoimatgesvilanova\posidoniavilanova3juliol20234bandas\codis\Posidonia_Region_Growing.png"

# Carregar la imatge en escala de grisos
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print(" Error: No es pot carregar la imatge.")
else:
    print(" Imatge carregada correctament.")

    # Convertir a escala de grisos si és necessari
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image  # Ja és en escala de grisos

    # Millora del contrast mitjançant CLAHE i correcció gamma
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)

    gamma = 5
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in np.arange(0, 256)]).astype("uint8")
    gray_contrast = cv2.LUT(gray_clahe, table)

    # Retallar la part inferior central de la imatge per enfocar l’anàlisi
    height, width = gray.shape[:2]
    gray_cropped = gray_contrast[(height // 2) - 190:height - 300, :]

    # Inicialització de la màscara i la matriu de píxels visitats
    height, width = gray_cropped.shape
    mask = np.zeros_like(gray_cropped, dtype=np.uint8)
    visited = np.zeros_like(gray_cropped, dtype=np.uint8)
    seed_queue = [SEED_POINT] 

    seed_value = gray_cropped[SEED_POINT]  

    # Algorisme de Region Growing
    while seed_queue:
        y, x = seed_queue.pop(0)

        if x < 0 or x >= width or y < 0 or y >= height or visited[y, x]:
            continue

        if abs(int(gray_cropped[y, x]) - int(seed_value)) < THRESHOLD:
            mask[y, x] = 255
            visited[y, x] = 1

            # Determinar veïns segons la connectivitat
            if CONNECTIVITY == 8:
                neighbors = [(y-1, x-1), (y-1, x), (y-1, x+1),
                             (y, x-1),           (y, x+1),
                             (y+1, x-1), (y+1, x), (y+1, x+1)]
            else:
                neighbors = [(y-1, x), (y, x-1), (y, x+1), (y+1, x)]

            for ny, nx in neighbors:
                if 0 <= nx < width and 0 <= ny < height and not visited[ny, nx]:
                    seed_queue.append((ny, nx))

    # Guardar la màscara resultant (segmentació de Posidònia)
    cv2.imwrite(output_path, mask)

    # Mostrar imatge original i resultat de la segmentació
    fig, ax = plt.subplots(2, 1, figsize=(12, 6))
    ax[0].imshow(gray_cropped, cmap="gray")
    ax[0].set_title("Imatge original (retallada)")
    ax[0].axis("off")

    ax[1].imshow(mask, cmap="gray")
    ax[1].set_title("Segmentació de la Posidònia (Region Growing)")
    ax[1].axis("off")

    plt.tight_layout()
    plt.show()

    print(f"*Posidònia* detectada i guardada com a: '{output_path}'")
