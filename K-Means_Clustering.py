# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 13:51:16 2025

@author: Moi
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage import exposure, morphology

# Ruta de la imatge
image_path = r"C:\Users\Moi\Desktop\tfg\videoimatgesvilanova\10juriol2018\imatges\Blue_NIR_resta\Blue_Corrected2_cropped.png"

# Comprovar si la imatge existeix
if os.path.exists(image_path):
    image = cv2.imread(image_path)
    
    if image is None:
        print("Error: No s'ha pogut carregar la imatge. Verifica la ruta.")
    else:
        print(" Imatge carregada correctament.")

        # Convertir a escala de grisos si cal
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Millorar el contrast (CLAHE + correcció gamma)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray_clahe = clahe.apply(gray)

        gamma = 5
        invGamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** invGamma * 255 for i in np.arange(256)]).astype("uint8")
        gray_contrast = cv2.LUT(gray_clahe, table)

        # Retallar la imatge (part inferior central)
        height, width = gray.shape[:2]
        gray_cropped = gray_contrast[(height // 2) - 190:height - 300, :]

        # Preparar per a K-means
        pixel_values = gray_cropped.reshape((-1, 1)).astype(np.float32)

        # Aplicar K-means clustering
        k = 2
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 28, 2.2)
        _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Reconstruir la imatge segmentada
        segmented_image = labels.reshape(gray_cropped.shape)

        # Crear màscara binària per la Posidònia
        posidonia_cluster = np.argmax(centers)
        posidonia_mask = (segmented_image == posidonia_cluster).astype(np.uint8) * 255

        # Neteja morfològica de la màscara
        selem = morphology.disk(3)
        posidonia_mask = morphology.remove_small_objects(posidonia_mask.astype(bool), min_size=500).astype(np.uint8) * 255
        posidonia_mask = morphology.closing(posidonia_mask, selem)

        # Guardar la màscara de Posidònia
        output_directory = r"C:\Users\Moi\Desktop\tfg\videoimatgesvilanova\10juriol2018\imatges"
        output_posidonia_path = os.path.join(output_directory, "K-means_Posidonia_Area.png")
        cv2.imwrite(output_posidonia_path, posidonia_mask)
        print(f" L'àrea de *Posidònia* s'ha guardat com '{output_posidonia_path}'.")

        # Mostrar resultats
        fig, ax = plt.subplots(2, 1, figsize=(24, 12))
        ax[0].imshow(gray_cropped, cmap="gray")
        ax[0].set_title("Imatge retallada")
        ax[0].axis("off")

        ax[1].imshow(posidonia_mask, cmap="gray")
        ax[1].set_title("Àrea detectada de *Posidònia*")
        ax[1].axis("off")

        plt.tight_layout()
        plt.show()


    print(f" Error: La imatge no existeix a la ruta proporcionada: {image_path}")

