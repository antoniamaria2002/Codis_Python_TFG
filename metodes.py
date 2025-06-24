# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 08:02:29 2025

@author: Moi
"""
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage import filters, exposure, feature
from scipy import ndimage as ndi
from skimage import morphology

# Ruta completa de la imatge
image_path = r"C:\Users\Moi\Desktop\tfg\videoimatgesvilanova\posidoniavilanova3juliol20234bandas\codis\Blue-Nir_resta2023\Blue_Corrected2_cropped.png"

# Directori de sortida
directory = r"C:\Users\Moi\Desktop\tfg\videoimatgesvilanova\posidoniavilanova3juliol20234bandas\codis\edgemetodoscorregit"

# Carregar la imatge
image = cv2.imread(image_path)

if image is None:
    print("Error: No s'ha pogut carregar la imatge. Verifica la ruta.")
else:
    print("Imatge carregada correctament.")

    # Convertir a escala de grisos si és necessari
    if len(image.shape) == 3:  
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image  # Ja està en escala de grisos

    # --- Augmentar el contrast (CLAHE + correcció Gamma) ---
    # Aplicar CLAHE (Equalització adaptativa de l'histograma)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)

    # Aplicar correcció Gamma
    gamma = 5  # Ajustar brillantor (ajustable)
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in np.arange(0, 256)]).astype("uint8")
    gray_contrast = cv2.LUT(gray_clahe, table)

    # --- Retallar la imatge a la primera meitat ---
    height, width = gray.shape[:2] 
    gray_cropped = gray_contrast[(height // 2) - 190:height - 300, :]

    # Aplicar detecció de vores sobre la imatge retallada 
    
    # Prewitt
    sigma_prewitt = 3
    smoothed_gray = filters.gaussian(gray_cropped, sigma=sigma_prewitt)
    prewitt_edges = filters.prewitt(smoothed_gray)
    prewitt_edges_rescaled = exposure.rescale_intensity(prewitt_edges, in_range=(0, 1), out_range=(0, 255))
    prewitt_edges_8bit = np.uint8(255 * (prewitt_edges - prewitt_edges.min()) / (prewitt_edges.max() - prewitt_edges.min()))

    # Kirsch
    sigma_kirsch = 3
    smoothed_gray_kirsch = filters.gaussian(gray_cropped, sigma=sigma_kirsch)
    kirsch_kernels = [
        np.array([[-3, -3, -3], [5, 0, 5], [5, 5, 5]]),
        np.array([[5, -3, -3], [5, 0, -3], [5, 5, -3]]),
        np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]]),
        np.array([[5, 5, 5], [5, 0, -3], [-3, -3, -3]]),
        np.array([[5, 5, 5], [5, 0, 5], [-3, -3, -3]]),
        np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]]),
        np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]),
        np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]])
    ]
    kirsch_responses = [ndi.convolve(smoothed_gray_kirsch, k) for k in kirsch_kernels]
    kirsch_edges = np.max(kirsch_responses, axis=0)
    kirsch_edges = exposure.rescale_intensity(kirsch_edges, in_range=(0, np.max(kirsch_edges)), out_range=(0, 255))
    kirsch_edges_8bit = np.uint8(255 * (kirsch_edges - kirsch_edges.min()) / (kirsch_edges.max() - kirsch_edges.min()))
    
    # Canny
    canny_edges = feature.canny(gray_cropped, sigma=0.57)
    canny_edges_float = canny_edges.astype(np.float32)
    canny_edges_8bit = np.uint8(255 * (canny_edges_float - canny_edges_float.min()) / (canny_edges_float.max() - canny_edges_float.min()))

    # LoG (Laplace de Gaussiana)
    
    def LoG_filter(image, sigma, size=None):
        # Generar el kernel LoG
        if size is None:
            size = int(6 * sigma + 1) if sigma >= 0.5 else 7

        if size % 3 == 0:
            size += 1

        x, y = np.meshgrid(np.arange(-size//2+1, size//2+1), np.arange(-size//2+1, size//2+1))
        kernel = -(1/(np.pi * sigma**4)) * (1 - ((x**2 + y**2) / (2 * sigma**2))) * np.exp(-(x**2 + y**2) / (2 * sigma**2))
        kernel = kernel / np.sum(np.abs(kernel))

        # Realizar la convolución
        result = ndi.convolve(image, kernel)

        return result
    
    # Aplicar el filtre LoG
    log_sigma = 4.0
    log_edges = LoG_filter(gray_cropped, log_sigma)
    log_edges_rescaled = exposure.rescale_intensity(log_edges, in_range='image', out_range=(0, 255))
    log_edges_8bit = np.uint8(255 * (log_edges_rescaled - log_edges_rescaled.min()) / (log_edges_rescaled.max() - log_edges_rescaled.min()))

    # Vores morfològiques
    selem = morphology.disk(3)
    morph_edges = morphology.dilation(gray_cropped, selem) - gray_cropped
    morph_edges = exposure.rescale_intensity(morph_edges, in_range='image', out_range=(0, 255))
    morph_edges_8bit = np.uint8(255 * (morph_edges - morph_edges.min()) / (morph_edges.max() - morph_edges.min()))

    # Subpíxel
    sigma_subpixel = 8  
    smoothed_gray_subpixel = filters.gaussian(gray_cropped, sigma=sigma_subpixel)
    subpixel_edges = cv2.Laplacian(smoothed_gray_subpixel, cv2.CV_64F, ksize=7)
    subpixel_edges = exposure.rescale_intensity(subpixel_edges, in_range='image', out_range=(0, 255))
    subpixel_edges_8bit = np.uint8(255 * (subpixel_edges - subpixel_edges.min()) / (subpixel_edges.max() - subpixel_edges.min()))

    # Guardar totes les imatges de vores per separat
    output_prewitt_path = os.path.join(directory, "prewitt_edges.png")
    cv2.imwrite(output_prewitt_path, prewitt_edges_8bit)

    output_kirsch_path = os.path.join(directory, "kirsch_edges.png")
    cv2.imwrite(output_kirsch_path, kirsch_edges_8bit)

    output_canny_path = os.path.join(directory, "canny_edges.png")
    cv2.imwrite(output_canny_path, canny_edges_8bit)

    output_log_path = os.path.join(directory, "loG_edges.png")
    cv2.imwrite(output_log_path, log_edges_8bit)

    output_morph_path = os.path.join(directory, "morph_edges.png")
    cv2.imwrite(output_morph_path, morph_edges_8bit)

    output_subpixel_path = os.path.join(directory, "subpixel_edges.png")
    cv2.imwrite(output_subpixel_path, subpixel_edges_8bit)

    # Mostrar totes les vores detectades 
    fig, axes = plt.subplots(8, 1, figsize=(32, 16))
    titles = [
        "(a) Original recortado", "(b) Imagen suavizada", "(c) Mapa Prewitt", "(d) Mapa Kirsch",
        "(e) Mapa Canny", "(f) Mapa LoG", "(g) Mapa Morfológico", "(h) Mapa Subpíxel"
    ]
    images = [gray_cropped, smoothed_gray, prewitt_edges, kirsch_edges,
              canny_edges, log_edges, morph_edges, subpixel_edges]

    for ax, img, title in zip(axes.ravel(), images, titles):
        ax.imshow(img, cmap='gray')
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    
    # Guardar la figures amb totes les vores detectades
    output_figure_path = os.path.join(directory, "output_all_cropped.png")
    plt.savefig(output_figure_path, dpi=600)
    plt.show()
    
    # Pas 1: Umbralitzar les vores per crear una màscara binària
    _, binary_mask = cv2.threshold(prewitt_edges_8bit, 28, 255, cv2.THRESH_BINARY)
    
    # Pas 2: Operacions morfològiques per refinar les vores
    kernel = np.ones((3, 3), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=2)  
    binary_mask = cv2.dilate(binary_mask, kernel, iterations=1)  

    # Pas 3: Trobar contorns
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Pas 4: Dibuixar el contorn més gran sobre la imatge original
    contour_image = gray_cropped.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)  
    
    output_contour_image_path = os.path.join(directory, "Gray_cropped_amb_contorns.png")
    cv2.imwrite(output_contour_image_path, contour_image)

    # Mostrar els resultats
    cv2.imshow("Contorns detectats", contour_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f" Resultats guardats a: {directory}")
