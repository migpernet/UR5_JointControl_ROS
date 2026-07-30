#!/usr/bin/env python3
import cv2
import numpy as np

class WorkspaceFilterOpenCV:
    """
    Classe modular para filtragem de nuvem de pontos/imagens de profundidade.
    Remove fundo (mesa) e ruídos usando operações morfológicas do OpenCV e NumPy.
    """
    def __init__(self, min_depth=100, max_depth=600, kernel_size=3):
        # Limites de profundidade (em milímetros)
        self.min_depth = min_depth
        self.max_depth = max_depth
        
        # Cria o "Kernel" (matriz 3x3) que será a "espátula" para a erosão/dilatação
        self.kernel = np.ones((kernel_size, kernel_size), np.uint8)

    def process_depth(self, depth_img):
        """
        Recebe a imagem de profundidade bruta e retorna ela filtrada.
        """
        # 1. FAXINA DE DADOS: Transforma pixels vazios (NaN) ou infinitos em zero.
        # Isso impede que o filtro quebre com dados sujos do Gazebo.
        clean_depth = np.nan_to_num(depth_img, nan=0.0, posinf=0.0, neginf=0.0)

        # 2. MÁSCARA ROBUSTA COM NUMPY
        # Verifica pixel por pixel: Está entre min_depth e max_depth? 
        # Se sim, vira 255 (Branco). Se não, 0 (Preto).
        mask = np.logical_and(clean_depth >= self.min_depth, clean_depth <= self.max_depth).astype(np.uint8) * 255

        # 3. FILTROS MORFOLÓGICOS (Remove os ruídos)
        mask_clean = cv2.erode(mask, self.kernel, iterations=1)
        mask_clean = cv2.dilate(mask_clean, self.kernel, iterations=1)

        # 4. APLICAÇÃO
        filtered_depth = clean_depth.copy()
        filtered_depth[mask_clean == 0] = 0.0

        return filtered_depth