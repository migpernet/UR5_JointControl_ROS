#!/usr/bin/env python3
import cv2
import numpy as np
from ultralytics import YOLO

class YoloAttentionFilter:
    def __init__(self, model_name="yolov8n.pt", conf_threshold=0.20): # <--- Confiança reduzida por padrão
        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold

    def process(self, color_img, depth_img):
        height, width = color_img.shape[:2]
        
        # Define a altura da Zona Morta (ex: os últimos 150 pixels da base da tela)
        # Ajuste esse valor se a garra for mais alta ou mais baixa na sua câmera
        dead_zone_y = height - 150 

        results = self.model(color_img, conf=self.conf_threshold, verbose=False)
        filtered_depth = np.zeros_like(depth_img)
        
        best_box = None
        highest_conf = -1
        best_cls = -1

        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                # Extrai as coordenadas
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf)
                
                # REGRA 1: Filtro de Auto-Oclusão (Ignora a garra)
                # Se a base da bounding box invadir a zona morta, pule para o próximo objeto.
                if y2 > dead_zone_y:
                    continue
                
                # REGRA 2: Maior Confiança (Dos objetos válidos na mesa)
                if conf > highest_conf:
                    highest_conf = conf
                    best_box = (x1, y1, x2, y2)
                    best_cls = int(box.cls)

            # Se encontrou um objeto válido (fora da zona morta)
            if best_box is not None:
                x1, y1, x2, y2 = best_box
                
                # Aplica a máscara na profundidade
                filtered_depth[y1:y2, x1:x2] = depth_img[y1:y2, x1:x2]
                
                # Desenha o debug
                class_name = self.model.names[best_cls]
                label = f"{class_name} {highest_conf:.2f}"
                cv2.rectangle(color_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(color_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Desenha a linha da Zona Morta no Debug (Para calibração)
        cv2.line(color_img, (0, dead_zone_y), (width, dead_zone_y), (0, 0, 255), 2)
        cv2.putText(color_img, "ZONA MORTA (IGNORAR GARRA)", (10, dead_zone_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        return filtered_depth, color_img, best_box