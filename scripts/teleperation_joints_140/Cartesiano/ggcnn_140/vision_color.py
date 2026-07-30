#!/usr/bin/env python3
import cv2
import numpy as np

class ColorAttentionFilter:
    """
    Módulo "Drop-in" substituto para o YOLO no simulador.
    Isola a Bounding Box baseada em cores sólidas (Vermelho, Verde, Azul).
    Prioriza o objeto mais próximo ao CENTRO da imagem (foco da GGCNN).
    """
    def __init__(self):
        pass

    def process(self, color_img, depth_img):
        height, width = color_img.shape[:2]
        
        # Define a altura da Zona Morta (ex: os últimos 150 pixels da base da tela)
        dead_zone_y = height - 150 
        
        # Pega o centro da imagem inteira (Para o foco da GGCNN)
        img_center_x = width // 2
        img_center_y = height // 2
        
        # Converte para HSV para facilitar a detecção de cor
        hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)
        
        # Cria máscaras genéricas para as 3 cores (valores amplos para o Gazebo)
        # Vermelho (tem dois ranges no HSV)
        mask_red1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
        mask_red2 = cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
        mask_red = mask_red1 | mask_red2
        
        # Verde e Azul
        mask_green = cv2.inRange(hsv, np.array([40, 100, 100]), np.array([80, 255, 255]))
        mask_blue = cv2.inRange(hsv, np.array([100, 150, 0]), np.array([140, 255, 255]))
        
        # Junta todas as cores em uma só máscara
        combined_mask = mask_red | mask_green | mask_blue

        # Encontra os contornos das manchas coloridas
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        filtered_depth = np.zeros_like(depth_img)
        best_box = None
        min_distance = float('inf') # Começa com distância infinita

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Filtra ruídos minúsculos
            if area > 500:
                x, y, w, h = cv2.boundingRect(cnt)
                
                # REGRA 1: Filtro de Auto-Oclusão (Ignora a garra na zona morta)
                if (y + h) > dead_zone_y:
                    continue
                
                # REGRA 2: Pega o bloco mais próximo ao CENTRO da tela (Foco da GGCNN)
                box_center_x = x + (w // 2)
                box_center_y = y + (h // 2)
                
                # Teorema de Pitágoras (Distância ao quadrado) para achar o mais central
                distance = (box_center_x - img_center_x)**2 + (box_center_y - img_center_y)**2
                
                if distance < min_distance:
                    min_distance = distance
                    best_box = (x, y, x + w, y + h)

        # Se encontrou um objeto válido (fora da zona morta e mais central)
        if best_box is not None:
            x1, y1, x2, y2 = best_box
            
            # Aplica a máscara na profundidade: copia apenas a área da bounding box
            filtered_depth[y1:y2, x1:x2] = depth_img[y1:y2, x1:x2]
            
            # Desenha o debug para visualização
            cv2.rectangle(color_img, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(color_img, "Alvo Central", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        # Desenha a linha da Zona Morta no Debug
        cv2.line(color_img, (0, dead_zone_y), (width, dead_zone_y), (0, 0, 255), 2)
        cv2.putText(color_img, "ZONA MORTA (IGNORAR GARRA)", (10, dead_zone_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Desenha uma mira amarela no centro exato da imagem
        cv2.drawMarker(color_img, (img_center_x, img_center_y), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)

        return filtered_depth, color_img, best_box







# #!/usr/bin/env python3
# import cv2
# import numpy as np

# class ColorAttentionFilter:
#     """
#     Módulo "Drop-in" substituto para o YOLO no simulador.
#     Isola a Bounding Box baseada em cores sólidas (Vermelho, Verde, Azul).
#     """
#     def __init__(self):
#         pass

#     def process(self, color_img, depth_img):
#         height, width = color_img.shape[:2]
#         dead_zone_y = height - 150 
        
#         # Converte para HSV para facilitar a detecção de cor
#         hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)
        
#         # Cria máscaras genéricas para as 3 cores (valores amplos para o Gazebo)
#         mask_red1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
#         mask_red2 = cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
#         mask_red = mask_red1 | mask_red2
        
#         mask_green = cv2.inRange(hsv, np.array([40, 100, 100]), np.array([80, 255, 255]))
#         mask_blue = cv2.inRange(hsv, np.array([100, 150, 0]), np.array([140, 255, 255]))
        
#         # Junta todas as cores em uma só máscara
#         combined_mask = mask_red | mask_green | mask_blue

#         # Encontra os contornos das manchas coloridas
#         contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
#         filtered_depth = np.zeros_like(depth_img)
#         best_box = None
#         max_area = 0

#         for cnt in contours:
#             area = cv2.contourArea(cnt)
#             # Filtra ruídos minúsculos
#             if area > 500:
#                 x, y, w, h = cv2.boundingRect(cnt)
                
#                 # REGRA 1: Filtro de Auto-Oclusão (Ignora a garra)
#                 if (y + h) > dead_zone_y:
#                     continue
                
#                 # REGRA 2: Pega o maior bloco da mesa
#                 if area > max_area:
#                     max_area = area
#                     best_box = (x, y, x + w, y + h)

#         if best_box is not None:
#             x1, y1, x2, y2 = best_box
#             filtered_depth[y1:y2, x1:x2] = depth_img[y1:y2, x1:x2]
            
#             # Desenha o debug igual ao YOLO
#             cv2.rectangle(color_img, (x1, y1), (x2, y2), (255, 0, 255), 2)
#             cv2.putText(color_img, "Bloco Detectado", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

#         # Desenha a linha da Zona Morta
#         cv2.line(color_img, (0, dead_zone_y), (width, dead_zone_y), (0, 0, 255), 2)

#         return filtered_depth, color_img, best_box

