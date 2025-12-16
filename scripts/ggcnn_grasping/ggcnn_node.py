#!/usr/bin/env python3

import rospy
import torch
import torch.nn.functional as F  
import numpy as np
import cv2
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32, Float32MultiArray # Adicionando Float32 para a largura

# Importar o modelo da GG-CNN (o script do repositório)
from models.ggcnn import GGCNN

# Configurações do projeto
MODEL_PATH = '/home/miguel/miguel_ws/src/ggcnn_pkg/scripts/ggcnn_grasping/models_trined/ggcnn_weights_cornell/ggcnn_epoch_23_cornell_statedict.pt'
CAMERA_TOPIC = '/camera/depth/image_raw'   
CAMERA_INFO_TOPIC = '/camera/depth/camera_info'        
IMAGE_SIZE = 300 

class GGCNNNode:
    def __init__(self):
        rospy.init_node('ggcnn_node', anonymous=True)
        self.bridge = CvBridge()
        
        # Carregar o modelo
        self.model = GGCNN().to(torch.device("cpu"))
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
        self.model.eval()

        self.camera_info_msg = rospy.wait_for_message(CAMERA_INFO_TOPIC, CameraInfo)
        
        # Publishers para a pose e a largura da preensão
        self.grasp_pose_pub = rospy.Publisher('/grasp/pose', PoseStamped, queue_size=1)
        self.grasp_width_pub = rospy.Publisher('/grasp/width', Float32, queue_size=1)

        rospy.loginfo("Nó GG-CNN inicializado e publicando no tópico /grasp.")
        rospy.Subscriber(CAMERA_TOPIC, Image, self.depth_callback)
        rospy.spin()

    def depth_callback(self, msg):
        try:
            rospy.loginfo("Recebendo nova imagem de profundidade...")
            # Processamento da imagem de profundidade para gerar um tensor de 4 dimensões com o formato (batch_size, channels, height, width).
            depth_image = self.bridge.imgmsg_to_cv2(msg, "16UC1")
            depth_image = depth_image.astype(np.float32)
            depth_image /= 1000.0  # Convertendo para metros
            depth_tensor = torch.from_numpy(depth_image).unsqueeze(0).unsqueeze(0)   # Adicionando dimensões de batch e canal
            
            depth_tensor_resized = F.interpolate(depth_tensor, size=(IMAGE_SIZE, IMAGE_SIZE), mode='bilinear', align_corners=False) # Redimensionando para a entrada da GG-CNN
            # depth_tensor_resized --> tensor de 4 dimensões (batch_size, channels, height, width)
            '''
            batch_size: processa uma única imagem de cada vez, então é 1;
            channels: 1 canal para a imagem de profundidade;
            height e width: 300x300 pixels após o redimensionamento.
            '''

            with torch.no_grad():
                pos_pred, angle_pred, width_pred, qual_pred = self.model(depth_tensor_resized)
            
            qual_array = qual_pred.squeeze().cpu().numpy()
            best_y, best_x = np.unravel_index(np.argmax(qual_array), qual_array.shape)
            rospy.loginfo(f"Melhor ponto de preensão encontrado: y={best_y}, x={best_x}")
            
            best_angle = angle_pred.squeeze()[best_y, best_x].item()
            best_width = width_pred.squeeze()[best_y, best_x].item()
            best_qual = qual_pred.squeeze()[best_y, best_x].item()
            rospy.loginfo(f"Melhor ângulo: {best_angle:.2f}, Melhor largura: {best_width:.4f}, Qualidade: {best_qual:.4f}")

            best_width = float(np.abs(best_width))
            best_width = min(best_width, 0.085)
            
            fx = self.camera_info_msg.K[0]
            fy = self.camera_info_msg.K[4]
            cx = self.camera_info_msg.K[2]
            cy = self.camera_info_msg.K[5]

            z = depth_image[best_y, best_x]
            x = (best_x * (msg.width / IMAGE_SIZE) - cx) * z / fx
            y = (best_y * (msg.height / IMAGE_SIZE) - cy) * z / fy
            
            rospy.loginfo(f"Qualidade da preensão: {best_qual:.4f}")
            rospy.loginfo(f"Parâmetros da preensão: x={x:.4f}, y={y:.4f}, z={z:.4f}, ângulo={best_angle:.2f}, largura={best_width:.4f}")
            
            # Publicar a pose
            grasp_pose = PoseStamped()
            grasp_pose.header.frame_id = "camera_link" # Publicamos no frame da câmera
            grasp_pose.header.stamp = rospy.Time.now()
            grasp_pose.pose.position.x = float(x)
            grasp_pose.pose.position.y = float(y)
            grasp_pose.pose.position.z = float(z)

            # Publicar a largura
            self.grasp_width_pub.publish(Float32(data=float(best_width)))
            self.grasp_pose_pub.publish(grasp_pose)
            
            # Aqui podemos adicionar uma lógica de shutdown ou de loop para continuar escutando
            # Por enquanto, mantemos o shutdown para testar uma única preensão
            rospy.signal_shutdown("Parâmetros da preensão publicados.")
            
        except Exception as e:
            rospy.logerr("Erro no callback da imagem: %s", str(e))

if __name__ == '__main__':
    try:
        input = input("Pressione Enter para iniciar o nó GG-CNN...")

        GGCNNNode()
    except rospy.ROSInterruptException:
        pass