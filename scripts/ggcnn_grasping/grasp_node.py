#!/usr/bin/env python3

import rospy
import torch
import torch.nn.functional as F  
import numpy as np
import cv2
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import moveit_commander
from geometry_msgs.msg import PoseStamped
# from std_msgs.msg import Float64MultiArray   --- IGNORE ---
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from tf.transformations import quaternion_from_euler

# Importar o modelo da GG-CNN (o script do repositório)
from models.ggcnn import GGCNN

# Configurações do projeto
MODEL_PATH = '/home/miguel/miguel_ws/src/ggcnn_pkg/scripts/ggcnn_grasping/models_trined/ggcnn_weights_cornell/ggcnn_epoch_23_cornell_statedict.pt' # Caminho relativo ao seu pacote
CAMERA_TOPIC = '/camera/depth/image_raw'   # Tópico da imagem de profundidade
CAMERA_INFO_TOPIC = '/camera/depth/camera_info' # Tópico das informações da câmera
GRIPPER_COMMAND_TOPIC = '/ur5/gripper_controller/command' # Tópico para comandos da garra
IMAGE_SIZE = 300 # Tamanho da imagem de entrada para a GG-CNN 224x224 é o tamanho padrão usado na maioria dos modelos GG-CNN

class GraspingNode:
    def __init__(self):
        rospy.init_node('ggcnn_grasping_node', anonymous=True)
        self.bridge = CvBridge()
        
        # Carregar o modelo
        # Passo 1: Instanciar a arquitetura da rede neural
        self.model = GGCNN().to(torch.device("cpu"))        

        # Passo 2: Carregar os pesos do state_dict no modelo
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
        self.model.eval()

        # Configurar MoveIt!
        moveit_commander.roscpp_initialize([])
        self.robot = moveit_commander.RobotCommander() 
        self.scene = moveit_commander.PlanningSceneInterface()
        self.arm_group = moveit_commander.MoveGroupCommander("manipulator")
        
        # Publisher para a garra
        self.gripper_pub = rospy.Publisher(GRIPPER_COMMAND_TOPIC, JointTrajectory, queue_size=1)
        
        rospy.loginfo("Nó de preensão inicializado e pronto!")

        # Aguardar as mensagens da câmera
        self.camera_info_msg = rospy.wait_for_message(CAMERA_INFO_TOPIC, CameraInfo)
        rospy.Subscriber(CAMERA_TOPIC, Image, self.depth_callback)
        rospy.spin()

    def depth_callback(self, msg):
        try:
            rospy.loginfo("Recebendo nova imagem de profundidade...")

            # Converte a mensagem de imagem do ROS para uma imagem NumPy
            depth_image = self.bridge.imgmsg_to_cv2(msg, "16UC1")  # Profundidade em milímetros "16UC1"
            
            # Pré-processamento e normalização da imagem
            depth_image = depth_image.astype(np.float32)
            depth_image /= 1000.0 # Convertendo para metros
            depth_tensor = torch.from_numpy(depth_image).unsqueeze(0).unsqueeze(0)

            # Redimensionar para a entrada da GG-CNN
            depth_tensor_resized = F.interpolate(depth_tensor, size=(IMAGE_SIZE, IMAGE_SIZE), mode='bilinear', align_corners=False)
            
            with torch.no_grad():
                rospy.loginfo(f'depth_tensor_resized: {depth_tensor_resized.shape}')
                pos_pred, angle_pred, width_pred, qual_pred= self.model(depth_tensor_resized) 
                
                # Esta é a correção. A saída da rede é um único tensor que precisa ser dividido.
                # output = self.model(depth_tensor_resized)
                # pos_pred, angle_pred, width_pred = torch.split(output, 1, dim=1)
                               
            
            # Encontrar o melhor ponto de preensão
            qual_array = qual_pred.squeeze().cpu().numpy()  # Baseado na maior qualidade de preensão
            best_y, best_x = np.unravel_index(np.argmax(qual_array), qual_array.shape)  # Baseado na maior qualidade de preensão      
            
            # Obtém os valores de ângulo, largura e qualidade para a melhor preensão
            best_angle = angle_pred.squeeze()[best_y, best_x].item()
            best_width = width_pred.squeeze()[best_y, best_x].item()
            best_qual = qual_pred.squeeze()[best_y, best_x].item() # Obtém o valor de qualidade
            
            best_width = np.abs(best_width) 
            best_width = min(best_width, 0.085) # O máximo da Robotiq 2F-85 é 85mm

            rospy.loginfo(f"Pontuação de qualidade da melhor preensão: {best_qual:.4f}")

            # Calcular a pose 3D no frame da câmera
            fx = self.camera_info_msg.K[0]
            fy = self.camera_info_msg.K[4]
            cx = self.camera_info_msg.K[2]
            cy = self.camera_info_msg.K[5]

            z = depth_image[best_y, best_x]
            x = (best_x * (msg.width / IMAGE_SIZE) - cx) * z / fx
            y = (best_y * (msg.height / IMAGE_SIZE) - cy) * z / fy
            
            self.execute_grasp(float(x), float(y), float(z), float(best_angle), float(best_width))
            rospy.signal_shutdown("Preensão concluída.")
            
        except Exception as e:
            rospy.logerr("Erro no callback da imagem: %s", str(e))
    
    def execute_grasp(self, x, y, z, angle, width):
        rospy.loginfo(f"Grasp point encontrado: (x={x:.4f}, y={y:.4f}, z={z:.4f}) com ângulo {angle:.2f} rad e largura {width:.4f} m")

        # Conversão de coordenadas da câmera para o frame do robô
        # TODO: Implementar a transformação TF. Por enquanto, assumimos um frame simples.
        # Em um cenário real, você precisa transformar o ponto (x, y, z) do frame da câmera
        # para o frame da base do robô usando um listener TF.

        # Criar a pose da garra
        grasp_pose = PoseStamped()
        grasp_pose.header.frame_id = "base_link" # Mude para o frame da base do seu robô
        grasp_pose.pose.position.x = x
        grasp_pose.pose.position.y = y
        grasp_pose.pose.position.z = z-z+0.19
        
        # Calcular a orientação da garra (rotação em torno de Z para o ângulo)
        quaternion = quaternion_from_euler(0, np.pi, angle) # Exemplo de rotação
        grasp_pose.pose.orientation.x = quaternion[0]
        grasp_pose.pose.orientation.y = quaternion[1]
        grasp_pose.pose.orientation.z = quaternion[2]
        grasp_pose.pose.orientation.w = quaternion[3]
        
        # Planejar e executar a trajetória
        self.arm_group.set_pose_target(grasp_pose.pose)
        
        rospy.loginfo("Planejando a preensão...")
        plan = self.arm_group.plan()

        if plan[0]:
            rospy.loginfo("Plano de preensão encontrado. Executando...")
            self.arm_group.execute(plan[1], wait=True)
            rospy.sleep(1.0) # Espera o robô se mover
            
            # Fechar a garra
            gripper_msg = JointTrajectory()
            gripper_msg.joint_names = ['robotiq_85_left_knuckle_joint'] # Verifique os nomes corretos no seu URDF

            point = JointTrajectoryPoint()
            point.positions = [width/2.0, width/2.0]
            point.time_from_start = rospy.Duration(1.0)

            gripper_msg.points.append(point)
            self.gripper_pub.publish(gripper_msg)
            
            rospy.sleep(1.0)
            
            # Levantar o robô
            grasp_pose.pose.position.z += 0.1
            self.arm_group.set_pose_target(grasp_pose.pose)
            self.arm_group.go(wait=True)
        else:
            rospy.logwarn("Não foi possível encontrar um plano de preensão.")
        
        self.arm_group.stop()
        self.arm_group.clear_pose_targets()
        
if __name__ == '__main__':
    try:
        GraspingNode()
    except rospy.ROSInterruptException:
        pass