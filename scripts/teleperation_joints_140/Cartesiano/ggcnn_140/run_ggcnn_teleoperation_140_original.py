#! /usr/bin/env python3
# Backup de script. Este está funcionando. Guardei este para poder tentar fazer os gráficos de preensão com boundbox.
# Agora, estou utilizando uma cópia para fazer as adaptações e melhorias como no cálculo do ângulo.

# Python
import time
import numpy as np
import argparse
from skimage.draw import circle_perimeter

# CNN
import torch
import torch.nn as nn
import torch.nn.functional as F
# Alterado para usar tf2
import tf2_ros
import tf2_geometry_msgs

# Image
import cv2

# ROS
import rospy
import rospkg
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32MultiArray, Float32
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from geometry_msgs.msg import TransformStamped, PoseStamped # Adicionado para tf2
import math

# Importe a classe GGCNN que você criou
from models.ggcnn import GGCNN 


class TimeIt:
    def __init__(self, s):
        self.s = s
        self.t0 = None
        self.t1 = None
        self.print_output = False

    def __enter__(self):
        self.t0 = time.time()

    def __exit__(self, t, value, traceback):
        self.t1 = time.time()
        print('%s: %s' % (self.s, self.t1 - self.t0))

def parse_args():
    parser = argparse.ArgumentParser(description='GGCNN grasping')
    parser.add_argument('--real', action='store_true', help='Consider the real intel realsense')
    parser.add_argument('--plot', action='store_true', help='Plot depth image')
    args = parser.parse_args()
    return args

class ggcnn_grasping(object):
    def __init__(self, args):
        rospy.init_node('ggcnn_detection')

        self.args = args
        self.bridge = CvBridge()
        self.latest_depth_message = None
        self.color_img = None
        
        # Carregar a rede PyTorch
        rospack = rospkg.RosPack()
        Home = rospack.get_path('ggcnn_pkg')
        MODEL_FILE = Home + '/scripts/ggcnn_grasping/models_trined/ggcnn_weights_cornell/ggcnn_epoch_23_cornell_statedict.pt'
        self.model = GGCNN()
        self.model.load_state_dict(torch.load(MODEL_FILE, map_location=torch.device('cpu')))
        self.model.eval()


        # Configurar TF2
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        # Load GGCN parameters
        self.crop_size = rospy.get_param("/GGCNN/crop_size", 300)
        self.FOV = rospy.get_param("/GGCNN/FOV", 60)
        self.camera_topic_info = rospy.get_param("/GGCNN/camera_topic_info", "/camera/depth/camera_info")
        if self.args.real:
            self.camera_topic = rospy.get_param("/GGCNN/camera_topic_realsense", "/camera/depth/image_raw")
        else:
            self.camera_topic = rospy.get_param("/GGCNN/camera_topic", "/camera/depth/image_raw")

        # Output publishers.
        self.grasp_pub = rospy.Publisher('ggcnn/img/grasp', Image, queue_size=1)
        self.depth_pub = rospy.Publisher('ggcnn/img/depth', Image, queue_size=1)
        self.width_pub = rospy.Publisher('ggcnn/img/width', Image, queue_size=1)
        self.depth_pub_shot = rospy.Publisher('ggcnn/img/depth_shot', Image, queue_size=1)
        self.ang_pub = rospy.Publisher('ggcnn/img/ang', Image, queue_size=1)
        self.cmd_pub = rospy.Publisher('ggcnn/out/command', Float32MultiArray, queue_size=1)
        self.cmd_pub_grasp = rospy.Publisher('ggcnn/out/command_grasp', Float32MultiArray, queue_size=1) # Adicionado para publicar a largura
        self.unity_pose_pub = rospy.Publisher('/ggcnn/unity_target_pose', PoseStamped, queue_size=1)  # Publisher pose para o Unity
        self.unity_width_pub = rospy.Publisher('/ggcnn/unity_gripper_width', Float32, queue_size=1)  # Publisher largura para o Unity

        # Initialize some var
        self.grasping_point = []
        self.depth_image_shot = None
        
        # Get the camera parameters
        camera_info_msg = rospy.wait_for_message(self.camera_topic_info, CameraInfo)
        K = camera_info_msg.K
        self.fx = K[0]
        self.cx = K[2]
        self.fy = K[4]
        self.cy = K[5]

        # Subscribers
        rospy.Subscriber(self.camera_topic, Image, self.get_depth_callback, queue_size=10)
        rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback, queue_size=10)


    def get_depth_callback(self, depth_message):
        self.latest_depth_message = depth_message
        

    def image_callback(self, color_msg):
        self.color_img = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")


    #************************

    def get_depth_image_shot(self):
        self.depth_image_shot = rospy.wait_for_message("camera/depth/image_raw", Image)
        self.depth_image_shot.header = self.depth_message.header
        self.depth_pub_shot.publish(self.depth_image_shot)   # publicar imagem de profundidade original


    def draw_grasp(self, img, x, y, theta, width, height=20, color=(0,255,0), thickness=2):
        """
        Desenha um retângulo de preensão na imagem.

        Parâmetros:
            img     : imagem onde desenhar
            x, y    : coordenadas do centro da preensão
            theta   : ângulo da garra (em radianos)
            width   : largura (abertura da garra, em pixels)
            height  : altura fixa do retângulo (espessura da garra, default=20)
            color   : cor do retângulo
            thickness: espessura da linha
        """
        # Metade da largura e altura
        dx = width / 2
        dy = height / 2

        # Coordenadas do retângulo antes da rotação
        rect = np.array([
            [-dx, -dy],
            [ dx, -dy],
            [ dx,  dy],
            [-dx,  dy]
        ])

        # Matriz de rotação
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])

        # Aplica rotação
        rect = rect @ R.T

        # Translada para (x, y)
        rect[:, 0] += x
        rect[:, 1] += y

        # Converte para inteiros
        rect = rect.astype(np.int32)

        # Desenha o polígono
        cv2.polylines(img, [rect], isClosed=True, color=color, thickness=thickness)

        return img


    #************************

    def _normalize_and_colorize_map(self, map_array, min_val=0, max_val=1):
        """ Normaliza um array para 0-255 e aplica um mapa de cores para visualização. """
        if np.max(map_array) > np.min(map_array):
            normalized_map = (map_array - np.min(map_array)) / (np.max(map_array) - np.min(map_array))
        else:
            normalized_map = np.zeros_like(map_array, dtype=np.float32)

        # Escala para [0, 255], clipe valores e converte para uint8
        normalized_map = np.clip(normalized_map * 255, 0, 255).astype(np.uint8)

        # Garante que é exatamente 2D (H, W)
        normalized_map = np.ascontiguousarray(normalized_map)

        # Aplica o colormap
        colorized_map = cv2.applyColorMap(normalized_map, cv2.COLORMAP_JET)
        return colorized_map



    def _generate_visual_maps(self, pos_out, ang_out, width_out, qual_out):
        """ Gera os mapas de preensão visualizados a partir dos arrays de previsão. """
        # Normaliza e colore os mapas para visualização
        pos_img = self._normalize_and_colorize_map(pos_out)
        ang_img = self._normalize_and_colorize_map(ang_out, min_val=-np.pi/2, max_val=np.pi/2)
        width_img = self._normalize_and_colorize_map(width_out)
        qual_img = self._normalize_and_colorize_map(qual_out)
        
        # chama a função para inserir os retângulos de preensão
        qual_img = self.draw_grasp(qual_img, self.best_x, self.best_y, -1*self.ang, self.width_px, height=15, color=(255,255,255), thickness=1)

        # Desenha o ponto de preensão no mapa de qualidade
        #  cv2.circle(qual_img, (self.best_x, self.best_y), 5, (255, 255, 0), -1) # pode ser utilizada esta opção ou "circle_perimeter" 

        rr, cc = circle_perimeter(self.best_y, self.best_x, 5)
        qual_img[rr, cc] = 255

        return pos_img, ang_img, width_img, qual_img




    # Esta função calcula a pose de preensão em 3D, no frame da câmera. Em seguida, ela chama a função get_grasp_params_in_base_link.
    def depth_process_ggcnn(self):
            depth_message = self.latest_depth_message
            if depth_message is None or self.color_img is None:
                return

            # INPUT
            depth = self.bridge.imgmsg_to_cv2(depth_message, "16UC1")
            depth = depth.astype(np.float32)  
            # depth /= 1000.0 
            depth_copy_for_point_depth = depth.copy()
            
            height_res, width_res = depth.shape
            # # It crops a 300x300 resolution square at the top of the depth image - depth[0:300, 170:470]
            depth_crop = depth[0 : self.crop_size, 
                            (width_res - self.crop_size)//2 : (width_res - self.crop_size)//2 + self.crop_size]
            # Creates a deep copy of the depth_crop image
            depth_crop = depth_crop.copy()
            # Returns the positions represented by nan values
            depth_nan = np.isnan(depth_crop)
            depth_nan = depth_nan.copy()
            # Substitute nan values by zero
            depth_crop[depth_nan] = 0

            # INPAINT PROCESS - Usando OpenCV
            depth_crop = cv2.copyMakeBorder(depth_crop, 1, 1, 1, 1, cv2.BORDER_DEFAULT) # Adiciona bordas para o inpaint 
            # se o numero que esta no vetor acima for 0, retorna o numero 1 na mesma posicao (como se fosse True)
            # se depth_crop == 0, retorna 1 como inteiro.
            # Ou seja, copia os pixels pretos da imagem e a posicao deles
            mask = (depth_crop == 0).astype(np.uint8)
            # Scale to keep as float, but has to be in bounds -1:1 to keep opencv happy.
            depth_scale = np.abs(depth_crop).max()
            # Normalize
            depth_crop = depth_crop.astype(np.float32) / depth_scale # Has to be float32, 64 not supported
            # Substitute mask values by near values. See opencv doc for more detail
            depth_crop = cv2.inpaint(depth_crop, mask, 1, cv2.INPAINT_NS) 
            # Back to original size and value range.
            depth_crop = depth_crop[1:-1, 1:-1]
            # reescale image
            depth_crop = depth_crop * depth_scale

            # INFERENCE PROCESS
            depth_crop = depth_crop/1000.0
            # values smaller than -1 become -1, and values larger than 1 become 1.
            depth_crop = np.clip((depth_crop - depth_crop.mean()), -1, 1)
            # converte um array NumPy chamado depth_crop em um tensor PyTorch e adiciona duas dimensões extra no início
            depth_tensor = torch.from_numpy(depth_crop).unsqueeze(0).unsqueeze(0) # 1x1x300x300 (batch_size, channels, height, width).
            
            self.model.eval() # Define o modelo para o modo de avaliação. Isso desativa camadas como Dropout e BatchNorm
            with torch.no_grad(): # Desliga o cálculo do gradiente para economizar memória e acelerar a inferência
                pred_out = self.model(depth_tensor)  # obter previsões do modelo.
                # pos_out, ang_out, width_out, qual_out = pred_out
            
            # Processamento e filtragem dos outputs
            points_out = pred_out[0].squeeze().cpu().numpy()
            cos_out = pred_out[1].squeeze().cpu().numpy()
            sin_out = pred_out[2].squeeze().cpu().numpy()
            ang_out = np.arctan2(sin_out, cos_out) / 2.0  # Cálculo do ângulo em radianos
            width_out = pred_out[3].squeeze().cpu().numpy() * 150 
            
           

            # Filter the outputs.
            pos_out_filtered = cv2.GaussianBlur(points_out, (5, 5), 0) # Possui o pixel de maior qualidade
            pos_out_filtered = np.clip(pos_out_filtered, 0.0, 1.0 - 1e-3)
            ang_out_filtered = cv2.GaussianBlur(ang_out, (5, 5), 0)
            width_out_filtered = cv2.GaussianBlur(width_out, (5, 5), 0)
               
                
            # CONTROL PROCESS
            try:
                transform_stamped = self.tf_buffer.lookup_transform("base_link", "robotiq_arg2f_base_link", rospy.Time(0), rospy.Duration(1.0))
                ROBOT_Z = transform_stamped.transform.translation.z
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                ROBOT_Z = 0.0
            
            # realiza a transformação do frame "robotiq_arg2f_base_link" para o frame "base_link"
            transform_stamped = self.tf_buffer.lookup_transform("base_link", "robotiq_arg2f_base_link", rospy.Time(0), rospy.Duration(1.0))
            ROBOT_Z = transform_stamped.transform.translation.z

            # Track the global max.
            # max_pixel correponds to the position of the max value in points_out_filtered
            # Encontra o pixel de preensão com a maior pontuação de qualidade
            max_pixel = np.array(np.unravel_index(np.argmax(pos_out_filtered), pos_out_filtered.shape)) # Obtem as coordenadas x,y do pixel de maior qualidade
            grasp_quality = pos_out_filtered[max_pixel[0], max_pixel[1]]   # Extrair a qualidade (score) do agarre
            # Return max_pixel position as an int (300x300)
            max_pixel = max_pixel.astype(int)
            self.best_y, self.best_x =  max_pixel
            ang = ang_out_filtered[max_pixel[0], max_pixel[1]]   # Extract the values for the best grasp
            width_px = abs(width_out_filtered[max_pixel[0], max_pixel[1]])  # in pixels
            reescaled_height = int(max_pixel[0]) 
            reescaled_width = int((width_res - self.crop_size) // 2 + max_pixel[1])
            max_pixel_reescaled = [reescaled_height, reescaled_width]
            point_depth = depth_copy_for_point_depth[max_pixel_reescaled[0], max_pixel_reescaled[1]] 


            # GRASP WIDTH PROCESS
            g_width = 2.0 * (ROBOT_Z + 0.24) * np.tan(self.FOV / height_res * width_px / 2.0 / 180.0 * np.pi) #* 0.37
            crop_size_width = float(self.crop_size)
            width_m = width_out_filtered / crop_size_width * 2.0 * point_depth * np.tan(self.FOV * crop_size_width / height_res / 2.0 / 180.0 * np.pi) / 1000 #* 0.37
            width_m = abs(width_m[max_pixel[0], max_pixel[1]])

            '''
            Este bloco de código é usado para converter as coordenadas de um pixel de uma imagem de 
            profundidade em coordenadas 3D no sistema de coordenadas da câmera. O processo é chamado 
            de projeção inversa ou desprojeção.
            max_pixel_reescaled = (u, v) em pixel
            ''' 
            if not np.isnan(point_depth):
                # These magic numbers are my camera intrinsic parameters.
                x = (max_pixel_reescaled[1] - self.cx)/(self.fx) * point_depth 
                y = (max_pixel_reescaled[0] - self.cy)/(self.fy) * point_depth
                grasping_point = [x, y, point_depth] # in meters


            # OUTPUT
            self.ang_out = ang_out
            self.width_out = width_out
            self.points_out = points_out
            self.depth_message_ggcnn = depth_message
            self.depth_crop = depth_crop
            self.ang = ang
            self.width_px = width_px
            self.max_pixel = max_pixel
            self.max_pixel_reescaled = max_pixel_reescaled
            self.g_width = g_width
            self.width_m = width_m
            self.point_depth = point_depth
            self.grasping_point = grasping_point
            self.qual_out = grasp_quality   # valor da qualidade
            self.pos_out_filtered = pos_out_filtered  # mapa de qualidade
            # rospy.loginfo(f"grasping_point (x, y, z) em relação à camera: ({grasping_point[0]:.4f}, {grasping_point[1]:.4f}, {grasping_point[2]:.4f})")


    def publish_images(self):
        if self.points_out is not None:
            # GERE os mapas visuais
            pos_img, ang_img, width_img, qual_img = self._generate_visual_maps(
                self.points_out, self.ang_out, self.width_out, self.pos_out_filtered
            )
            
            # PUBLIQUE os mapas visualizados
            pos_msg = self.bridge.cv2_to_imgmsg(pos_img, 'bgr8')
            pos_msg.header = self.depth_message_ggcnn.header
            self.grasp_pub.publish(pos_msg)

            ang_msg = self.bridge.cv2_to_imgmsg(ang_img, 'bgr8')
            ang_msg.header = self.depth_message_ggcnn.header
            self.ang_pub.publish(ang_msg)

            width_msg = self.bridge.cv2_to_imgmsg(width_img, 'bgr8')
            width_msg.header = self.depth_message_ggcnn.header
            self.width_pub.publish(width_msg)
            
            qual_msg = self.bridge.cv2_to_imgmsg(qual_img, 'bgr8')
            qual_msg.header = self.depth_message_ggcnn.header
            self.depth_pub.publish(qual_msg)




    # publica a pose do objeto (object_detected) para visualização no RViz em relação à câmera.
    def publish_data_to_robot(self):
        if not self.grasping_point:
            return

        # Output the best grasp pose relative to camera.
        cmd_msg = Float32MultiArray()
        # ["x", "y", "z", "angulo", "abertura_min", "abertura_max"]
        cmd_msg.data = [self.grasping_point[0]/1000.0, self.grasping_point[1]/1000.0, self.grasping_point[2]/1000.0, -1*self.ang, self.width_m, self.g_width]
        self.cmd_pub.publish(cmd_msg)
        
        # Publica a transformação do frame "object_detected" em relação ao frame "camera_depth_optical_frame"
        grasp_transform = TransformStamped()
        grasp_transform.header.stamp = rospy.Time.now()
        grasp_transform.header.frame_id = "camera_depth_optical_frame"
        grasp_transform.child_frame_id = "object_detected"
        grasp_transform.transform.translation.x = cmd_msg.data[0]
        grasp_transform.transform.translation.y = cmd_msg.data[1]
        grasp_transform.transform.translation.z = cmd_msg.data[2]
        q = quaternion_from_euler(3.14, 0, -1*cmd_msg.data[3])
        grasp_transform.transform.rotation.x = q[0]
        grasp_transform.transform.rotation.y = q[1]
        grasp_transform.transform.rotation.z = q[2]
        grasp_transform.transform.rotation.w = q[3]

        self.tf_broadcaster.sendTransform(grasp_transform)

        
        # rospy.loginfo("--- Frame 'object_detected' em relação ao frame 'camera_depth_optical_frame' ---")
        # rospy.loginfo(f"Posição (x, y, z): ({cmd_msg.data[0]:.4f}, {cmd_msg.data[1]:.4f}, {cmd_msg.data[2]:.4f})")
        # rospy.loginfo(f"Ângulo: {cmd_msg.data[3]:.4f} radianos")
        rospy.loginfo(f"Largura: {cmd_msg.data[4]:.4f} metros")
        rospy.loginfo(f"Qualidade da Preensão: {self.qual_out:.4f}")
        # rospy.loginfo(f"Pixel de preensão (x, y): ({self.best_x}, {self.best_y})")
        # rospy.loginfo("--------------------------------------------------")



    def get_transform_between_frames(self, target_frame, source_frame):
        """
        Busca e retorna a transformação entre dois frames.
        """
        try:
            # Tenta obter a transformação
            transform = self.tf_buffer.lookup_transform(target_frame, source_frame, rospy.Time(0), rospy.Duration(1.0))
            
            # Acessa os componentes da translação
            x = transform.transform.translation.x
            y = transform.transform.translation.y
            z = transform.transform.translation.z   
            
            # Orientação que será utilizada para preensão planar e antipodal obtida no ggcnn
            roll = 3.140
            pitch = 0.0
            yaw = -1 * self.ang  # Ângulo obtido do ggcnn   
            quat = quaternion_from_euler(roll, pitch, yaw)
            euler = euler_from_quaternion(quat)
            
            rospy.loginfo(f"Transformação de {source_frame} para {target_frame}: [x: {x:.4f}, y: {y:.4f}, z: {z:.4f}]")

            # =========================================================
            # --- NOVO BLOCO PARA O UNITY (BAIXA LATÊNCIA) ---
            unity_pose = PoseStamped()
            unity_pose.header.stamp = rospy.Time.now()
            unity_pose.header.frame_id = target_frame 
            
            unity_pose.pose.position.x = x
            unity_pose.pose.position.y = y
            unity_pose.pose.position.z = z
            
            unity_pose.pose.orientation.x = quat[0]
            unity_pose.pose.orientation.y = quat[1]
            unity_pose.pose.orientation.z = quat[2]
            unity_pose.pose.orientation.w = quat[3]

            self.unity_pose_pub.publish(unity_pose)
            rospy.loginfo(">> SUCESSO: Pose publicada no tópico /ggcnn/unity_target_pose!")

            self.unity_width_pub.publish(self.width_m)
            rospy.loginfo(">> SUCESSO: Largura publicada no tópico /ggcnn/unity_gripper_width!")
            # =========================================================

            # Continua o envio para os tópicos Float32 do robô
            cmd_msg_grasp = Float32MultiArray()
            cmd_msg_grasp.data = [x, y, z, self.ang, self.width_m, self.g_width]
            self.cmd_pub_grasp.publish(cmd_msg_grasp)

            # Chama a função para publicar a transformação estática 
            self.publish_static_transform(x, y, z, roll, pitch, yaw, 'base_link', 'object_grasp')

            return transform

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logerr(f"Erro ao buscar transformação: {e}")
            return None



    # def get_transform_between_frames(self, target_frame, source_frame):
    #     """
    #     Busca e retorna a transformação entre dois frames.
    #     Equivalente a 'rosrun tf tf_echo '.
    #     """
    #     try:
    #         # Tenta obter a transformação
    #         transform = self.tf_buffer.lookup_transform(target_frame, source_frame, rospy.Time(0), rospy.Duration(1.0))
            
    #         rospy.loginfo(f"Transformação de {source_frame} para {target_frame}:")
            
    #         # Acessa os componentes da translação
    #         x = transform.transform.translation.x
    #         y = transform.transform.translation.y
    #         z = transform.transform.translation.z   
    #         # Acessa os componentes da rotação (quaternion)
    #         qx = transform.transform.rotation.x
    #         qy = transform.transform.rotation.y
    #         qz = transform.transform.rotation.z
    #         qw = transform.transform.rotation.w 


            
    #         # *************************************************
    #         # # Orientação geral (sem considerar o ggcnn)
    #         # Converte quaternion para Euler para facilitar a leitura
    #         # quat = [qx, qy, qz, qw]
    #         # euler = euler_from_quaternion(quat)

    #         # *************************************************

    #         # Orientação que será utilizada para preensão planar e antipodal obtida no ggcnn
    #         roll = 3.140
    #         pitch = 0.0
    #         yaw = -1*self.ang  # Ângulo obtido do ggcnn   
    #         quat = quaternion_from_euler(roll, pitch, yaw)
    #         euler = euler_from_quaternion(quat)
            

    #         # Imprime a translação
    #         rospy.loginfo(f"  - Translação: [x: {x:.4f}, y: {y:.4f}, z: {z:.4f}]")
    #         # Imprime a rotação
    #         rospy.loginfo(f"  - Rotação (quaternion): [x: {quat[0]:.4f}, y: {quat[1]:.4f}, z: {quat[2]:.4f}, w: {quat[3]:.4f}]")
    #         rospy.loginfo(f"  - Rotação (euler): [roll: {euler[0]:.4f}, pitch: {euler[1]:.4f}, yaw: {euler[2]:.4f}]")
    #         rospy.loginfo("--------------------------------------------------")
    #         rospy.loginfo('\n')

    #         # --- NOVO BLOCO PARA O UNITY (BAIXA LATÊNCIA) ---
    #         unity_pose = PoseStamped()
    #         unity_pose.header.stamp = rospy.Time.now()
    #         unity_pose.header.frame_id = target_frame # Geralmente "base_link"

    #         unity_pose.pose.position.x = x
    #         unity_pose.pose.position.y = y
    #         unity_pose.pose.position.z = z

    #         # Usa a rotação (quaternion) calculada com o ângulo da GGCNN
    #         unity_pose.pose.orientation.x = quat[0]
    #         unity_pose.pose.orientation.y = quat[1]
    #         unity_pose.pose.orientation.z = quat[2]
    #         unity_pose.pose.orientation.w = quat[3]

    #         rospy.loginfo(">> SUCESSO: Publicando a Pose para o Unity!")

    #         self.unity_pose_pub.publish(unity_pose)
    #         # ------------------------------------------------

    #         cmd_msg_grasp = Float32MultiArray()
    #         cmd_msg_grasp.data = [x, y, z, self.ang, self.width_m, self.g_width]
    #         self.cmd_pub_grasp.publish(cmd_msg_grasp)

    #         # Chama a função para publicar a transformação estática 
    #         self.publish_static_transform(x, y, z, roll, pitch, yaw, 'base_link', 'object_grasp')

    #         return transform

    #     except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
    #         rospy.logerr(f"Erro ao buscar transformação: {e}")
    #         return None


    # Observação: Este método não foi utilizado no algoritmo, porém é muito útil
    def publish_static_transform(self, x, y, z, roll, pitch, yaw, parent_frame, child_frame):

        """
        Busca e retorna a transformação entre dois frames.
        Equivalente a 'rosrun tf2_ros static_transform_publisher x y z R P Y parent_frame child_frame'.
        """

        # Cria uma instância do broadcaster
        tf_broadcaster = tf2_ros.StaticTransformBroadcaster()

        # Cria uma mensagem do tipo TransformStamped
        static_transform_stamped = TransformStamped()
        
        # Preenche o cabeçalho
        static_transform_stamped.header.stamp = rospy.Time.now()
        static_transform_stamped.header.frame_id = parent_frame
        static_transform_stamped.child_frame_id = child_frame

        # Preenche a translação (x, y, z)
        static_transform_stamped.transform.translation.x = x
        static_transform_stamped.transform.translation.y = y
        static_transform_stamped.transform.translation.z = z

        # Converte os ângulos de Euler (roll, pitch, yaw) para um quaternion
        quat = quaternion_from_euler(roll, pitch, yaw)
        static_transform_stamped.transform.rotation.x = quat[0]
        static_transform_stamped.transform.rotation.y = quat[1]
        static_transform_stamped.transform.rotation.z = quat[2]
        static_transform_stamped.transform.rotation.w = quat[3]

        # Publica a transformação
        tf_broadcaster.sendTransform(static_transform_stamped)
        # rospy.loginfo(f"Transformação estática publicada: {parent_frame} -> {child_frame}")




def main():
    # Chame a função parse_args() para obter os argumentos
    args = parse_args()
    
    # E passe o objeto de argumentos para a classe
    grasp_detection = ggcnn_grasping(args)
    
    rospy.sleep(3.0)

    # Inicia o loop principal
    input("Press enter to start the GGCNN.....")
    rospy.loginfo("Iniciando processo")
    
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        # Chama as funções de processamento e publicação em sequência
        grasp_detection.depth_process_ggcnn()
        grasp_detection.publish_images()
        grasp_detection.publish_data_to_robot()
        grasp_detection.get_transform_between_frames("base_link", "object_detected")
        
        rate.sleep()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        print("Programa interrompido pelo usuário.")
