#! /usr/bin/env python3
# # Este script foi feito utilizando como base o trabalho "mvp_grasp". Pode ser acessado em 
# # https://github.com/dougsm/mvp_grasp/blob/master/ggcnn/src/ggcnn/ggcnn_torch.py

# import os
# # from os import path
# import sys

# import cv2
# import numpy as np
# import scipy.ndimage as ndimage

# import torch
# import time


# import rospy
# from sensor_msgs.msg import Image
# import cv_bridge
# import numpy as np
# from models.ggcnn import GGCNN 


# # from dougsm_helpers.timeit import TimeIt



# class GraspingNode:
#     def __init__(self):
#         self.latest_depth_msg = None
#         rospy.Subscriber("/camera/depth/image_raw", Image, self.depth_callback)

#     def depth_callback(self, msg):
#         self.latest_depth_msg = msg


# class TimeIt:
#     def __init__(self, s):
#         self.s = s 
#         self.t0 = None
#         self.t1 = None
#         self.print_output = False

#     def __enter__(self): 
#         self.t0 = time.time() # Início do temporizador

#     def __exit__(self, t, value, traceback):
#         self.t1 = time.time() # Fim do temporizador
#         print('%s: %s' % (self.s, self.t1 - self.t0))


# MODEL_FILE_PATH = os.path.join(
#     os.getenv('HOME'),
#     'miguel_ws',
#     'src',
#     'ggcnn_pkg',
#     'scripts',
#     'ggcnn_grasping',
#     'models_trined',
#     'ggcnn_weights_cornell',
#     'ggcnn_epoch_23_cornell_statedict.pt'
# )

# if not os.path.exists(MODEL_FILE_PATH):
#     raise FileNotFoundError(f"O arquivo do modelo não foi encontrado em: {MODEL_FILE_PATH}")

# # Passo 1: Instanciar a classe do modelo PyTorch
# model = GGCNN()

# # Passo 2: Carregar o state_dict (pesos) no modelo
# state_dict = torch.load(MODEL_FILE_PATH, map_location=torch.device('cpu'))
# model.load_state_dict(state_dict)

# # Passo 3: Colocar o modelo em modo de avaliação (necessário para inferência)
# model.eval()

# device = torch.device("cpu")




# def process_depth_image(depth, crop_size, out_size=300, return_mask=False, crop_y_offset=0):
#     imh, imw = depth.shape

#     with TimeIt('1'):
#         # Crop.
#         depth_crop = depth[(imh - crop_size) // 2 - crop_y_offset:(imh - crop_size) // 2 + crop_size - crop_y_offset,
#                            (imw - crop_size) // 2:(imw - crop_size) // 2 + crop_size]
#     # depth_nan_mask = np.isnan(depth_crop).astype(np.uint8)

#     # Inpaint
#     # OpenCV inpainting does weird things at the border.
#     with TimeIt('2'):
#         depth_crop = cv2.copyMakeBorder(depth_crop, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
#         depth_nan_mask = np.isnan(depth_crop).astype(np.uint8)

#     with TimeIt('3'):
#         depth_crop[depth_nan_mask==1] = 0

#     with TimeIt('4'):
#         # Scale to keep as float, but has to be in bounds -1:1 to keep opencv happy.
#         depth_scale = np.abs(depth_crop).max()
#         depth_crop = depth_crop.astype(np.float32) / depth_scale  # Has to be float32, 64 not supported.

#         with TimeIt('Inpainting'):
#             depth_crop = cv2.inpaint(depth_crop, depth_nan_mask, 1, cv2.INPAINT_NS)

#         # Back to original size and value range.
#         depth_crop = depth_crop[1:-1, 1:-1]
#         depth_crop = depth_crop * depth_scale

#     with TimeIt('5'):
#         # Resize
#         depth_crop = cv2.resize(depth_crop, (out_size, out_size), cv2.INTER_AREA)

#     if return_mask:
#         with TimeIt('6'):
#             depth_nan_mask = depth_nan_mask[1:-1, 1:-1]
#             depth_nan_mask = cv2.resize(depth_nan_mask, (out_size, out_size), cv2.INTER_NEAREST)
#         return depth_crop, depth_nan_mask
#     else:
#         return depth_crop


# def predict(depth, process_depth=True, crop_size=300, out_size=300, depth_nan_mask=None, crop_y_offset=0, filters=(2.0, 1.0, 1.0)):
#     if process_depth:
#         depth, depth_nan_mask = process_depth_image(depth, crop_size, out_size=out_size, return_mask=True, crop_y_offset=crop_y_offset)

#     # Inference
#     depth = np.clip((depth - depth.mean()), -1, 1)
#     depthT = torch.from_numpy(depth.reshape(1, 1, out_size, out_size).astype(np.float32)).to(device)
#     with torch.no_grad():
#         pred_out = model(depthT)

#     points_out = pred_out[0].cpu().numpy().squeeze()
#     points_out[depth_nan_mask] = 0

#     # Calculate the angle map.
#     cos_out = pred_out[1].cpu().numpy().squeeze()
#     sin_out = pred_out[2].cpu().numpy().squeeze()
#     ang_out = np.arctan2(sin_out, cos_out) / 2.0

#     width_out = pred_out[3].cpu().numpy().squeeze() * 150.0  # Scaled 0-150:0-1

#     # Filter the outputs.
#     if filters[0]:
#         points_out = ndimage.filters.gaussian_filter(points_out, filters[0])  # 3.0
#     if filters[1]:
#         ang_out = ndimage.filters.gaussian_filter(ang_out, filters[1])
#     if filters[2]:
#         width_out = ndimage.filters.gaussian_filter(width_out, filters[2])

#     points_out = np.clip(points_out, 0.0, 1.0-1e-3)

#     # SM
#     # temp = 0.15
#     # ep = np.exp(points_out / temp)
#     # points_out = ep / ep.sum()

#     # points_out = (points_out - points_out.min())/(points_out.max() - points_out.min())

#     return points_out, ang_out, width_out, depth.squeeze()


# def run_prediction_and_log():
#     rospy.init_node('ggcnn_prediction_node')
#     bridge = cv_bridge.CvBridge()

#     # Wait for the first depth message
#     rospy.loginfo("Waiting for the first depth image message...")
#     try:
#         depth_msg = rospy.wait_for_message("/camera/depth/image_raw", Image, timeout=10.0)
#     except rospy.ROSException as e:
#         rospy.logerr(f"Failed to receive a depth message: {e}")
#         return

#     # Convert ROS message to numpy array
#     depth_image = bridge.imgmsg_to_cv2(depth_msg, "16UC1")

#     # Run the prediction
#     points_out, ang_out, width_out, depth_image = predict(depth_image, process_depth=True)

#     # Find the best grasp
#     max_pixel = np.array(np.unravel_index(np.argmax(points_out), points_out.shape))
#     best_y, best_x = max_pixel[0], max_pixel[1]

#     # Extract the values for the best grasp
#     best_angle = ang_out[best_y, best_x]
#     best_width = width_out[best_y, best_x]
#     best_quality = points_out[best_y, best_x]

#     # Log the results
#     rospy.loginfo("--- GG-CNN Prediction Results ---")
#     rospy.loginfo(f"Best Grasp Quality: {best_quality}")
#     rospy.loginfo(f"Grasp Pixel (y, x): ({best_y}, {best_x})")
#     rospy.loginfo(f"Grasp Angle: {best_angle} radians")
#     rospy.loginfo(f"Grasp Width: {best_width} (in pixel units)")
#     rospy.loginfo("-----------------------------------")
    
#     rospy.signal_shutdown("Prediction completed.")

# if __name__ == '__main__':
#     try:
#         run_prediction_and_log()
#     except rospy.ROSInterruptException:
#         pass









# #! /usr/bin/env python3

# # Este script foi feito utilizando como base o trabalho "ggcnn_kinova_grasping". Pode ser acessado em 
# # https://github.com/dougsm/ggcnn_kinova_grasping/blob/master/ggcnn_kinova_grasping/scripts/run_ggcnn.py
# # O script original foi feito com TensorFlow, porém, fiz a conversão para Pytorch


# import time

# import numpy as np
# import tensorflow as tf
# from keras.models import load_model

# import cv2
# import scipy.ndimage as ndimage
# from skimage.draw import circle
# from skimage.feature import peak_local_max

# import rospy
# from cv_bridge import CvBridge
# from geometry_msgs.msg import PoseStamped
# from sensor_msgs.msg import Image, CameraInfo
# from std_msgs.msg import Float32MultiArray

# bridge = CvBridge()

# # Load the Network.
# MODEL_FILE = 'PATH/TO/model.hdf5'
# model = load_model(MODEL_FILE)

# rospy.init_node('ggcnn_detection')

# # Output publishers.
# grasp_pub = rospy.Publisher('ggcnn/img/grasp', Image, queue_size=1)
# grasp_plain_pub = rospy.Publisher('ggcnn/img/grasp_plain', Image, queue_size=1)
# depth_pub = rospy.Publisher('ggcnn/img/depth', Image, queue_size=1)
# ang_pub = rospy.Publisher('ggcnn/img/ang', Image, queue_size=1)
# cmd_pub = rospy.Publisher('ggcnn/out/command', Float32MultiArray, queue_size=1)

# # Initialise some globals.
# prev_mp = np.array([150, 150])
# ROBOT_Z = 0

# # Tensorflow graph to allow use in callback.
# graph = tf.get_default_graph()

# # Get the camera parameters
# camera_info_msg = rospy.wait_for_message('/camera/depth/camera_info', CameraInfo)
# K = camera_info_msg.K
# fx = K[0]
# cx = K[2]
# fy = K[4]
# cy = K[5]


# # Execution Timing
# class TimeIt:
#     def __init__(self, s):
#         self.s = s
#         self.t0 = None
#         self.t1 = None
#         self.print_output = False

#     def __enter__(self):
#         self.t0 = time.time()

#     def __exit__(self, t, value, traceback):
#         self.t1 = time.time()
#         if self.print_output:
#             print('%s: %s' % (self.s, self.t1 - self.t0))


# def robot_pos_callback(data):
#     global ROBOT_Z
#     ROBOT_Z = data.pose.position.z


# def depth_callback(depth_message):
#     global model
#     global graph
#     global prev_mp
#     global ROBOT_Z
#     global fx, cx, fy, cy

#     with TimeIt('Crop'):
#         depth = bridge.imgmsg_to_cv2(depth_message)

#         # Crop a square out of the middle of the depth and resize it to 300*300
#         crop_size = 400
#         depth_crop = cv2.resize(depth[(480-crop_size)//2:(480-crop_size)//2+crop_size, (640-crop_size)//2:(640-crop_size)//2+crop_size], (300, 300))

#         # Replace nan with 0 for inpainting.
#         depth_crop = depth_crop.copy()
#         depth_nan = np.isnan(depth_crop).copy()
#         depth_crop[depth_nan] = 0

#     with TimeIt('Inpaint'):
#         # open cv inpainting does weird things at the border.
#         depth_crop = cv2.copyMakeBorder(depth_crop, 1, 1, 1, 1, cv2.BORDER_DEFAULT)

#         mask = (depth_crop == 0).astype(np.uint8)
#         # Scale to keep as float, but has to be in bounds -1:1 to keep opencv happy.
#         depth_scale = np.abs(depth_crop).max()
#         depth_crop = depth_crop.astype(np.float32)/depth_scale  # Has to be float32, 64 not supported.

#         depth_crop = cv2.inpaint(depth_crop, mask, 1, cv2.INPAINT_NS)

#         # Back to original size and value range.
#         depth_crop = depth_crop[1:-1, 1:-1]
#         depth_crop = depth_crop * depth_scale

#     with TimeIt('Calculate Depth'):
#         # Figure out roughly the depth in mm of the part between the grippers for collision avoidance.
#         depth_center = depth_crop[100:141, 130:171].flatten()
#         depth_center.sort()
#         depth_center = depth_center[:10].mean() * 1000.0

#     with TimeIt('Inference'):
#         # Run it through the network.
#         depth_crop = np.clip((depth_crop - depth_crop.mean()), -1, 1)
#         with graph.as_default():
#             pred_out = model.predict(depth_crop.reshape((1, 300, 300, 1)))

#         points_out = pred_out[0].squeeze()
#         points_out[depth_nan] = 0

#     with TimeIt('Trig'):
#         # Calculate the angle map.
#         cos_out = pred_out[1].squeeze()
#         sin_out = pred_out[2].squeeze()
#         ang_out = np.arctan2(sin_out, cos_out)/2.0

#         width_out = pred_out[3].squeeze() * 150.0  # Scaled 0-150:0-1

#     with TimeIt('Filter'):
#         # Filter the outputs.
#         points_out = ndimage.filters.gaussian_filter(points_out, 5.0)  # 3.0
#         ang_out = ndimage.filters.gaussian_filter(ang_out, 2.0)

#     with TimeIt('Control'):
#         # Calculate the best pose from the camera intrinsics.
#         maxes = None

#         ALWAYS_MAX = False  # Use ALWAYS_MAX = True for the open-loop solution.

#         if ROBOT_Z > 0.34 or ALWAYS_MAX:  # > 0.34 initialises the max tracking when the robot is reset.
#             # Track the global max.
#             max_pixel = np.array(np.unravel_index(np.argmax(points_out), points_out.shape))
#             prev_mp = max_pixel.astype(np.int)
#         else:
#             # Calculate a set of local maxes.  Choose the one that is closes to the previous one.
#             maxes = peak_local_max(points_out, min_distance=10, threshold_abs=0.1, num_peaks=3)
#             if maxes.shape[0] == 0:
#                 return
#             max_pixel = maxes[np.argmin(np.linalg.norm(maxes - prev_mp, axis=1))]

#             # Keep a global copy for next iteration.
#             prev_mp = (max_pixel * 0.25 + prev_mp * 0.75).astype(np.int)

#         ang = ang_out[max_pixel[0], max_pixel[1]]
#         width = width_out[max_pixel[0], max_pixel[1]]

#         # Convert max_pixel back to uncropped/resized image coordinates in order to do the camera transform.
#         max_pixel = ((np.array(max_pixel) / 300.0 * crop_size) + np.array([(480 - crop_size)//2, (640 - crop_size) // 2]))
#         max_pixel = np.round(max_pixel).astype(np.int)

#         point_depth = depth[max_pixel[0], max_pixel[1]]

#         # These magic numbers are my camera intrinsic parameters.
#         x = (max_pixel[1] - cx)/(fx) * point_depth
#         y = (max_pixel[0] - cy)/(fy) * point_depth
#         z = point_depth

#         if np.isnan(z):
#             return

#     with TimeIt('Draw'):
#         # Draw grasp markers on the points_out and publish it. (for visualisation)
#         grasp_img = np.zeros((300, 300, 3), dtype=np.uint8)
#         grasp_img[:,:,2] = (points_out * 255.0)

#         grasp_img_plain = grasp_img.copy()

#         rr, cc = circle(prev_mp[0], prev_mp[1], 5)
#         grasp_img[rr, cc, 0] = 0
#         grasp_img[rr, cc, 1] = 255
#         grasp_img[rr, cc, 2] = 0

#     with TimeIt('Publish'):
#         # Publish the output images (not used for control, only visualisation)
#         grasp_img = bridge.cv2_to_imgmsg(grasp_img, 'bgr8')
#         grasp_img.header = depth_message.header
#         grasp_pub.publish(grasp_img)

#         grasp_img_plain = bridge.cv2_to_imgmsg(grasp_img_plain, 'bgr8')
#         grasp_img_plain.header = depth_message.header
#         grasp_plain_pub.publish(grasp_img_plain)

#         depth_pub.publish(bridge.cv2_to_imgmsg(depth_crop))

#         ang_pub.publish(bridge.cv2_to_imgmsg(ang_out))

#         # Output the best grasp pose relative to camera.
#         cmd_msg = Float32MultiArray()
#         cmd_msg.data = [x, y, z, ang, width, depth_center]
#         cmd_pub.publish(cmd_msg)


# depth_sub = rospy.Subscriber('/camera/depth/image_meters', Image, depth_callback, queue_size=1)
# robot_pos_sub = rospy.Subscriber('/m1n6s200_driver/out/tool_pose', PoseStamped, robot_pos_callback, queue_size=1)

# while not rospy.is_shutdown():
#     rospy.spin()




















# Backup do script. Fiz com base no trabalho de Caio convertendo de TensorFlow para Pytorch

# ! /usr/bin/env python3

# Python
import time
import numpy as np
import argparse # Para argumentos de linha de comando
from copy import deepcopy

# CNN
import torch
import torch.nn as nn
# import torch.nn.functional as F
import tf2_ros
import tf2_geometry_msgs

# Image
import cv2
# from skimage.draw import circle
import scipy.ndimage as ndimage

# ROS
import rospy
import rospkg
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32MultiArray
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from geometry_msgs.msg import TransformStamped, PoseStamped # Adicionado para tf2

# Importe a classe GGCNN que você criou
from models.ggcnn import GGCNN  

class TimeIt:
    def __init__(self, s):
        self.s = s 
        self.t0 = None
        self.t1 = None
        self.print_output = False

    def __enter__(self): 
        self.t0 = time.time() # Início do temporizador

    def __exit__(self, t, value, traceback):
        self.t1 = time.time() # Fim do temporizador
        print('%s: %s' % (self.s, self.t1 - self.t0))

def parse_args():
    parser = argparse.ArgumentParser(description='GGCNN grasping')
    parser.add_argument('--real', action='store_true', help='Consider the real intel realsense')
    parser.add_argument('--plot', action='store_true', help='Plot depth image')
    parser.add_argument('--ggcnn', action = 'store', help = 'Publish the ggcnn results')
    args = parser.parse_args()
    return args

class ggcnn_grasping(object):
    def __init__(self, args):
        rospy.init_node('ggcnn_detection', anonymous=True)

        self.args = args
        self.bridge = CvBridge()

        
        # Carregar a rede PyTorch
        rospack = rospkg.RosPack()
        Home = rospack.get_path('ggcnn_pkg')
        MODEL_FILE = Home + '/scripts/ggcnn_grasping/models_trined/ggcnn_weights_cornell/ggcnn_epoch_23_cornell_statedict.pt'  # /data/epoch_29_model.hdf5 --> (treinamento de Caio)
        self.model = GGCNN().to(torch.device("cpu"))
        self.model.load_state_dict(torch.load(MODEL_FILE, map_location=torch.device('cpu')))
        self.model.eval()

        # Configurar TF2
        self.tf_buffer = tf2_ros.Buffer() # Cria o buffer do tf2 para armazenar as transformações
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer) # Cria o listener do tf2 para ouvir as transformações
        self.tf_broadcaster = tf2_ros.TransformBroadcaster() # Cria o broadcaster do tf2 para publicar as transformações

        # Load GGCN parameters
        self.crop_size = rospy.get_param("/GGCNN/crop_size", 300)
        self.FOV = rospy.get_param("/GGCNN/FOV", 60)
        self.camera_topic_info = rospy.get_param("/GGCNN/camera_topic_info", "/camera/depth/camera_info")
        if self.args.real:
            self.camera_topic = rospy.get_param("/GGCNN/camera_topic_realsense", "/camera/depth/image_raw")
        else:
            self.camera_topic = rospy.get_param("/GGCNN/camera_topic", "/camera/depth/image_raw")

        # Output publishers.
        self.grasp_pub = rospy.Publisher('ggcnn/img/grasp', Image, queue_size=1)  # Imagem com o pixel de maior probabilidade de preensão
        self.depth_pub = rospy.Publisher('ggcnn/img/depth', Image, queue_size=1) # Imagem de profundidade processada
        self.width_pub = rospy.Publisher('ggcnn/img/width', Image, queue_size=1) # Imagem da largura da preensão
        self.depth_pub_shot = rospy.Publisher('ggcnn/img/depth_shot', Image, queue_size=1) # Imagem de profundidade original
        self.ang_pub = rospy.Publisher('ggcnn/img/ang', Image, queue_size=1) # Imagem do ângulo da preensão
        self.cmd_pub = rospy.Publisher('ggcnn/out/command', Float32MultiArray, queue_size=1) # Comando de preensão [x,y,z,theta,width]
        
        # Initialize some var
        self.grasping_point = []
        self.depth_image_shot = None
        self.points_vec = []
        self.depth_message = None
        # self.latest_depth_message = None  # Armazena a última mensagem de profundidade recebida
        self.color_img = None # Armazena a última imagem colorida recebida

        # Get the camera parameters
        camera_info_msg = rospy.wait_for_message(self.camera_topic_info, CameraInfo)
        K = camera_info_msg.K
        self.fx = K[0]
        self.cx = K[2]
        self.fy = K[4]
        self.cy = K[5]

        # Subscribers
        rospy.Subscriber(self.camera_topic, Image, self.get_depth_callback, queue_size=10) # Suscriber para a imagem de profundidade
        rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback, queue_size=10) # Suscriber para a imagem colorida

    def get_depth_callback(self, depth_message):
        self.depth_message = depth_message
        # cv2.imshow("Imagem_depth", self.latest_depth_message)
        # cv2.waitKey(1)
        
    def image_callback(self, color_msg):
        self.color_img = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
        # cv2.imshow("Imagem_RGB", self.color_img)
        # cv2.waitKey(1)



    def get_depth_image_shot(self):
        self.depth_image_shot = rospy.wait_for_message("camera/depth/image_raw", Image)
        self.depth_image_shot.header = self.depth_message.header
        self.depth_pub_shot.publish(self.depth_image_shot)   # publicar imagem de profundidade original

        # if self.points_out is None or self.depth_message_ggcnn is None or self.depth_crop is None:
        #     return

        # # --- Visualização da imagem de Profundidade ---
        # # 1. Copie o depth_crop para evitar modificar o original
        # depth_for_display = self.depth_crop.copy()

        # # 2. Normalização: Mapeia os valores de profundidade para 0-255
        # #    Encontra os valores mínimo e máximo na imagem de profundidade
        # min_depth_val = np.min(depth_for_display[depth_for_display > 0]) # Ignora zeros (possíveis Nans)
        # max_depth_val = np.max(depth_for_display)

        # if max_depth_val > min_depth_val:
        #     # Escala para 0-1
        #     depth_for_display = (depth_for_display - min_depth_val) / (max_depth_val - min_depth_val)
        #     # Converte para 0-255 e tipo de inteiro para visualização
        #     depth_for_display = (depth_for_display * 255).astype(np.uint8)
        #     # Para visualização em cores falsas (melhor para distinguir profundidades)
        #     depth_for_display = cv2.applyColorMap(depth_for_display, cv2.COLORMAP_JET)
        # else:
        #     # Caso não haja variação de profundidade ou todos os valores sejam zero
        #     depth_for_display = np.zeros_like(self.depth_crop, dtype=np.uint8)
        #     depth_for_display = cv2.applyColorMap(depth_for_display, cv2.COLORMAP_JET) # Para manter o formato

        # # Publica a imagem de profundidade normalizada
        # depth_msg = self.bridge.cv2_to_imgmsg(depth_for_display, 'bgr8') # 'bgr8' porque applyColorMap cria 3 canais
        # depth_msg.header = self.depth_message_ggcnn.header
        # self.depth_pub.publish(depth_msg)

        # # ... (código para grasp_img, ang_out, width_img) ...







   # ************************************************************

    #ESTA FUNÇÃO PRECISA DE MELHORIAS, TIPO REDE CONVOLUCIONAL YOLO PARA DETECTAR OBJETOS
    def copy_obj_to_depth_img(self):
        points = self.points_vec
        depth_image_shot = deepcopy(self.depth_image_shot)
        depth_image_shot = self.bridge.imgmsg_to_cv2(depth_image_shot)
        depth_image_shot_copy = depth_image_shot.copy()

        depth_message = self.depth_message
        depth_message = self.bridge.imgmsg_to_cv2(depth_message)
        depth_message_copy = depth_message.copy()

        number_of_boxes = len(points)
        i = 0
        while i < number_of_boxes:
            depth_image_shot_copy[points[i][1] : points[i][3], points[i][0] : points[i][2]] \
             = depth_message_copy[points[i][1] : points[i][3], points[i][0] : points[i][2]] 
            i += 1

        depth_image_shot = self.bridge.cv2_to_imgmsg(depth_image_shot_copy)
        depth_image_shot.header = self.depth_message.header
        self.depth_image_shot_with_object_copied = depth_image_shot

        self.depth_pub_copied_img.publish(depth_image_shot)
        self.depth_pub_shot.publish(self.depth_image_shot)   # publicar imagem de profundidade original

    # ************************************************************

    def depth_process_ggcnn(self):
        depth_message = self.depth_message
        if depth_message is None or self.color_img is None:
            return

        # INPUT
        depth = self.bridge.imgmsg_to_cv2(depth_message, "16UC1") # Converte a imagem de profundidade para um array numpy com medidas em milímetros
        depth = depth.astype(np.float32) # Converte para float32
        # depth /= 1000.0 # Converte para metros
        depth_copy_for_point_depth = depth.copy() # Cópia para pegar a profundidade do ponto de preensão
        
        height_res, width_res = depth.shape # height_res e width_res são as dimensões da imagem de profundidade
        
        # It crops a 300x300 resolution square at the top of the depth image - depth[0:300, 170:470]
        depth_crop = depth[0 : self.crop_size, (width_res - self.crop_size)//2 : (width_res - self.crop_size)//2 + self.crop_size] # Crop centralizado da imagem de profundidade (representa a área visível pela câmera)
        depth_crop = depth_crop.copy() # Creates a deep copy of the depth_crop image
        # Returns the positions represented by nan values
        depth_nan = np.isnan(depth_crop) # Máscara para valores NaN
        depth_nan = depth_nan.copy() 
        # Substitute nan values by zero
        depth_crop[depth_nan] = 0
        
        # ************************************************
        # INPAINT PROCESS - Usando OpenCV
        depth_crop = cv2.copyMakeBorder(depth_crop, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
        # se o numero que esta no vetor acima for 0, retorna o numero 1 na mesma posicao (como se fosse True)
        # se depth_crop == 0, retorna 1 como inteiro.
        # Ou seja, copia os pixels pretos da imagem e a posicao deles
        mask = (depth_crop == 0).astype(np.uint8)
        # Scale to keep as float, but has to be in bounds -1:1 to keep opencv happy.
        depth_scale = np.abs(depth_crop).max()
        # Normalize
        depth_crop = depth_crop.astype(np.float32) / depth_scale # Normaliza a imagem de profundidade para o intervalo [-1, 1]. Has to be float32, 64 not supported
        # Substitute mask values by near values. See opencv doc for more details
        depth_crop = cv2.inpaint(depth_crop, mask, 1, cv2.INPAINT_NS) 
        # Back to original size and value range.
        depth_crop = depth_crop[1:-1, 1:-1] # Remove a borda adicionada anteriormente
        # reescale image
        depth_crop = depth_crop * depth_scale # Desnormaliza a imagem de profundidade

        # ************************************************
        # INFERENCE PROCESS
        depth_crop = np.clip((depth_crop - depth_crop.mean()), -1, 1) # Normaliza a imagem de profundidade para ter média zero e valores entre -1 e 1
        self.depth_crop = depth_crop.copy()
        depth_tensor = torch.from_numpy(depth_crop).unsqueeze(0).unsqueeze(0)
        # mostrar o formato do tensor
        rospy.loginfo(f"Formato do tensor de profundidade: {depth_tensor.shape}") # Deve ser (1, 1, 300, 300)
        

        with torch.no_grad():
            pred_out = self.model(depth_tensor) # Formato de entrada: (batch_size, channels, height, width)
            # qual_out = points_out  # Usando a saída de pontos como qualidade, valor entre 0 e 1

        # Remover dimensão do batch e canal extra
        points_out = pred_out[0].squeeze().cpu().numpy()
        cos_out = pred_out[1].squeeze().cpu().numpy()
        sin_out = pred_out[2].squeeze().cpu().numpy()
        width_out  = (pred_out[3].squeeze().cpu().numpy()) * 150.0
        # Zerar valores inválidos
        points_out[depth_nan] = 0
        # Calcular ângulo
        ang_out = np.arctan2(sin_out, cos_out) / 2.0

        # ************************************************
        # Filtros (simplificados)
        # FILTERING PROCESS
        # The filters are applied to augment the chances of getting a good grasp pose
        # points_out_filtered  = cv2.GaussianBlur(points_out, (5, 5), 0)  
        # points_out_filtered  = np.clip(points_out_filtered , 0.0, 1.0 - 1e-3)
        # ang_out_filtered = cv2.GaussianBlur(ang_out, (5, 5), 0)
        # width_out_filtered = cv2.GaussianBlur(width_out, (5, 5), 0)


        filters=(2.0, 1.0, 1.0)
        points_out_filtered  = ndimage.filters.gaussian_filter(points_out, filters[0])  # 3.0
        points_out_filtered  = np.clip(points_out_filtered, 0.0, 1.0-1e-3)
        ang_out_filtered = ndimage.filters.gaussian_filter(ang_out, filters[1])
        width_out_filtered = ndimage.filters.gaussian_filter(width_out, filters[2])

        # ************************************************        
        # CONTROL PROCESS
        # Corrigido para tf2
        grasping_link = 'robotiq_85_base_link'
        # link_pose = self.tf_buffer.lookup_transform("base_link", grasping_link, rospy.Time(0), rospy.Duration(1.0))
        # ROBOT_Z = link_pose.transform.translation.z    

        try:
            link_pose = self.tf_buffer.lookup_transform("base_link", grasping_link, rospy.Time(0), rospy.Duration(1.0))
            ROBOT_Z = link_pose.transform.translation.z
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            ROBOT_Z = 0.0 # Valor padrão em caso de erro



        # Track the global max.
        # max_pixel correponds to the position of the max value in points_out_filtered
        max_pixel = np.array(np.unravel_index(np.argmax(points_out_filtered), points_out_filtered.shape))  # maior pontuação de qualidade
        # Return max_pixel position as an int (300x300)
        max_pixel = max_pixel.astype(int)

        rospy.loginfo(f"Pixel com maior qualidade: x = {max_pixel[1]}, y = {max_pixel[0]}")
        
        ang = ang_out_filtered[max_pixel[0], max_pixel[1]]
        width_px = width_out_filtered[max_pixel[0], max_pixel[1]]
        
        # Conversão de pixel para metro
        # height_res, width_res = depth.shape
        reescaled_height = int(max_pixel[0])
        reescaled_width = int((width_res - self.crop_size) // 2 + max_pixel[1])
        max_pixel_reescaled = [reescaled_height, reescaled_width]
        point_depth = depth_copy_for_point_depth[max_pixel_reescaled[0], max_pixel_reescaled[1]]

        # GRASP WIDTH PROCESS
        g_width = 2.0 * (ROBOT_Z + 0.24) * np.tan(self.FOV / height_res * width_px / 2.0 / 180.0 * np.pi) #* 0.37
        crop_size_width = float(self.crop_size)
        width_m = width_out_filtered / crop_size_width * 2.0 * point_depth * np.tan(self.FOV * crop_size_width / height_res / 2.0 / 180.0 * np.pi) / 1000 #* 0.37
        width_m = abs(width_m[max_pixel[0], max_pixel[1]])


        if not np.isnan(point_depth):
            # These magic numbers are my camera intrinsic parameters.
            x = (max_pixel_reescaled[1] - self.cx)/(self.fx) * point_depth
            y = (max_pixel_reescaled[0] - self.cy)/(self.fy) * point_depth
            grasping_point = [x, y, point_depth]

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

        # rospy.loginfo(f"Parâmetros da preensão: x={points_out[0]:.4f}, y={points_out[1]:.4f}, z={point_depth:.4f}, ângulo={ang:.2f}, largura={width_m:.4f}")


    def publish_data_for_image_reading(self):
        width_px = self.width_px
        max_px = self.max_pixel
        ang = self.ang
        max_px_h = float(max_px[0])
        max_px_w = float(max_px[1])
        ggcnn_cmd_msg = Float32MultiArray()
        ggcnn_cmd_msg.data = [width_px, max_px_h, max_px_w, ang]


    def get_grasp_image(self):
        """
        Show the detected grasp regions of the image
        """
        points_out = self.points_out

        if points_out is not None:
            max_pixel = self.max_pixel

            # Draw grasp markers on the points_out and publish it. (for visualisation)
            # points_out was used in gaussian_filter for last
            grasp_img = np.zeros((self.crop_size, self.crop_size, 3), dtype=np.uint8)
            # Draw the red area in the image
            grasp_img[:,:,2] = (points_out * 255.0)
            # draw the circle at the green point
            cv2.circle(grasp_img, (max_pixel[1], max_pixel[0]), 5, (0, 255, 0), -1)

            self.grasp_img = grasp_img



    def publish_images(self):
        grasp_img = self.grasp_img
        depth_message = self.depth_message_ggcnn
        ang_out = self.ang_out
        depth_crop = self.depth_crop
        width_img = self.width_out

        if grasp_img is not None:
            #Publish the output images (not used for control, only visualisation)
            grasp_img = self.bridge.cv2_to_imgmsg(grasp_img, 'bgr8')
            grasp_img.header = depth_message.header 
            self.grasp_pub.publish(grasp_img) # publicar imagem com o pixel de maior probabilidade de preensão
            
            depth_crop = self.bridge.cv2_to_imgmsg(depth_crop)
            depth_crop.header = depth_message.header
            self.depth_pub.publish(depth_crop)  # publicar imagem de profundidade processada
            
            self.ang_pub.publish(self.bridge.cv2_to_imgmsg(ang_out))  # publicar imagem do ângulo da preensão
            self.width_pub.publish(self.bridge.cv2_to_imgmsg(width_img)) # publicar imagem da largura da preensão



        if self.points_out is not None:
            grasp_img = np.zeros((self.crop_size, self.crop_size, 3), dtype=np.uint8)
            grasp_img[:,:,2] = (self.points_out * 255.0)
            max_pixel = np.array(np.unravel_index(np.argmax(self.points_out), self.points_out.shape))
            cv2.circle(grasp_img, (max_pixel[1], max_pixel[0]), 5, (0, 255, 0), -1)

            grasp_msg = self.bridge.cv2_to_imgmsg(grasp_img, 'bgr8')
            grasp_msg.header = self.depth_message_ggcnn.header
            self.grasp_pub.publish(grasp_msg)

            depth_msg = self.bridge.cv2_to_imgmsg(self.depth_crop)
            depth_msg.header = self.depth_message_ggcnn.header
            self.depth_pub.publish(depth_msg)



    def publish_data_to_robot(self):
        if not self.grasping_point:
            return

        grasping_point = self.grasping_point
        ang = self.ang
        width_m = self.width_m
        g_width = self.g_width

        # Output the best grasp pose relative to camera.
        cmd_msg = Float32MultiArray()
        cmd_msg.data = [grasping_point[0]/1000, grasping_point[1]/1000, grasping_point[2]/1000, -1*ang, width_m, g_width]
        
        # Mostrar no loginfo os valores calculados 
        rospy.loginfo(f"Comando de preensão: x={cmd_msg.data[0]:.4f}, y={cmd_msg.data[1]:.4f}, z={cmd_msg.data[2]:.4f}, ângulo={cmd_msg.data[3]:.2f}, largura={cmd_msg.data[4]:.4f}")        
        
        # Publicar os valores
        self.cmd_pub.publish(cmd_msg)  # publicar comando de preensão [x,y,z,theta,width_m, g_width]
        
        # Corrigido para tf2
        grasp_transform = TransformStamped()  # geometry_msgs.msg.TransformStamped
        grasp_transform.header.stamp = rospy.Time.now() # Sempre use o tempo atual
        grasp_transform.header.frame_id = "camera_depth_optical_frame" # Frame da câmera (pai)
        grasp_transform.child_frame_id = "object_detected" # Nome do frame filho que estamos criando
        grasp_transform.transform.translation.x = cmd_msg.data[0]
        grasp_transform.transform.translation.y = cmd_msg.data[1]
        grasp_transform.transform.translation.z = cmd_msg.data[2]
        q = quaternion_from_euler(0.0, 0.0, -1*cmd_msg.data[3])
        grasp_transform.transform.rotation.x = q[0]
        grasp_transform.transform.rotation.y = q[1]
        grasp_transform.transform.rotation.z = q[2]
        grasp_transform.transform.rotation.w = q[3]

        self.tf_broadcaster.sendTransform(grasp_transform)

        # # Publica continuamente a transformação
        # rate = rospy.Rate(10)
        # while not rospy.is_shutdown():
        #     grasp_transform.header.stamp = rospy.Time.now()
        #     self.tf_broadcaster.sendTransform(grasp_transform)
        #     rate.sleep()


def main():
    args = parse_args()
    ggcnn_grasp = ggcnn_grasping(args)
    rospy.sleep(3.0)

    if args.ggcnn:
        input("Move the robot to the pre-grasp position.")
        # ggcnn_grasp.get_depth_image_shot()

    input("Press enter to start the GGCNN ...")

    rospy.loginfo("Starting process")
    
    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        # if args.ssggcnn:
        #     ggcnn_grasp.copy_obj_to_depth_img()
        with TimeIt('ggcnn_process'):
            ggcnn_grasp.depth_process_ggcnn()

        ggcnn_grasp.publish_data_for_image_reading()
        ggcnn_grasp.get_grasp_image()
        ggcnn_grasp.publish_images()
        ggcnn_grasp.publish_data_to_robot()

        ggcnn_grasp.get_depth_image_shot()
        # ggcnn_grasp.copy_obj_to_depth_img()

        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        print ("Program interrupted before completion")



















# BACKUP DO CÓDIGO ANTIGO (run_ggcnn_2.py) --> caso as modificações não funcionem
## ! /usr/bin/env python3

# # Python
# import time
# import numpy as np
# import argparse # Para argumentos de linha de comando

# # CNN
# import torch
# import torch.nn as nn
# # import torch.nn.functional as F
# import tf2_ros
# import tf2_geometry_msgs

# # Image
# import cv2
# # from skimage.draw import circle

# # ROS
# import rospy
# import rospkg
# from cv_bridge import CvBridge
# from sensor_msgs.msg import Image, CameraInfo
# from std_msgs.msg import Float32MultiArray
# from tf.transformations import quaternion_from_euler, euler_from_quaternion
# from geometry_msgs.msg import TransformStamped, PoseStamped # Adicionado para tf2

# # Importe a classe GGCNN que você criou
# from models.ggcnn import GGCNN  

# class TimeIt:
#     def __init__(self, s):
#         self.s = s 
#         self.t0 = None
#         self.t1 = None
#         self.print_output = False

#     def __enter__(self): 
#         self.t0 = time.time() # Início do temporizador

#     def __exit__(self, t, value, traceback):
#         self.t1 = time.time() # Fim do temporizador
#         print('%s: %s' % (self.s, self.t1 - self.t0))

# def parse_args():
#     parser = argparse.ArgumentParser(description='GGCNN grasping')
#     parser.add_argument('--real', action='store_true', help='Consider the real intel realsense')
#     parser.add_argument('--plot', action='store_true', help='Plot depth image')
#     args = parser.parse_args()
#     return args

# class ggcnn_grasping(object):
#     def __init__(self, args):
#         rospy.init_node('ggcnn_detection', anonymous=True)

#         self.args = args
#         self.bridge = CvBridge()
#         self.latest_depth_message = None  # Armazena a última mensagem de profundidade recebida
#         self.color_img = None # Armazena a última imagem colorida recebida
        
#         # Carregar a rede PyTorch
#         rospack = rospkg.RosPack()
#         Home = rospack.get_path('ggcnn_pkg')
#         MODEL_FILE = Home + '/scripts/ggcnn_grasping/models_trined/ggcnn_weights_cornell/ggcnn_epoch_23_cornell_statedict.pt'  # /data/epoch_29_model.hdf5 --> (treinamento de Caio)
#         self.model = GGCNN().to(torch.device("cpu"))
#         self.model.load_state_dict(torch.load(MODEL_FILE, map_location=torch.device('cpu')))
#         self.model.eval()

#         # Configurar TF2
#         self.tf_buffer = tf2_ros.Buffer() # Cria o buffer do tf2 para armazenar as transformações
#         self.tf_listener = tf2_ros.TransformListener(self.tf_buffer) # Cria o listener do tf2 para ouvir as transformações
#         self.tf_broadcaster = tf2_ros.TransformBroadcaster() # Cria o broadcaster do tf2 para publicar as transformações

#         # Load GGCN parameters
#         self.crop_size = rospy.get_param("/GGCNN/crop_size", 300)
#         self.FOV = rospy.get_param("/GGCNN/FOV", 60)
#         self.camera_topic_info = rospy.get_param("/GGCNN/camera_topic_info", "/camera/depth/camera_info")
#         if self.args.real:
#             self.camera_topic = rospy.get_param("/GGCNN/camera_topic_realsense", "/camera/depth/image_raw")
#         else:
#             self.camera_topic = rospy.get_param("/GGCNN/camera_topic", "/camera/depth/image_raw")

#         # Output publishers.
#         self.grasp_pub = rospy.Publisher('ggcnn/img/grasp', Image, queue_size=1)  # Imagem com o pixel de maior probabilidade de preensão
#         self.depth_pub = rospy.Publisher('ggcnn/img/depth', Image, queue_size=1) # Imagem de profundidade processada
#         self.width_pub = rospy.Publisher('ggcnn/img/width', Image, queue_size=1) # Imagem da largura da preensão
#         self.depth_pub_shot = rospy.Publisher('ggcnn/img/depth_shot', Image, queue_size=1) # Imagem de profundidade original
#         self.ang_pub = rospy.Publisher('ggcnn/img/ang', Image, queue_size=1) # Imagem do ângulo da preensão
#         self.cmd_pub = rospy.Publisher('ggcnn/out/command', Float32MultiArray, queue_size=1) # Comando de preensão [x,y,z,theta,width]
        
#         # Initialize some var
#         self.grasping_point = []
#         self.depth_image_shot = None
#         self.depth_image_shot = None  # adicionei essa linha para utilizar na função "get_depth_image_shot"
        
#         # Get the camera parameters
#         camera_info_msg = rospy.wait_for_message(self.camera_topic_info, CameraInfo)
#         K = camera_info_msg.K
#         self.fx = K[0]
#         self.cx = K[2]
#         self.fy = K[4]
#         self.cy = K[5]

#         # Subscribers
#         rospy.Subscriber(self.camera_topic, Image, self.get_depth_callback, queue_size=10) # Suscriber para a imagem de profundidade
#         rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback, queue_size=10) # Suscriber para a imagem colorida

#     def get_depth_callback(self, depth_message):
#         self.latest_depth_message = depth_message
        
        
#     def image_callback(self, color_msg):
#         self.color_img = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
#         # cv2.imshow("Color Image", self.color_img)
#         # cv2.waitKey(1)


# # **************************************************
#     # Adicionaei esse função
#     def get_depth_image_shot(self):
#         self.depth_image_shot = rospy.wait_for_message("camera/depth/image_raw", Image)
#         self.depth_image_shot.header = self.depth_message.header 

# #***************************************************




#     def depth_process_ggcnn(self):
#         depth_message = self.latest_depth_message
#         if depth_message is None or self.color_img is None:
#             return

#         # INPUT
#         depth = self.bridge.imgmsg_to_cv2(depth_message, "16UC1")
#         depth = depth.astype(np.float32)
#         depth /= 1000.0
#         depth_copy_for_point_depth = depth.copy()
        
#         height_res, width_res = depth.shape
#         depth_crop = depth[0 : self.crop_size, (width_res - self.crop_size)//2 : (width_res - self.crop_size)//2 + self.crop_size]
#         depth_crop = depth_crop.copy()
#         depth_nan = np.isnan(depth_crop)
#         depth_nan = depth_nan.copy()
#         depth_crop[depth_nan] = 0

#         # INPAINT PROCESS - Usando OpenCV
#         depth_crop = cv2.copyMakeBorder(depth_crop, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
#         mask = (depth_crop == 0).astype(np.uint8)
#         depth_scale = np.abs(depth_crop).max()
#         depth_crop = depth_crop.astype(np.float32) / depth_scale
#         depth_crop = cv2.inpaint(depth_crop, mask, 1, cv2.INPAINT_NS)
#         depth_crop = depth_crop[1:-1, 1:-1]
#         depth_crop = depth_crop * depth_scale

#         # INFERENCE PROCESS
#         depth_crop = np.clip((depth_crop - depth_crop.mean()), -1, 1)
#         self.depth_crop = depth_crop.copy()
#         depth_tensor = torch.from_numpy(depth_crop).unsqueeze(0).unsqueeze(0)
        
#         with torch.no_grad():
#             pos_out, ang_out, width_out, qual_out = self.model(depth_tensor)

#         pos_out = pos_out.squeeze().cpu().numpy()   # shape: (300, 300)
#         ang_out = ang_out.squeeze().cpu().numpy()  # shape: (300, 300)
#         width_out = width_out.squeeze().cpu().numpy()  # shape: (300, 300)
#         qual_out = qual_out.squeeze().cpu().numpy()  # shape: (300, 300)

#         # Filtros (simplificados)
#         pos_out_filtered = cv2.GaussianBlur(pos_out, (5, 5), 0)  
#         pos_out_filtered = np.clip(pos_out_filtered, 0.0, 1.0 - 1e-3)
#         ang_out_filtered = cv2.GaussianBlur(ang_out, (5, 5), 0)
#         width_out_filtered = cv2.GaussianBlur(width_out, (5, 5), 0)

#         qual_out_filtered = cv2.GaussianBlur(qual_out, (5, 5), 0)    # Inclui essa parte




#         # Mostrar no loginfo o valor do pixel com maior qualidade  (TESTANDO)
#         # best_y, best_x = np.unravel_index(np.argmax(qual_out_filtered), qual_out_filtered.shape)
#         # rospy.loginfo(f"Pixel com maior qualidade: y={best_y}, x={best_x}")
#         # best_angle = ang_out_filtered[best_y, best_x]
#         # best_width = width_out_filtered[best_y, best_x]
#         # best_qual = qual_out[best_y, best_x]
#         # rospy.loginfo(f"Melhor ângulo: {best_angle:.2f}, Melhor largura: {best_width:.4f}, Qualidade: {best_qual:.4f}")
#         # best_width = float(np.abs(best_width))

        




        
#         # CONTROL PROCESS
#         # Corrigido para tf2
#         try:
#             transform = self.tf_buffer.lookup_transform("base_link", "grasping_link", rospy.Time(0), rospy.Duration(1.0))
#             ROBOT_Z = transform.transform.translation.z
#         except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
#             ROBOT_Z = 0.0 # Valor padrão em caso de erro

#         # Encontra o pixel com a maior pontuação de probabilidade
#         # max_pixel = np.array(np.unravel_index(np.argmax(pos_out_filtered), pos_out_filtered.shape))    # maior pontuação de probabilidade
#         max_pixel = np.array(np.unravel_index(np.argmax(qual_out_filtered), qual_out_filtered.shape))  # maior pontuação de qualidade
#         max_pixel = max_pixel.astype(int)
#         rospy.loginfo(f"Pixel com maior probabilidade: y = {max_pixel[0]}, x = {max_pixel[1]}")
        
#         ang = ang_out_filtered[max_pixel[0], max_pixel[1]]
#         width_px = width_out_filtered[max_pixel[0], max_pixel[1]]
        
#         # Conversão de pixel para metro
#         height_res, width_res = depth.shape
#         reescaled_height = int(max_pixel[0])
#         reescaled_width = int((width_res - self.crop_size) // 2 + max_pixel[1])
#         max_pixel_reescaled = [reescaled_height, reescaled_width]
#         point_depth = depth_copy_for_point_depth[max_pixel_reescaled[0], max_pixel_reescaled[1]]

#         x = (max_pixel_reescaled[1] - self.cx)/(self.fx) * point_depth
#         y = (max_pixel_reescaled[0] - self.cy)/(self.fy) * point_depth
#         grasping_point = [x, y, point_depth]

#         width_m = abs(width_out_filtered[max_pixel[0], max_pixel[1]])
#         width_m = min(width_m, 0.085) # Clamp to max gripper width

#         # OUTPUT
#         self.ang = ang
#         self.width_m = width_m
#         self.grasping_point = grasping_point
#         self.points_out = pos_out_filtered
#         self.ang_out = ang_out_filtered
#         self.width_out = width_out_filtered
#         self.depth_message_ggcnn = depth_message

#     def publish_images(self):
#         if self.points_out is not None:
#             grasp_img = np.zeros((self.crop_size, self.crop_size, 3), dtype=np.uint8)
#             grasp_img[:,:,2] = (self.points_out * 255.0)
#             max_pixel = np.array(np.unravel_index(np.argmax(self.points_out), self.points_out.shape))
#             cv2.circle(grasp_img, (max_pixel[1], max_pixel[0]), 5, (0, 255, 0), -1)

#             grasp_msg = self.bridge.cv2_to_imgmsg(grasp_img, 'bgr8')
#             grasp_msg.header = self.depth_message_ggcnn.header
#             self.grasp_pub.publish(grasp_msg)

#             depth_msg = self.bridge.cv2_to_imgmsg(self.depth_crop)
#             depth_msg.header = self.depth_message_ggcnn.header
#             self.depth_pub.publish(depth_msg)

#     def publish_data_to_robot(self):
#         if not self.grasping_point:
#             return

#         cmd_msg = Float32MultiArray()
#         cmd_msg.data = [self.grasping_point[0], self.grasping_point[1], self.grasping_point[2], self.ang, self.width_m]
#         # Mostrar no loginfo os valores calculados 
#         rospy.loginfo(f"Comando de preensão: x={cmd_msg.data[0]:.4f}, y={cmd_msg.data[1]:.4f}, z={cmd_msg.data[2]:.4f}, ângulo={cmd_msg.data[3]:.2f}, largura={cmd_msg.data[4]:.4f}")
#         # Publicar os valores
#         self.cmd_pub.publish(cmd_msg)
        
#         # Corrigido para tf2
#         grasp_transform = TransformStamped()  # geometry_msgs.msg.TransformStamped
#         grasp_transform.header.stamp = rospy.Time.now() # Sempre use o tempo atual
#         grasp_transform.header.frame_id = "camera_depth_optical_frame" # Frame da câmera (pai)
#         grasp_transform.child_frame_id = "object_detected" # Nome do frame filho que estamos criando
#         grasp_transform.transform.translation.x = cmd_msg.data[0]
#         grasp_transform.transform.translation.y = cmd_msg.data[1]
#         grasp_transform.transform.translation.z = cmd_msg.data[2]
#         q = quaternion_from_euler(0.0, 0.0, cmd_msg.data[3])
#         grasp_transform.transform.rotation.x = q[0]
#         grasp_transform.transform.rotation.y = q[1]
#         grasp_transform.transform.rotation.z = q[2]
#         grasp_transform.transform.rotation.w = q[3]

#         # self.tf_broadcaster.sendTransform(grasp_transform)

#         # Publica continuamente a transformação
#         rate = rospy.Rate(10)
#         while not rospy.is_shutdown():
#             grasp_transform.header.stamp = rospy.Time.now()
#             self.tf_broadcaster.sendTransform(grasp_transform)
#             rate.sleep()


# def main():
#     args = parse_args()
#     ggcnn_detection = ggcnn_grasping(args)
#     rospy.sleep(3.0)

#     input("Press enter to start the GGCNN")
#     rospy.loginfo("Starting process")
    
#     rate = rospy.Rate(10)
#     while not rospy.is_shutdown():
#         ggcnn_detection.depth_process_ggcnn()
#         ggcnn_detection.publish_images()
#         ggcnn_detection.publish_data_to_robot()
#         rate.sleep()

# if __name__ == '__main__':
#     try:
#         main()
#     except rospy.ROSInterruptException:
#         print ("Program interrupted before completion")


