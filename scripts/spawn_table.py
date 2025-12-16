#!/usr/bin/env python3
import rospy
import tf
import rospkg
from gazebo_msgs.srv import SpawnModel
import time
from geometry_msgs.msg import Pose, Point, Quaternion
from tf.transformations import quaternion_from_euler
import os

class SpawnModelHelper:
    """
    Uma classe auxiliar para carregar e lançar modelos no Gazebo.
    """
    def __init__(self, model_name, path):
        self.model_name = model_name
        self.path = path
        # Inicializa o serviço do Gazebo para lançar modelos SDF
        self.spawn_model_client = rospy.ServiceProxy("gazebo/spawn_sdf_model", SpawnModel)
        rospy.wait_for_service("gazebo/spawn_sdf_model")
        rospy.loginfo("Serviço 'gazebo/spawn_sdf_model' está pronto.")
        
    def spawn(self, x_pose, y_pose, z_pose, roll, pitch, yaw):
        """
        Lança o modelo no Gazebo com uma pose específica.
        """
        rospy.loginfo(f"Tentando lançar o modelo: {self.model_name}")

        # Converte a orientação de Euler para um Quaternio
        orientation_q = quaternion_from_euler(roll, pitch, yaw)
        
        # Define a pose do modelo
        model_pose = Pose(
            position=Point(x=x_pose, y=y_pose, z=z_pose),
            orientation=Quaternion(*orientation_q)
        )
        
        # Lê o conteúdo do arquivo SDF
        try:
            with open(self.path, 'r') as f:
                product_xml = f.read()
        except FileNotFoundError:
            rospy.logerr(f"Arquivo SDF não encontrado: {self.path}")
            return

        # Chama o serviço para lançar o modelo
        try:
            self.spawn_model_client(self.model_name, product_xml, "", model_pose, "world")
            rospy.loginfo(f"Modelo '{self.model_name}' lançado com sucesso.")
        except rospy.ServiceException as e:
            rospy.logerr(f"Falha ao chamar o serviço de spawn: {e}")

def spawn_table():
    """
    Função principal para lançar a mesa.
    """
    # Inicializa o nó ROS
    rospy.init_node('spawn_table_node', anonymous=True)
    
    # Obtém o caminho do pacote ggcnn_pkg
    rospack = rospkg.RosPack()
    try:
        package_path = rospack.get_path('ggcnn_pkg')
    except rospkg.ResourceNotFound as e:
        rospy.logerr(f"Pacote 'ggcnn_pkg' não encontrado: {e}")
        return

    # Define o caminho completo para o arquivo model.sdf
    table_sdf_path = os.path.join(package_path, 'models', 'lar_table_great', 'model.sdf')
    rospy.loginfo(f"Caminho do arquivo SDF: {table_sdf_path}")

    # Posição e orientação da mesa
    # Ajuste estes valores para posicionar a mesa onde você precisa
    x = 0.0  # Posição X em metros
    y = -0.4  # Posição Y em metros
    z = 0.0  # Posição Z em metros (geralmente 0 para modelos no chão)
    roll = 0.0
    pitch = 0.0
    yaw = 1.57 # Ajuste para girar a mesa, se necessário (ex: 1.57 para 90 graus)
    
    # Cria uma instância da classe SpawnModelHelper e lança a mesa
    spawner = SpawnModelHelper("lar_table_great", table_sdf_path)
    spawner.spawn(x, y, z, roll, pitch, yaw)

    rospy.spin()

if __name__ == '__main__':
    try:
        spawn_table()
    except rospy.ROSInterruptException:
        pass