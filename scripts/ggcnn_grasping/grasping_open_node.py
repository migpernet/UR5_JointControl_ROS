#!/usr/bin/env python3

# ROS
import rospy
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Pose
import tf2_ros
import tf2_geometry_msgs
from control_msgs.msg import GripperCommand
from tf.transformations import quaternion_from_euler

# Bibliotecas de terceiros
import sys
import moveit_commander
import moveit_msgs.msg
from math import pi

# A classe para realizar a preensão
class Grasping(object):
    def __init__(self):
        rospy.init_node('grasping_node', anonymous=True)

        # MoveIt! Commander initialization
        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.move_group = moveit_commander.MoveGroupCommander("manipulator")
        self.gripper_group = moveit_commander.MoveGroupCommander("gripper")
        self.move_group.set_planner_id("RRTConnectkConfigDefault")
        self.move_group.set_pose_reference_frame('base_link')

        # Publisher for the gripper
        self.gripper_pub = rospy.Publisher('/robotiq_2f_85_controller/gripper_cmd', GripperCommand, queue_size=10)
        rospy.sleep(2.0)
        
        # TF2
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Variables
        self.grasp_data = None
        self.pre_grasp_z_offset = 0.2  # Distância (em metros) acima do objeto para pré-preensão
        self.drop_off_pose = Pose() # Posição de descarte do objeto
        self.drop_off_pose.position.x = 0.4
        self.drop_off_pose.position.y = 0.4
        self.drop_off_pose.position.z = 0.5
        
        self.is_holding_object = False

        # Subscriber for GGCNN data
        rospy.Subscriber('ggcnn/out/command_grasp', Float32MultiArray, self.grasp_callback)
        rospy.loginfo("Nó de preensão inicializado e esperando por dados do GGCNN...")
        
    def grasp_callback(self, msg):
        """
        Callback para processar a mensagem do ggcnn com os dados da preensão.
        A mensagem Float32MultiArray deve conter: [x, y, z, angulo, largura_m, largura_g]
        """
        if self.is_holding_object:
            rospy.loginfo("Já estou segurando um objeto. Ignorando nova preensão.")
            return

        self.grasp_data = msg.data
        rospy.loginfo("Dados de preensão recebidos.")
        self.execute_grasp()
        
    def move_to_named_pose(self, pose_name):
        """Move o robô para uma pose pré-definida."""
        rospy.loginfo(f"Movendo para a pose: {pose_name}")
        self.move_group.set_named_target(pose_name)
        plan = self.move_group.plan()
        if plan[0]:
            self.move_group.execute(plan[1], wait=True)
            self.move_group.stop()
            self.move_group.clear_pose_targets()
        else:
            rospy.logerr("Falha ao planejar o caminho para a pose: " + pose_name)

    def gripper_control(self, position, max_effort=10.0):
        """
        Controla a garra.
        position: 0.0 (fechada) a 0.8 (aberta)
        """
        rospy.loginfo(f"Movendo a garra para a posição: {position}")
        cmd = GripperCommand()
        cmd.position = position
        cmd.max_effort = max_effort
        self.gripper_pub.publish(cmd)
        rospy.sleep(1.0) # Wait for the gripper to move

    def execute_grasp(self):
        """
        Sequência de preensão em malha aberta.
        """
        if self.grasp_data is None:
            rospy.loginfo("Nenhum dado de preensão disponível.")
            return

        # 1. Mover para a pose inicial "Home"
        self.move_to_named_pose("home")
        
        # 2. Abrir a garra para a largura detectada
        # The value `self.grasp_data[4]` is the width_m, which should be adjusted for the gripper.
        self.gripper_control(self.grasp_data[4] + 0.01)

        # 3. Transformar a pose de preensão para o frame do robô
        rospy.loginfo("Obtendo pose de preensão...")
        x, y, z, angulo, largura_m, largura_g = self.grasp_data
        
        # Create a Pose for the object in the camera frame
        camera_pose = Pose()
        camera_pose.position.x = x
        camera_pose.position.y = y
        camera_pose.position.z = z
        # The orientation of the gripper needs to be defined
        # For a vertical gripper approach, roll=pi, pitch=0, yaw=angulo
        q = quaternion_from_euler(pi, 0, angulo)
        camera_pose.orientation.x = q[0]
        camera_pose.orientation.y = q[1]
        camera_pose.orientation.z = q[2]
        camera_pose.orientation.w = q[3]

        grasp_pose_base_link = Pose()
        try:
            # Transform the pose from the camera frame to the robot's base frame
            transform = self.tf_buffer.lookup_transform('base_link', 'camera_depth_optical_frame', rospy.Time(0), rospy.Duration(1.0))
            camera_pose_stamped = tf2_geometry_msgs.PoseStamped()
            camera_pose_stamped.pose = camera_pose
            camera_pose_stamped.header.frame_id = 'camera_depth_optical_frame'
            camera_pose_stamped.header.stamp = rospy.Time.now()
            
            transformed_pose_stamped = tf2_geometry_msgs.do_transform_pose(camera_pose_stamped, transform)
            grasp_pose_base_link = transformed_pose_stamped.pose
            
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as ex:
            rospy.logerr(f"Erro ao buscar transformação: {ex}")
            return
        
        # 4. Criar a pose de pré-preensão (um pouco acima da pose de preensão)
        pre_grasp_pose = Pose()
        pre_grasp_pose.position.x = grasp_pose_base_link.position.x
        pre_grasp_pose.position.y = grasp_pose_base_link.position.y
        pre_grasp_pose.position.z = grasp_pose_base_link.position.z + self.pre_grasp_z_offset
        pre_grasp_pose.orientation = grasp_pose_base_link.orientation
        
        rospy.loginfo("Movendo para a pose de pré-preensão...")
        self.move_group.set_pose_target(pre_grasp_pose)
        plan = self.move_group.plan()
        if plan[0]:
            self.move_group.execute(plan[1], wait=True)
        else:
            rospy.logerr("Falha ao planejar o caminho para a pré-preensão.")
            return

        # 5. Mover para a pose de preensão
        rospy.loginfo("Movendo para a pose de preensão final...")
        self.move_group.set_pose_target(grasp_pose_base_link)
        plan = self.move_group.plan()
        if plan[0]:
            self.move_group.execute(plan[1], wait=True)
        else:
            rospy.logerr("Falha ao planejar o caminho para a preensão.")
            return
        
        # 6. Fechar a garra
        rospy.loginfo("Fechando a garra para prender o objeto...")
        self.gripper_control(0.0)
        self.is_holding_object = True

        # 7. Mover para cima com o objeto
        rospy.loginfo("Subindo com o objeto...")
        post_grasp_pose = Pose()
        post_grasp_pose.position.x = grasp_pose_base_link.position.x
        post_grasp_pose.position.y = grasp_pose_base_link.position.y
        post_grasp_pose.position.z = grasp_pose_base_link.position.z + self.pre_grasp_z_offset
        post_grasp_pose.orientation = grasp_pose_base_link.orientation
        self.move_group.set_pose_target(post_grasp_pose)
        plan = self.move_group.plan()
        if plan[0]:
            self.move_group.execute(plan[1], wait=True)
        else:
            rospy.logerr("Falha ao planejar o caminho para levantar o objeto.")
            return

        # 8. Mover para a pose de descarte
        rospy.loginfo("Movendo para a pose de descarte...")
        self.move_group.set_pose_target(self.drop_off_pose)
        plan = self.move_group.plan()
        if plan[0]:
            self.move_group.execute(plan[1], wait=True)
        else:
            rospy.logerr("Falha ao planejar o caminho para o descarte.")
            return
        
        # 9. Soltar o objeto
        rospy.loginfo("Soltando o objeto...")
        self.gripper_control(0.08) # Open the gripper to release

        # 10. Mover para a pose inicial "Home" e resetar a garra
        self.is_holding_object = False
        self.move_to_named_pose("home")
        self.gripper_control(0.08)

def main():
    try:
        grasp_action = Grasping()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        moveit_commander.roscpp_shutdown()
        moveit_commander.os._exit(0)

if __name__ == '__main__':
    main()