#!/usr/bin/env python3

import rospy
import moveit_commander
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from tf.transformations import quaternion_from_euler
import tf2_ros
import tf2_geometry_msgs  # Importe este módulo
import numpy as np

class GraspExecutionNode:
    def __init__(self):
        rospy.init_node('grasp_execution_node', anonymous=True)
        
        # Variáveis para armazenar as mensagens recebidas
        self.grasp_pose = None
        self.grasp_width = None
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Configurar MoveIt!
        moveit_commander.roscpp_initialize([])
        self.robot = moveit_commander.RobotCommander() 
        self.scene = moveit_commander.PlanningSceneInterface()
        self.arm_group = moveit_commander.MoveGroupCommander("manipulator")
        
        self.gripper_pub = rospy.Publisher('/ur5/gripper_controller/command', JointTrajectory, queue_size=1)
        
        rospy.Subscriber('/grasp/pose', PoseStamped, self.pose_callback)
        rospy.Subscriber('/grasp/width', Float32, self.width_callback)
        
        rospy.loginfo("Nó de execução de preensão inicializado e assinando tópicos.")
        rospy.spin()

    def pose_callback(self, msg):
        self.grasp_pose = msg
        self.execute_if_ready()
        
    def width_callback(self, msg):
        self.grasp_width = msg
        self.execute_if_ready()

    def execute_if_ready(self):
        # Executa a preensão somente se ambos os tópicos tiverem sido recebidos
        if self.grasp_pose and self.grasp_width:
            rospy.loginfo("Ambos os parâmetros de preensão recebidos. Executando...")
            
            # # CORREÇÃO: Converter a pose da câmera para o frame da base do robô
            # try:
            #     transform = self.tf_buffer.lookup_transform('base_link', 'camera_link', rospy.Time(0), rospy.Duration(1.0))
            #     grasp_point_robot = tf2_geometry_msgs.do_transform_pose(self.grasp_pose, transform)
            # except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            #     rospy.logerr("Erro na transformação TF: %s", str(e))
            #     return
            grasp_point_robot = self.grasp_pose
            self.execute_grasp(grasp_point_robot, self.grasp_width.data)
            
            # Resetar as variáveis para esperar a próxima preensão
            self.grasp_pose = None
            self.grasp_width = None

    def execute_grasp(self, grasp_pose, width):
        rospy.loginfo(f"Grasp point recebido: x={grasp_pose.pose.position.x:.4f}, y={grasp_pose.pose.position.y:.4f}, z={grasp_pose.pose.position.z:.4f}")

        # Criar a orientação da garra a partir da pose
        quaternion = quaternion_from_euler(0, np.pi, 0) # Assumimos uma orientação padrão, ou podemos obtê-la da pose
        grasp_pose.pose.orientation.x = quaternion[0]
        grasp_pose.pose.orientation.y = quaternion[1]
        grasp_pose.pose.orientation.z = quaternion[2]
        grasp_pose.pose.orientation.w = quaternion[3]
        
        self.arm_group.set_pose_target(grasp_pose.pose)
        
        rospy.loginfo("Planejando a preensão...")
        plan = self.arm_group.plan()
        
        if plan[0]:
            rospy.loginfo("Plano de preensão encontrado. Executando...")
            self.arm_group.execute(plan[1], wait=True)
            rospy.sleep(1.0)
            
            gripper_msg = JointTrajectory()
            gripper_msg.joint_names = ['robotiq_85_left_knuckle_joint']
            point = JointTrajectoryPoint()
            point.positions = [width/2.0, width/2.0]
            point.time_from_start = rospy.Duration(1.0)
            gripper_msg.points.append(point)
            self.gripper_pub.publish(gripper_msg)
            
            rospy.sleep(1.0)
            
            grasp_pose.pose.position.z += 0.1
            self.arm_group.set_pose_target(grasp_pose.pose)
            self.arm_group.go(wait=True)
        else:
            rospy.logwarn("Não foi possível encontrar um plano de preensão.")
        
        self.arm_group.stop()
        self.arm_group.clear_pose_targets()
        
if __name__ == '__main__':
    try:
        GraspExecutionNode()
    except rospy.ROSInterruptException:
        pass