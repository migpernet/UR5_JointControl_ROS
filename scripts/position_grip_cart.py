#!/usr/bin/env python3

import sys
import rospy
import moveit_commander
import geometry_msgs.msg
import moveit_msgs.msg
# from tf.transformations import euler_from_quaternion
import tf.transformations
import garra
import position_grip

def move_ur5_to_pose(val, euler):
    # Inicializa o MoveIt! Commander e o nó ROS
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('ur5_move_to_pose', anonymous=True)

    # Cria um objeto MoveGroupCommander para o braço robótico UR5
    move_group = moveit_commander.MoveGroupCommander("manipulator")

    #Create a DisplayTrajectory ROS publisher which is used to display trajectories in Rviz:
    display_trajectory_publisher = rospy.Publisher(
        "/move_group/display_planned_path",
        moveit_msgs.msg.DisplayTrajectory,
        queue_size=20,
    )
   
    target_pose = geometry_msgs.msg.Pose()
    target_pose.position.x = val[0]   # Posição X (metros)
    target_pose.position.y = val[1]   # Posição Y (metros)
    target_pose.position.z = val[2]   # Posição Z (metros)
    
    # Define a orientação do end-effector usando ângulos de Euler
    
    roll = euler[0] # valor para roll
    pitch = euler[1] # valor para pitch (aproximadamente 90 graus)
    yaw = euler[2] # valor para yaw

    # Converte os ângulos de Euler para um quatérnion
    # A função quaternion_from_euler espera (roll,8 pitch, yaw) em radianos
    quaternion = tf.transformations.quaternion_from_euler(roll, pitch, yaw)    

    # Orientação definida como um quaternion (w,x,y,z)
    target_pose.orientation.x = quaternion[0]
    target_pose.orientation.y = quaternion[1]
    target_pose.orientation.z = quaternion[2]
    target_pose.orientation.w = quaternion[3]  # Define a orientação do end-effector

    # Define o alvo no espaço cartesiano
    move_group.set_pose_target(target_pose)

    # Planeja e executa o movimento
    plan = move_group.go(wait=True)  # Planeja e executa diretamente

    # Libera o alvo após o movimento para evitar que ele permaneça bloqueado
    move_group.stop()
    move_group.clear_pose_targets()

    # Finaliza a comunicação com o MoveIt!
    moveit_commander.roscpp_shutdown()
    rospy.loginfo("Movimento concluído!")

if __name__ == '__main__':
    try:

        # val = [-0.446, 0.109, 0.45]    # posição 
        # euler = [3.140, -0.190, 3.131]  # roll, pitch, yaw

        # val = [-0.446, 0.054, 0.415]    # posição 
        # euler = [3.135, -0.103, 3.130]  # roll, pitch, yaw

        # val = [0.2709, -0.0473, 0.19]    # posição 
        # euler = [0, 0, 0]  # roll, pitch, yaw

        # val = [-0.375, 0.098, 0.350]    # posição 
        # euler = [3.117, 0.140, 3.129]  # roll, pitch, yaw

        # val = [-0.373, 0.098, 0.344]    # posição 
        # euler = [3.116, 0.159, 3.129]  # roll, pitch, yaw

        val = [-0.5071, 0.0417, (0.0164+0.16)]    # posição 
        euler = [3.14, 0, 1.0838]  # roll, pitch, yaw

        move_ur5_to_pose(val, euler)
        # rospy.sleep(0.2)
        # garra.close_gripper()
        # garra.open_gripper()

    except rospy.ROSInterruptException:
        pass


