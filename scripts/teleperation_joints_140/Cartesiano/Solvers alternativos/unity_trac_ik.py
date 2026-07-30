#!/usr/bin/env python3
import rospy
import math
from geometry_msgs.msg import PoseStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from trac_ik_python.trac_ik import IK

rospy.init_node('unity_trac_ik_solver')

# 1. SOLVER SPEED: Muito melhor para escapar de singularidades que o Distance
ik_solver = IK("base_link", "tool0", solve_type="Speed", timeout=0.05)
joint_names = ik_solver.joint_names

current_joint_state = [0.0] * ik_solver.number_of_joints
has_received_joints = False

traj_pub = rospy.Publisher('/ur5/arm_controller/command', JointTrajectory, queue_size=1)

def joint_callback(msg):
    global current_joint_state, has_received_joints
    temp_state = [0.0] * len(joint_names)
    try:
        for i, name in enumerate(joint_names):
            idx = msg.name.index(name)
            temp_state[i] = msg.position[idx]
        current_joint_state = temp_state
        has_received_joints = True
    except ValueError:
        pass

def pose_callback(msg):
    global current_joint_state, has_received_joints
    
    if not has_received_joints:
        return

    x = msg.pose.position.x
    y = msg.pose.position.y
    z = msg.pose.position.z
    
    rx = msg.pose.orientation.x
    ry = msg.pose.orientation.y
    rz = msg.pose.orientation.z
    rw = msg.pose.orientation.w

    norm = math.sqrt(rx**2 + ry**2 + rz**2 + rw**2)
    if norm == 0:
        return
    rx /= norm; ry /= norm; rz /= norm; rw /= norm

    sol = None

    # --- CASCATA DE RESGATE DO SOLVER ---
    
    # Tentativa 1: Busca exata (Precisão máxima)
    sol = ik_solver.get_ik(current_joint_state, x, y, z, rx, ry, rz, rw,
                           bx=0.001, by=0.001, bz=0.001, brx=0.01, bry=0.01, brz=0.01)

    # Tentativa 2: O Milímetro Fantasma (Relaxa a tolerância espacial em 2cm se estiver fora do limite)
    if not sol:
        sol = ik_solver.get_ik(current_joint_state, x, y, z, rx, ry, rz, rw,
                               bx=0.02, by=0.02, bz=0.02, brx=0.1, bry=0.1, brz=0.1)

    # Tentativa 3: Preso na Singularidade (Usa uma pose "Home" dobrada como ponto de partida)
    if not sol:
        safe_seed = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]
        sol = ik_solver.get_ik(safe_seed, x, y, z, rx, ry, rz, rw,
                               bx=0.02, by=0.02, bz=0.02, brx=0.1, bry=0.1, brz=0.1)

    # Se achou uma saída, manda mover!
    if sol:
        traj_msg = JointTrajectory()
        traj_msg.joint_names = joint_names
        
        point = JointTrajectoryPoint()
        point.positions = sol
        point.time_from_start = rospy.Duration(0.1) 
        
        traj_msg.points.append(point)
        traj_pub.publish(traj_msg)
    else:
        rospy.logwarn_throttle(2.0, f"TRAC-IK Rejeitou (Fora da área de alcance total). X:{x:.2f} Y:{y:.2f} Z:{z:.2f}")

rospy.Subscriber('/ur5/joint_states', JointState, joint_callback)
rospy.Subscriber('unity/target_pose', PoseStamped, pose_callback)

rospy.loginfo("TRAC-IK Solver com Cascata de Resgate Iniciado!")
rospy.spin()