#!/usr/bin/env python
import sys
import rospy
from sensor_msgs.msg import JointState
import moveit_commander
import moveit_msgs.msg
import copy

def joint_cb(msg):
    rospy.loginfo("unity_to_moveit_bridge: received JointState with %d joints", len(msg.name))

    # Obter os nomes das juntas do grupo ativo
    group_joints = group.get_active_joints()  # lista de nomes na ordem do MoveIt
    rospy.loginfo("MoveIt active joints: %s", str(group_joints))

    # Montar lista de posições na ordem do MoveIt
    target_positions = []
    current_positions = group.get_current_joint_values()

    for j in group_joints:
        if j in msg.name:
            idx = msg.name.index(j)
            if idx < len(msg.position):
                target_positions.append(msg.position[idx])
            else:
                rospy.logwarn("Índice de posição fora do alcance para %s", j)
                target_positions.append(current_positions[len(target_positions)])
        else:
            # fallback: manter posição atual
            rospy.logwarn("Joint %s não encontrada na mensagem da Unity. Mantendo posição atual.", j)
            target_positions.append(current_positions[len(target_positions)])

    rospy.loginfo("Target positions (rad): %s", str(target_positions))

    # Define o target e solicita planejamento + execução
    group.set_joint_value_target(target_positions)

    # O método plan() retorna uma tupla. É preciso desempacotar.
    success, plan, planning_time, error_code = group.plan()

    if success and len(plan.joint_trajectory.points) > 0:
        rospy.loginfo("Plano gerado. Executando...")
        group.go(wait=True)
        rospy.loginfo("Execução concluída.")
    else:
        rospy.logwarn("Falha ao gerar plano para o target recebido.")

if __name__ == "__main__":
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("unity_to_moveit_bridge", anonymous=True)

    # Nome do move group
    group_name = "manipulator"
    group = moveit_commander.MoveGroupCommander(group_name)

    # Subscribe
    rospy.Subscriber("/unity/goal_joint_states", JointState, joint_cb)

    rospy.loginfo("unity_to_moveit_bridge node started, listening to /unity/goal_joint_states")
    rospy.spin()
    moveit_commander.roscpp_shutdown()
