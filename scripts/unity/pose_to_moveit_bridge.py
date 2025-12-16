#!/usr/bin/env python
import sys
import rospy
import moveit_commander
from geometry_msgs.msg import PoseStamped
from tf.transformations import quaternion_from_euler
from moveit_msgs.msg import RobotTrajectory, DisplayTrajectory

def pose_callback(msg):
    rospy.loginfo("Recebido nova pose do Unity:")
    rospy.loginfo("Posição: x=%.2f, y=%.2f, z=%.2f", msg.pose.position.x, msg.pose.position.y, msg.pose.position.z)
    rospy.loginfo("Orientação (quat): x=%.2f, y=%.2f, z=%.2f, w=%.2f", msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w)

    # Configura a pose de destino para o MoveIt
    try:
        pose_goal = PoseStamped()
        pose_goal.header.frame_id = "base_link"
        pose_goal.header.stamp = rospy.Time.now()

        pose_goal.pose = msg.pose

        group.set_pose_target(pose_goal)
        rospy.loginfo("Iniciando o planejamento para a nova pose...")

        # Planeja e executa
        success, plan, planning_time, error_code = group.plan()

        if success:
            rospy.loginfo("Plano gerado com sucesso. Executando...")
            group.go(wait=True)
            rospy.loginfo("Execução concluída.")
            
            # **PARTE CORRIGIDA:** Cria e publica a mensagem DisplayTrajectory
            display_trajectory = DisplayTrajectory()
            display_trajectory.trajectory_start = group.get_current_state()
            display_trajectory.trajectory.append(plan)
            
            display_trajectory_publisher.publish(display_trajectory)
        else:
            rospy.logwarn("Falha ao gerar plano para a pose recebida.")

    except Exception as e:
        rospy.logerr("Erro durante o planejamento/execução: %s", str(e))

if __name__ == '__main__':
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('unity_pose_bridge', anonymous=True)

    # Conecta-se ao grupo de planejamento do MoveIt
    group_name = "manipulator"
    group = moveit_commander.MoveGroupCommander(group_name)

    # Publisher para enviar a trajetória de volta ao Unity
    display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path', DisplayTrajectory, queue_size=20)

    # Subscreve no tópico que o Unity está publicando
    rospy.Subscriber("/ur5/goal_pose", PoseStamped, pose_callback)

    rospy.loginfo("Node 'unity_pose_bridge' started, listening for poses on /ur5/goal_pose")
    rospy.spin()

    moveit_commander.roscpp_shutdown()
