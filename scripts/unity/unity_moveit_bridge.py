#!/usr/bin/env python
import sys
import rospy
import moveit_commander
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState

class UnityMoveItBridge:
    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('unity_moveit_bridge', anonymous=True)

        # Use o nome do planning group do MoveIt para o UR5 (muitas configs usam "manipulator")
        self.group = moveit_commander.MoveGroupCommander("manipulator")

        # Publisher que o Unity vai escutar (pode ser /joint_states)
        self.joint_pub = rospy.Publisher('/joint_states', JointState, queue_size=10)

        rospy.Subscriber('pos_rot', PoseStamped, self.pose_cb, queue_size=1)
        rospy.loginfo("unity_moveit_bridge pronto - aguardando poses em pos_rot")

    def pose_cb(self, msg):
        rospy.loginfo("Recebida pose do Unity. Pedindo plano ao MoveIt...")
        self.group.clear_pose_targets()
        self.group.set_pose_target(msg.pose)

        plan = self.group.plan()  # retorna um RobotTrajectory (varia por versão)
        # Compatibilidade: algumas versões retornam (success, plan). Tentamos detectar:
        traj = None
        if isinstance(plan, tuple) and len(plan) == 2:
            # ex.: (success_flag, plan)
            traj = plan[1]
        else:
            traj = plan

        # Se não tiver trajetória, avisar
        if traj is None or not hasattr(traj, 'joint_trajectory') or len(traj.joint_trajectory.points) == 0:
            rospy.logwarn("Nenhum plano gerado pelo MoveIt (inviável ou fora do alcance).")
            return

        joint_names = traj.joint_trajectory.joint_names
        points = traj.joint_trajectory.points

        rospy.loginfo("Publicando %d pontos de trajetória para /joint_states", len(points))

        # Publica cada ponto com o tempo correto (tempo relativo em time_from_start)
        last_t = 0.0
        start = rospy.Time.now()
        for p in points:
            t = p.time_from_start.to_sec()
            dt = t - last_t
            if dt > 0:
                rospy.sleep(dt)
            js = JointState()
            js.header.stamp = rospy.Time.now()
            js.name = joint_names
            js.position = list(p.positions)
            # (opcional) preencher velocities/effort se quiser
            self.joint_pub.publish(js)
            last_t = t

        rospy.loginfo("Trajetória enviada ao Unity.")

if __name__ == '__main__':
    try:
        bridge = UnityMoveItBridge()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        moveit_commander.roscpp_shutdown()

