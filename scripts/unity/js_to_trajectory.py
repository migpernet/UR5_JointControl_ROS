#!/usr/bin/env python
import rospy
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# Nome do tópico do controller no seu sistema (ajuste conforme o seu controller)
TRAJ_TOPIC = '/ur5/eff_joint_traj_controller/command'

# lista de nomes do UR5 (ajuste se seus nomes forem diferentes)
UR5_JOINTS = [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint'
]

pub = None

def js_cb(msg):
    # cria traj com as posições recebidas (mapeia nomes)
    pos_map = {n: p for n, p in zip(msg.name, msg.position)}
    traj = JointTrajectory()
    traj.joint_names = UR5_JOINTS
    point = JointTrajectoryPoint()
    # preenche posições na ordem esperada; usa valor atual se não achou
    positions = [pos_map.get(n, 0.0) for n in UR5_JOINTS]
    point.positions = positions
    # pequeno tempo para execução (ajuste entre 0.1 e 1.0 para suavidade)
    point.time_from_start = rospy.Duration(0.5)
    traj.points = [point]
    pub.publish(traj)

def main():
    global pub
    rospy.init_node('js_to_trajectory')
    pub = rospy.Publisher(TRAJ_TOPIC, JointTrajectory, queue_size=1)
    rospy.Subscriber('/joint_states', JointState, js_cb)
    rospy.loginfo("js_to_trajectory started, forwarding /joint_states -> %s", TRAJ_TOPIC)
    rospy.spin()

if __name__ == '__main__':
    main()

