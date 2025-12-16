#!/usr/bin/env python3

import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

def move_to_position(positions_goal):
    rospy.init_node('ur5_move_to_pose', anonymous=True)
    pub = rospy.Publisher('/ur5/eff_joint_traj_controller/command', JointTrajectory, queue_size=10)

    rospy.sleep(1)  # Aguarde o Gazebo e os plugins inicializarem.

    trajectory = JointTrajectory()
    trajectory.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 
                              'elbow_joint', 'wrist_1_joint', 
                              'wrist_2_joint', 'wrist_3_joint']

    point = JointTrajectoryPoint()
                 
    # point.positions = [2.155043203566265, 5.683457372285261e-06, -1.619507999223913, 2.875849523968503, -2.0853028334266472, 1.571269556895201, -4.446531500660392]
    point.positions = positions_goal  # Position initial.
    point.time_from_start = rospy.Duration(3.0)  # Tempo para atingir a posição.

    trajectory.points.append(point)
    pub.publish(trajectory)

if __name__ == '__main__':
    try:
        positions_goal = [0, -1.57, 0, -1.57, 0, 1.57]  #position initial.
        move_to_position(positions_goal)
    except rospy.ROSInterruptException:
        pass