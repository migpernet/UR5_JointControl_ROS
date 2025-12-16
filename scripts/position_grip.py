#!/usr/bin/env python3

import sys
import rospy
import moveit_commander
import geometry_msgs.msg
import moveit_msgs.msg

def move_ur5_to_pose(position_joint):
    print(position_joint)
    # Inicializa o MoveIt! Commander e o nó ROS
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('ur5_move_to_pose_1', anonymous=True)

    # Cria um objeto MoveGroupCommander para o braço robótico UR5
    move_group = moveit_commander.MoveGroupCommander("manipulator")

    #Create a DisplayTrajectory ROS publisher which is used to display trajectories in Rviz:
    display_trajectory_publisher = rospy.Publisher(
        "/move_group/display_planned_path",
        moveit_msgs.msg.DisplayTrajectory,
        queue_size=20,
    )

    # We get the joint values from the group and change some of the values:
    joint_goal = move_group.get_current_joint_values()
    joint_goal[0] = position_joint[2]
    joint_goal[1] = position_joint[1]
    joint_goal[2] = position_joint[0]
    joint_goal[3] = position_joint[3]
    joint_goal[4] = position_joint[4]
    joint_goal[5] = position_joint[5]

    move_group.go(joint_goal, wait=True)
    move_group.stop()

    # Finaliza a comunicação com o MoveIt!
    moveit_commander.roscpp_shutdown()
    rospy.loginfo("Movimento concluído!")

if __name__ == '__main__':
    try:
        # position_joint =  [-1.0054704413213962, -1.5801199410581797, 0.00089008570573057, -1.9367237375089363, 1.5699304075994247, -1.5595961735668054]   # posição 1 de preensão
        # position_joint = [-1.83, -1.453, 0.00, -1.26, 1.57, -1.56] # posição 2 de preensão
        # position_joint = [-1.76, -1.43, 0.13, -1.41, 1.55, -1.43]
        # position_joint = [-1.954, -1.337, 0.033, -1.563, 1.552, -1.526]
        # position_joint = [-1.976, -1.34, 0.033, -1.572, 1.552, 1.559]
        position_joint = [-2.00, -1.298, 0.00, -1.510, 1.57, 1.627]

        move_ur5_to_pose(position_joint)
    except rospy.ROSInterruptException:
        pass





















# #!/usr/bin/env python3
# # Python 2/3 compatibility imports
# from __future__ import print_function 
# # from six.moves import input 

# import sys
# import copy
# import rospy
# import moveit_commander
# import moveit_msgs.msg
# import geometry_msgs.msg

# try:
#     from math import pi, tau, dist, fabs, cos
# except:  # For Python 2 compatibility
#     from math import pi, fabs, cos, sqrt

#     tau = 2.0 * pi

#     def dist(p, q):
#         return sqrt(sum((p_i - q_i) ** 2.0 for p_i, q_i in zip(p, q)))


# from std_msgs.msg import String 
# from moveit_commander.conversions import pose_to_list 

# moveit_commander.roscpp_initialize(sys.argv)
# rospy.init_node("move_group_python_interface_tutorial", anonymous=True)

# robot = moveit_commander.RobotCommander() # Objeto RobotCommander para acessar informações sobre o robô
# scene = moveit_commander.PlanningSceneInterface() # Objeto PlanningSceneInterface para interagir com a cena de planejamento
# group_name = "manipulator"  # Nome do grupo de planejamento do UR5
# move_group = moveit_commander.MoveGroupCommander(group_name)  # Objeto MoveGroupCommander para o braço robótico UR5

# display_trajectory_publisher = rospy.Publisher(
#     "/move_group/display_planned_path",
#     moveit_msgs.msg.DisplayTrajectory,
#     queue_size=20,
# )

# # We can get the name of the reference frame for this robot:
# planning_frame = move_group.get_planning_frame()
# print("============ Planning frame: %s" % planning_frame)

# # We can also print the name of the end-effector link for this group:
# eef_link = move_group.get_end_effector_link()
# print("============ End effector link: %s" % eef_link)

# # We can get a list of all the groups in the robot:
# group_names = robot.get_group_names()
# print("============ Available Planning Groups:", robot.get_group_names())


# # We get the joint values from the group and change some of the values:
# position = [-0.9863883901350183, -1.5762144678802201, 0.0006755823210884415, -1.9275579209404494, 1.7701385269807446, -1.5596449591229034]
# joint_goal = move_group.get_current_joint_values()
# joint_goal[0] = position[0]
# joint_goal[1] = position[1]
# joint_goal[2] = position[2]
# joint_goal[3] = position[3]
# joint_goal[4] = position[4]
# joint_goal[5] = position[5]
# # joint_goal[6] = 0

# move_group.go(joint_goal, wait=True)
# move_group.stop()



# # # Adiciona um pequeno delay para garantir que o MoveIt! tenha tempo de inicializar
# # rospy.sleep(1.0)

# pose_goal = geometry_msgs.msg.Pose()
# # orientation in Quaternion
# qx, qy, qz, qw = [0.001, 0.996, -0.055, -0.077]
# pose_goal.orientation.w = 1
# pose_goal.orientation.x = qx
# pose_goal.orientation.y = qy
# pose_goal.orientation.z = qz

# # Set the position of the end-effector in the world frame
# # px, py, pz = [-0.445, 0.099, 0.656]
# px, py, pz = [-0.4, 0.1, 0.4]
# pose_goal.position.x = px
# pose_goal.position.y = py
# pose_goal.position.z = pz


# # pose_goal = geometry_msgs.msg.Pose()
# # pose_goal.orientation.w = 1
# # pose_goal.position.x = 0.4
# # pose_goal.position.y = 0.1
# # pose_goal.position.z = 0.4



# move_group.set_pose_target(pose_goal)

# success = move_group.go(wait=True)
# move_group.stop()
# move_group.clear_pose_targets()













# #!/usr/bin/env python3

# import sys
# import rospy
# import moveit_commander

# def get_joint_angles():
#     # Inicializa o MoveIt! Commander e o nó ROS
#     moveit_commander.roscpp_initialize(sys.argv)
#     rospy.init_node('ur5_get_joint_angles', anonymous=True)

#     # Criar um objeto MoveGroupCommander para o braço do UR5
#     move_group = moveit_commander.MoveGroupCommander("manipulator")

#     # Obtém os valores atuais das juntas
#     joint_angles = move_group.get_current_joint_values()

#     # Exibe os ângulos das juntas no terminal
#     rospy.loginfo("Ângulos das Juntas do UR5 (em radianos):")
#     for i, angle in enumerate(joint_angles):
#         rospy.loginfo(f"Junta {i + 1}: {angle:.4f} rad")

#     # Finaliza o MoveIt!
#     moveit_commander.roscpp_shutdown()

# if __name__ == '__main__':
#     try:
#         get_joint_angles()
#     except rospy.ROSInterruptException:
#         pass





# #!/usr/bin/env python3

# import sys
# import rospy
# import moveit_commander
# # Não precisamos de geometry_msgs.msg ou moveit_msgs.msg para apenas ler as juntas

# def read_ur5_joint_states():
#     # Inicializa o MoveIt! Commander e o nó ROS
#     moveit_commander.roscpp_initialize(sys.argv)
#     rospy.init_node('ur5_read_joint_states', anonymous=True)

#     # Cria um objeto MoveGroupCommander para o braço robótico UR5
#     # O nome "manipulator" deve corresponder ao seu grupo de planejamento para o braço
#     move_group = moveit_commander.MoveGroupCommander("manipulator")

#     # Adicione um pequeno atraso para garantir que o MoveIt! tenha tempo de se inicializar
#     # e subscrever aos tópicos de estado do robô. Isso é crucial!
#     rospy.sleep(2.0)

#     rospy.loginfo("============ Juntas ativas do grupo 'manipulator': %s" % move_group.get_active_joints())
#     rospy.loginfo("============ Nomes das juntas do grupo 'manipulator': %s" % move_group.get_joints())

#     try:
#         # Obtém os valores de junta atuais do robô (conforme percebidos pelo MoveIt!)
#         current_joint_values = move_group.get_current_joint_values()

#         rospy.loginfo("============ Estado atual das juntas do UR5 (via MoveIt!):")
#         # Itera sobre os nomes e valores das juntas para uma saída mais legível
#         joint_names = move_group.get_active_joints() # Ou get_joints() para todas as juntas conhecidas
#         for i, name in enumerate(joint_names):
#             if i < len(current_joint_values): # Garante que não há IndexError
#                 rospy.loginfo(f"  - {name}: {current_joint_values[i]:.4f} rad") # Formata para 4 casas decimais
#             else:
#                 rospy.logwarn(f"  - Aviso: Nenhuma valor para a junta '{name}' nos dados atuais.")

#         rospy.loginfo("============ Leitura de estado concluída.")

#     except Exception as e:
#         rospy.logerr(f"Erro ao tentar ler o estado das juntas: {e}")

#     # Finaliza a comunicação com o MoveIt!
#     moveit_commander.roscpp_shutdown()
#     rospy.loginfo("MoveIt! Commander desligado.")

# if __name__ == '__main__':
#     try:
#         read_ur5_joint_states()
#     except rospy.ROSInterruptException:
#         rospy.loginfo("Nó ROS interrompido.")
#     except Exception as e:
#         rospy.logerr(f"Ocorreu um erro geral: {e}")





# #!/usr/bin/env python3

# import sys
# import rospy
# import moveit_commander
# import geometry_msgs.msg
# import moveit_msgs.msg

# def move_ur5_to_pose():
#     # Inicializa o MoveIt! Commander e o nó ROS
#     # É importante chamar roscpp_initialize antes de qualquer outra coisa do MoveIt!
#     moveit_commander.roscpp_initialize(sys.argv)
#     rospy.init_node('ur5_move_to_pose', anonymous=True)

#     # Cria um objeto RobotCommander para acessar informações sobre o robô completo
#     robot = moveit_commander.RobotCommander()

#     # Cria um objeto PlanningSceneInterface para interagir com a cena de planejamento
#     scene = moveit_commander.PlanningSceneInterface()

#     # Cria um objeto MoveGroupCommander para o braço robótico UR5
#     # O nome "manipulator" é o grupo de planejamento padrão para o braço do UR5
#     move_group = moveit_commander.MoveGroupCommander("manipulator")

#     # Opcional: Configura o planejador a ser usado (e.g., OMPL com RRTConnect)
#     # move_group.set_planner_id("RRTConnectkConfigDefault") # Exemplo
#     # move_group.set_planning_time(5) # Tempo máximo para planejamento em segundos

#     # Cria um publicador ROS para exibir trajetórias planejadas no Rviz
#     # O tópico "/move_group/display_planned_path" está presente nos seus tópicos.
#     display_trajectory_publisher = rospy.Publisher(
#         "/move_group/display_planned_path",
#         moveit_msgs.msg.DisplayTrajectory,
#         queue_size=20,
#     )

#     rospy.loginfo("============ Referência de planejamento: %s" % move_group.get_planning_frame())
#     rospy.loginfo("============ End effector link: %s" % move_group.get_end_effector_link())
#     rospy.loginfo("============ Juntas ativas do grupo '%s': %s" % (move_group.get_name(), move_group.get_active_joints()))
#     rospy.loginfo("============ Nomes das juntas do grupo '%s': %s" % (move_group.get_name(), move_group.get_joints()))


#     # --- Configuração do Objetivo de Junta ---
#     # Pegue os valores de junta atuais como ponto de partida
#     joint_goal = move_group.get_current_joint_values()
#     rospy.loginfo(f"============ Valores de junta atuais: {joint_goal}")
#     rospy.loginfo(f"============ Número de juntas no grupo 'manipulator': {len(joint_goal)}")

#     # Defina os novos valores de junta.
#     # IMPORTANTE: O UR5 possui 6 juntas. Certifique-se de que o array 'val' tenha 6 elementos
#     # e que eles correspondam à ordem das juntas do seu MoveIt! setup.
#     # Você pode verificar a ordem das juntas com move_group.get_joints()
    
#     # Exemplo de valores para uma pose específica (substitua pelos seus valores desejados)
#     # Estes são valores de exemplo. Certifique-se de que são válidos e alcançáveis.
#     # A ordem geralmente é: shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3
#     val = [1.32, 0.0, -1.33, 1.56, -1.69, -1.68] # Apenas 6 valores para as 6 juntas do UR5

#     if len(val) != len(joint_goal):
#         rospy.logerr(f"Erro: O número de valores no array 'val' ({len(val)}) não corresponde ao número de juntas do grupo MoveIt! ({len(joint_goal)}).")
#         rospy.logerr("Para o UR5, o grupo 'manipulator' deve ter 6 juntas.")
#         moveit_commander.roscpp_shutdown()
#         return

#     # Atribua os novos valores de junta
#     for i in range(len(val)):
#         joint_goal[i] = val[i]

#     rospy.loginfo(f"============ Tentando mover para os valores de junta: {joint_goal}")

#     # O comando go pode ser chamado com valores de junta, poses, ou sem parâmetros
#     # se você já tiver definido o alvo para o grupo.
#     success = move_group.go(joint_goal, wait=True)

#     # Libera o alvo após o movimento para evitar que ele permaneça bloqueado
#     move_group.stop()
#     # É uma boa prática limpar os alvos de pose após o movimento, especialmente se você for definir novas poses
#     move_group.clear_pose_targets()

#     if success:
#         rospy.loginfo("Movimento concluído com sucesso para a pose de junta!")
#     else:
#         rospy.logwarn("O planejamento ou execução do movimento falhou.")

#     # Finaliza a comunicação com o MoveIt!
#     moveit_commander.roscpp_shutdown()
#     rospy.loginfo("MoveIt! Commander desligado.")

# if __name__ == '__main__':
#     try:
#         move_ur5_to_pose()
#     except rospy.ROSInterruptException:
#         rospy.loginfo("Nó ROS interrompido.")
#     except Exception as e:
#         rospy.logerr(f"Ocorreu um erro: {e}")


