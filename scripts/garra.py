#!/usr/bin/env python3

import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import time

def send_gripper_command(joint_name, target_position, duration=3.0):
    # Inicializa o nó ROS
    if not rospy.core.is_initialized():
        rospy.init_node('ur5_gripper_commander', anonymous=True)

    # Cria um publicador para o tópico de comando da garra
    pub = rospy.Publisher('/ur5/gripper_controller/command', JointTrajectory, queue_size=10)

    # Espera um pouco para o publicador se conectar ao tópico
    rospy.sleep(0.5)

    # Cria a mensagem JointTrajectory
    gripper_command = JointTrajectory()
    gripper_command.header.stamp = rospy.Time.now() # Carimbo de tempo atual
    gripper_command.joint_names = [joint_name] # Nome da junta da garra

    # Cria um ponto de trajetória
    point = JointTrajectoryPoint()
    point.positions = [target_position] # Posição alvo da junta
    point.time_from_start = rospy.Duration(duration) # Tempo para alcançar a posição

    gripper_command.points.append(point) # Adiciona o ponto à trajetória

    # Publica a mensagem
    rospy.loginfo(f"Enviando comando para a garra: {joint_name} para a posição {target_position} em {duration} segundos.")
    pub.publish(gripper_command)

    # Espera a duração para o movimento ser concluído
    rospy.sleep(duration + 0.5) 

def open_gripper():
    """
    Comanda a garra para abrir.
    """
    GRIPPER_JOINT_NAME = 'robotiq_85_left_knuckle_joint' 

    OPEN_POSITION = 0.00 

    send_gripper_command(GRIPPER_JOINT_NAME, OPEN_POSITION)
    rospy.loginfo("Comando de abertura da garra enviado.")

def close_gripper():
    """
    Comanda a garra para fechar.
    """
    GRIPPER_JOINT_NAME = 'robotiq_85_left_knuckle_joint' 

    CLOSE_POSITION = 0.46     # 0.798

    send_gripper_command(GRIPPER_JOINT_NAME, CLOSE_POSITION)
    rospy.loginfo("Comando de fechamento da garra enviado.")

if __name__ == '__main__':
    try:
        # Exemplo de uso:
        # Para fechar a garra:
        # close_gripper()
        # rospy.sleep(2) # Espera um pouco antes de abrir novamente

        # Para abrir a garra:
        open_gripper()
        rospy.sleep(2) # Espera para ver o resultado

        rospy.loginfo("Demonstração de abertura e fechamento da garra concluída.")

    except rospy.ROSInterruptException:
        rospy.loginfo("Script de controle da garra interrompido.")
    finally:
        # Garante que o nó ROS seja desligado corretamente
        if rospy.core.is_initialized():
            rospy.signal_shutdown("Script de controle da garra finalizado.")

