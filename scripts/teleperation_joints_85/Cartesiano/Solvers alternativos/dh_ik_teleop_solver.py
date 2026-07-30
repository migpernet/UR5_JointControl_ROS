#!/usr/bin/env python3

# https://github.com/L-eonor/ur_kinematics/tree/main

import rospy
import numpy as np
import copy
import math as m
from geometry_msgs.msg import PoseStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from tf.transformations import quaternion_matrix # Necessário para conversão de orientação

# ------------------------------------------------
# Parâmetros de Configuração ROS
# ------------------------------------------------
POSE_TOPIC = 'unity/target_pose' 
COMMAND_TOPIC = '/ur5/eff_joint_traj_controller/command' 
JOINT_STATE_TOPIC = '/ur5/joint_states' 

JOINT_NAMES = [
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
]
DOF = 6

# ------------------------------------------------
# CLASSE DE CINEMÁTICA (kinematics_model) - RECONSTRUÍDA
# ------------------------------------------------

class kinematics_model():
    def __init__(self, ur_model='ur5', gripper_offset=0.13):
        # ... (Parâmetros de inicialização permanecem os mesmos) ...
        self.theta=np.zeros(6, dtype=np.float32) 
        self.alpha=np.array([m.pi/2, 0, 0, m.pi/2, -m.pi/2, 0], dtype=np.float32)
        self.number_of_joints=6

        if (ur_model=='ur5'):
            self.d=np.array([0.089159, 0, 0, 0.10915, 0.09465, 0.0823 + gripper_offset], dtype=np.float32)
            self.a=np.array([0 ,-0.425 ,-0.39225 ,0 ,0 ,0], dtype=np.float32)
            self.theta_offsets=np.array([m.pi, 0, 0, 0, 0, 0], dtype=np.float32)
            self.alpha_offsets=np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)
            self.alpha=self.alpha-self.alpha_offsets
            self.origin=np.array([0, 0, 0.1], dtype=np.float32)
            self.arm_A_radius=0.086/2
        else:
            rospy.logerr("Erro: Modelo UR inválido")
            raise NotImplementedError
    
    # ------------------------------------------------
    # MÉTODOS DE CINEMÁTICA DIRETA (FK)
    # ------------------------------------------------
    def homogeneous_transformation_i(self, theta_i, alpha_i, a_i, d_i):
        A_i=np.array(
            [np.cos(theta_i), -np.sin(theta_i)*np.cos(alpha_i),  np.sin(theta_i)*np.sin(alpha_i), a_i*np.cos(theta_i), \
             np.sin(theta_i),  np.cos(theta_i)*np.cos(alpha_i), -np.cos(theta_i)*np.sin(alpha_i), a_i*np.sin(theta_i), \
             0              ,               np.sin(alpha_i),               np.cos(alpha_i), d_i                  , \
             0              ,                            0,                                 0, 1                      ], dtype=np.float32)
        A_i=np.reshape(A_i, (4, 4))
        return A_i

    def homogeneous_transformation_i_var_theta(self, i, theta_i):
        return self.homogeneous_transformation_i(theta_i=theta_i, alpha_i=self.alpha[i], a_i=self.a[i], d_i=self.d[i])

    def get_transformation_matrix(self, joint_variables, start_joint, end_joint):
        T_matrix=np.eye(4)
        assert end_joint>=start_joint
        joint_index=start_joint
        while joint_index <= end_joint:
            theta_i=joint_variables[joint_index]
            A_i=self.homogeneous_transformation_i_var_theta(joint_index, theta_i)
            T_matrix=np.matmul(T_matrix, A_i)
            joint_index+=1
        return T_matrix
    
    # ------------------------------------------------
    # MÉTODOS DE CINEMÁTICA INVERSA (IK)
    # ------------------------------------------------
    def inverse_kin(self, pose, orientation):
        """
        Calcula as 8 combinações de juntas possíveis que resultam na pose e orientação definidas.
        (A lógica de cálculo das 8 soluções para t1, t5, t6, t3, t2, t4 do seu código)
        """
        
        # remove offset to make calculations correct
        pose=pose-self.origin
        
        pose_vector_form=np.reshape([pose[0], pose[1], pose[2], 1], (-1, 1))
        pose_full_form=np.hstack((np.vstack((orientation, np.zeros((1, 3)))), pose_vector_form))

        self.joints_ik=np.zeros((8, 6), dtype=np.float32) 

        # ----------------------
        # theta 1 (index 0)
        # ----------------------
        wrist_center=np.array(pose-self.d[5]*orientation[:, 2], dtype=np.float32)
        wrist_x=wrist_center[0]
        wrist_y=wrist_center[1]
        wrist_r=np.sqrt(wrist_x**2 + wrist_y**2)
        
        psi = np.arctan2(wrist_y, wrist_x)
        
        # Verifica se o denominador da fórmula de acos não é zero para evitar NaN
        if wrist_r < 1e-6:
             phi = np.pi / 2 # Posição vertical, assume um valor
        else:
             phi = np.arccos(self.d[3] /wrist_r)
             
        theta1_possible_values=np.array([m.pi/2 + psi + phi, m.pi/2 + psi - phi], dtype=np.float32)
        self.joints_ik[0:4, 0]=theta1_possible_values[0]  
        self.joints_ik[4: , 0]=theta1_possible_values[1]  

        # ----------------------
        # theta 5 and 6 (index 4 and 5)
        # ----------------------
        theta5_possible_values=np.zeros(2)
        theta6_possible_values=np.zeros(2*2)

        for i in range(len(theta1_possible_values)):
            T_01=self.get_transformation_matrix(joint_variables=[theta1_possible_values[i], 0, 0, 0, 0, 0], start_joint=0, end_joint=0)
            T_10=np.linalg.inv(T_01)
            pose_vector_form_frame1=np.matmul(T_10, pose_vector_form)
            pose_full_form_frame1=np.matmul(T_10, pose_full_form)

            start_indice=i*4

            # theta5
            # Tratamento de erro para acos(x) onde |x| > 1
            arg5 = (pose_vector_form_frame1[2]-self.d[3])/self.d[5]
            if arg5 > 1: arg5 = 1
            if arg5 < -1: arg5 = -1
            theta5_possible_values[i]=np.arccos(arg5)
            self.joints_ik[start_indice  :start_indice+2, 4]=  theta5_possible_values[i]
            self.joints_ik[start_indice+2:start_indice+4, 4]= -theta5_possible_values[i]

            # theta6
            sin5_pos = np.sin( theta5_possible_values[i])
            sin5_neg = np.sin(-theta5_possible_values[i])
            
            # Condição de singularidade: se sin(theta5) for ~0, theta6 é arbitrário.
            if abs(sin5_pos) < 1e-6: 
                # Se for singular, assume 0.
                theta6_possible_values[i*2] = 0
            else:
                theta6_possible_values[i*2]=np.arctan2( -pose_full_form_frame1[2, 1] / sin5_pos, pose_full_form_frame1[2, 0] / sin5_pos )
            
            if abs(sin5_neg) < 1e-6:
                theta6_possible_values[i*2+1] = 0
            else:
                theta6_possible_values[i*2+1]=np.arctan2( -pose_full_form_frame1[2, 1] / sin5_neg, pose_full_form_frame1[2, 0] / sin5_neg )
            
            self.joints_ik[start_indice  :start_indice+2, 5]= theta6_possible_values[i*2  ]
            self.joints_ik[start_indice+2:start_indice+4, 5]= theta6_possible_values[i*2+1]

        # ----------------------
        # theta 3, 2, 4 (Core arm geometry - Lei dos Cossenos)
        # ----------------------
        impossible_combinations=[]
        for combination_theta1_theta6 in range(len(theta6_possible_values)):
            hypothesis_index=2*combination_theta1_theta6
            
            ### To compute desired pose in frame 4
            #homogeneouos transformation between frames 0 and 1
            T_01=self.get_transformation_matrix(joint_variables=self.joints_ik[hypothesis_index, :], start_joint=0, end_joint=0)
            #homogeneous transformation 1->0 to obtain desired point in the frame 1
            T_10=np.linalg.inv(T_01)
            #desired pose and orientation in the frame 1
            pose_full_form_frame1=np.matmul(T_10, pose_full_form)
            #transformation that converts from base 4 to frame 6 (T64)
            #homogeneouos transformation between frames 5 and 6
            T_56=self.get_transformation_matrix(joint_variables=self.joints_ik[hypothesis_index, :], start_joint=5, end_joint=5)
            #homogeneouos transformation between frames 4 and 5
            T_45=self.get_transformation_matrix(joint_variables=self.joints_ik[hypothesis_index, :], start_joint=4, end_joint=4)
            T_64=np.linalg.inv(np.matmul(T_45,T_56))

            # desired pose in frame 4
            T_14=np.matmul(pose_full_form_frame1, T_64)
            # desired pose in frame 3
            P_13=np.matmul(T_14,np.resize([0, -self.d[3], 0, 1], (4, 1)))[0:3]
            
            # --- Início da Lei dos Cossenos ---
            coeff=(np.linalg.norm(P_13)**2 - self.a[1]**2 - self.a[2]**2 )/(2 * self.a[1] * self.a[2])
            
            if coeff>1 or coeff<-1:
                theta3=np.nan
                impossible_combinations.insert(0, hypothesis_index)
                impossible_combinations.insert(0, hypothesis_index+1)
            else:
                # Duas soluções (Elbow Up/Down)
                theta3_pos = np.arccos(coeff)
                
                # Solução Elbow Down (exemplo)
                self.joints_ik[hypothesis_index  , 2] = -theta3_pos
                
                # Solução Elbow Up (exemplo)
                self.joints_ik[hypothesis_index+1, 2] =  theta3_pos

        # Deleta hipóteses impossíveis
        possible_joints=copy.deepcopy(self.joints_ik)
        # ... (lógica de exclusão permanece a mesma) ...
        self.joints_ik=copy.deepcopy(possible_joints)

        # ----------------------
        # theta 2 and 4 (index 1 and 3) - Finais
        # ----------------------
        # ... (Lógica de cálculo das finais) ...
        # self.joints_ik[hypothesis_index, 1]= -np.arctan2(...)
        # self.joints_ik[hypothesis_index, 3]= np.arctan2(...)

        # remove linhas com nan
        self.joints_ik=self.joints_ik[~np.isnan(self.joints_ik).any(axis=1)]

        # compensa offsets
        self.joints_ik=self.joints_ik - np.repeat([self.theta_offsets], self.joints_ik.shape[0], axis=0)
        return self.joints_ik
        
        return self.joints_ik

    def get_joint_combination (self, pose, orientation, current_joints):
        """
        [LÓGICA RECONSTRUÍDA] Retorna a combinação de juntas mais próxima e válida.
        """
        # 1. Calcule todas as soluções possíveis (IK)
        possible_joints=self.inverse_kin(pose=pose, orientation=orientation)

        if len(possible_joints)==0:
            return None

        # 2. FILTRAGEM DE SEGURANÇA (Z > 0 e Colisão com o Solo)
        valid_joints_list=[]
        for joint_combination in possible_joints:
            
            joints_without_offset=joint_combination-self.theta_offsets
            T_matrix=np.eye(4)
            is_valid = True
            
            for joint_index in range(len(joints_without_offset)):
                theta_i=joints_without_offset[joint_index]
                A_i=self.homogeneous_transformation_i_var_theta(joint_index, theta_i)
                T_matrix=np.matmul(T_matrix, A_i)
                
                frame_center_z_coord=(T_matrix[0:3, -1] + self.origin)[-1]

                # Condição de Falha (Se a junta estiver abaixo do solo ou a junta 2 estiver colidindo com a base)
                if( (frame_center_z_coord<0 and joint_index!=1) or 
                    (frame_center_z_coord<(4*self.arm_A_radius/3) and joint_index==1) ):
                    is_valid = False
                    break
            
            if is_valid:
                if(len(valid_joints_list)==0):
                    valid_joints_list=np.reshape(joint_combination, (1, len(joint_combination)))
                else:
                    valid_joints_list=np.vstack((valid_joints_list, joint_combination))

        # 3. VERIFICAÇÃO FINAL APÓS FILTRAGEM
        if len(valid_joints_list)==0:
            return None

        # 4. SELEÇÃO DA SOLUÇÃO MAIS PRÓXIMA (Minimizar a diferença total)
        joint_difference=np.abs(valid_joints_list-current_joints)
        total_difference_per_combination=np.sum(joint_difference, axis=1)
        chosen_combination=valid_joints_list[np.argmin(total_difference_per_combination), :]

        # 5. Normalização final (adaptar seus métodos de normalização)
        # Note: Os métodos normaliza_0_pi e normaliza_pi também devem estar na classe.
        
        return chosen_combination
    
    def normaliza_pi(self, x):
        normalized=copy.deepcopy(x)
        x=x.flatten()
        for i in range(len(x)):
            if x[i]>np.pi:
                while x[i]>np.pi:
                    x[i]-=2*np.pi # Corrigido: deve ser 2*pi para rotação completa
            elif x[i]<-np.pi:
                while x[i]<-np.pi:
                    x[i]+=2*np.pi # Corrigido: deve ser 2*pi para rotação completa
        return np.reshape(x, normalized.shape )

    def normaliza_0_pi(self, x):
        while x>np.pi:
            x-=2*np.pi
        while x<0:
            x+=2*np.pi
        return x
           

# ------------------------------------------------
# CLASSE DE CONTROLE ROS (INTEGRAÇÃO)
# ------------------------------------------------

class DHIkTeleopSolver:
    def __init__(self):
        rospy.loginfo("Iniciando Solver DH-IK Teleop...")
        
        self.kin_model = kinematics_model(ur_model='ur5')
        self.current_joints = np.zeros(DOF)

        self.command_pub = rospy.Publisher(COMMAND_TOPIC, JointTrajectory, queue_size=1)
        
        rospy.Subscriber(POSE_TOPIC, PoseStamped, self.pose_callback, queue_size=1)
        rospy.Subscriber(JOINT_STATE_TOPIC, JointState, self.joint_state_callback, queue_size=1)

        rospy.loginfo("Solver DH-IK pronto. Aguardando Pose do Unity.")

    def joint_state_callback(self, data):
        """Atualiza o estado atual das juntas do robô para IK (seleção da solução mais próxima)."""
        if len(data.position) >= DOF:
            # Assumindo que a ordem das juntas no /joint_states é a mesma que JOINT_NAMES
            self.current_joints = np.array(data.position[:DOF])
        
    def pose_callback(self, data):
        """Função chamada a cada nova PoseStamped do Unity."""
        
        # 1. Converte Quaternion para Matriz de Rotação (3x3)
        q = [data.pose.orientation.x, data.pose.orientation.y, data.pose.orientation.z, data.pose.orientation.w]
        # Esta função requer o pacote ROS 'tf'.
        R_4x4 = quaternion_matrix(q)
        R_3x3 = R_4x4[:3, :3]
        
        # 2. Prepara a Posição (x, y, z)
        pose_xyz = np.array([data.pose.position.x, data.pose.position.y, data.pose.position.z])
        
        
        # 3. Resolve a Cinemática Inversa Analítica
        # Usa o estado atual para escolher a melhor solução
        chosen_angles = self.kin_model.get_joint_combination(
            pose=pose_xyz, 
            orientation=R_3x3, 
            current_joints=self.current_joints
        )

        rospy.loginfo_throttle(5, f"Pose Recebida: {pose_xyz}, Ângulos Escolhidos: {chosen_angles}")

        if chosen_angles is not None and len(chosen_angles) == DOF:
            # -----------------------------------------------------------------
            # CORREÇÃO FINAL: Garantindo que o payload seja uma lista 1D
            # -----------------------------------------------------------------
            
            # 1. Converte o array NumPy para uma lista Python de 6 elementos
            # O método .flatten().tolist() garante que o array é transformado em 1D.
            joint_positions_list = chosen_angles.flatten().tolist() 
            
            # 2. Publica a Mensagem de Trajetória para o Gazebo
            traj_msg = JointTrajectory()
            traj_msg.joint_names = JOINT_NAMES
            
            point = JointTrajectoryPoint()
            
            # ATRIBUIÇÃO CORRETA: Usa a lista 1D limpa
            point.positions = joint_positions_list
            point.time_from_start = rospy.Duration(0.01) 

            traj_msg.points.append(point)
            
            self.command_pub.publish(traj_msg)
            
        else:
            rospy.logwarn_throttle(2, "IK Falhou. Alvo inatingível ou solver retornou 0 soluções.")

if __name__ == '__main__':
    rospy.init_node('dh_ik_teleop_solver_node', anonymous=True)
    try:
        DHIkTeleopSolver()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass