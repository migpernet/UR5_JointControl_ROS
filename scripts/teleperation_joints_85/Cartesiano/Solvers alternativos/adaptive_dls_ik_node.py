#!/usr/bin/env python3

import rospy
import PyKDL as kdl
import tf_conversions.posemath as pm
from kdl_parser_py.urdf import treeFromUrdfModel
from urdf_parser_py.urdf import URDF
from geometry_msgs.msg import Pose, PoseStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Header
from sensor_msgs.msg import JointState
import numpy as np # IMPORTANTE: Necessário para os cálculos de matriz

# --- VARIÁVEL GLOBAL PARA MODO DE TESTE ---
TEST_MODE = False
# ----------------------------------------

class KDLTeleopSolver:
    """
    Servidor de cinemática inversa usando KDL para controle do UR5.
    Implementa Damped Least Squares (DLS) Adaptativo para estabilidade em singularidades.
    """
    
    # Parâmetros de Configuração
    BASE_LINK = 'base_link'
    EE_LINK = 'tool0'
    POSE_TOPIC = 'unity/target_pose'
    COMMAND_TOPIC = '/ur5/eff_joint_traj_controller/command'
    JOINT_STATES_TOPIC = '/ur5/joint_states'
    
    JOINT_NAMES = [
        'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
        'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
    ]

    def __init__(self):
        rospy.loginfo("Iniciando KDL Teleop Solver com DLS ADAPTATIVO...")
        
        self.has_received_joints = False
        
        # Parâmetros do DLS Adaptativo
        self.w0 = 0.04        # Limiar de manipulabilidade (começa a amortecer abaixo disso)
        self.lambda_max = 0.2 # Amortecimento máximo (na singularidade exata) 0.2
        
        self._load_robot_model()
        self._initialize_kdl_solvers()
        self._setup_ros_communication()
        
        if not TEST_MODE:
            rospy.loginfo("Modo ATIVO: Recebendo poses do Unity no tópico '%s'", self.POSE_TOPIC)
        else:
            rospy.loginfo("Modo ATIVO: Teste Manual Gazebo. Publicando pose fixa.")

    def _load_robot_model(self):
        try:
            self.robot = URDF.from_parameter_server()
            success, kdl_tree_object = treeFromUrdfModel(self.robot)
            
            if not success:
                rospy.logerr("Falha Crítica: KDL não conseguiu construir a árvore a partir do URDF.")
                raise Exception("Falha ao construir a árvore KDL.")
                
            self.kdl_tree = kdl_tree_object
            self.chain = self.kdl_tree.getChain(self.BASE_LINK, self.EE_LINK)
            self.num_joints = self.chain.getNrOfJoints()
            rospy.loginfo("Modelo do robô carregado com sucesso. Juntas: %d", self.num_joints)
            
        except Exception as e:
            rospy.logerr("Erro ao carregar modelo do robô: %s", str(e))
            raise

    def _initialize_kdl_solvers(self):
        self.kdl_solver_fk = kdl.ChainFkSolverPos_recursive(self.chain)
        self.kdl_solver_vel = kdl.ChainIkSolverVel_wdls(self.chain)
        
        # Inicia com um valor quase zero (comportamento de pseudoinversa)
        self.kdl_solver_vel.setLambda(0.001) 
        
        self.kdl_solver_pos = kdl.ChainIkSolverPos_NR(
            self.chain, 
            self.kdl_solver_fk, 
            self.kdl_solver_vel,
            maxiter=500,
            eps=1e-3
        )
        
        # NOVO: Solver para extrair a matriz Jacobiana
        self.jac_solver = kdl.ChainJntToJacSolver(self.chain)
        
        self.q_init = kdl.JntArray(self.num_joints)

    def _setup_ros_communication(self): 
        self.command_pub = rospy.Publisher(self.COMMAND_TOPIC, JointTrajectory, queue_size=1)
        rospy.Subscriber(self.JOINT_STATES_TOPIC, JointState, self.joint_states_callback, queue_size=1)
        
        if not TEST_MODE:
            self.pose_sub = rospy.Subscriber(self.POSE_TOPIC, PoseStamped, self.pose_callback, queue_size=1)

    def joint_states_callback(self, msg):
        for i, joint_name in enumerate(self.JOINT_NAMES):
            if joint_name in msg.name:
                idx = msg.name.index(joint_name)
                self.q_init[i] = msg.position[idx]
        self.has_received_joints = True

    def calculate_adaptive_lambda(self):
        """Calcula o lambda adaptativo baseado na manipulabilidade atual."""
        # 1. Extrai o Jacobiano da configuração atual
        jac_kdl = kdl.Jacobian(self.num_joints)
        self.jac_solver.JntToJac(self.q_init, jac_kdl)
        
        # 2. Converte para Numpy para operações matriciais
        J = np.zeros((6, self.num_joints))
        for i in range(6):
            for j in range(self.num_joints):
                J[i, j] = jac_kdl[i, j]
                
        # 3. Calcula a manipulabilidade de Yoshikawa: w = sqrt(det(J * J^T))
        det_JJT = np.linalg.det(np.dot(J, J.T))
        w = np.sqrt(abs(det_JJT)) # abs evita erros numéricos de floats próximos a zero
        
        # 4. Aplica a lei de adaptação (Maciejewski/Nakamura)
        if w >= self.w0:
            # Longe da singularidade: erro zero
            return 0.001 
        else:
            # Perto da singularidade: amortecimento cresce quadraticamente
            return self.lambda_max * (1.0 - (w / self.w0))**2

    def process_pose(self, pose_stamped_msg):
        if not self.has_received_joints:
            rospy.logwarn_throttle(2.0, "KDL: Aguardando semente física do Gazebo...")
            return False

        # --- ATUALIZAÇÃO DO LAMBDA ADAPTATIVO ---
        lambda_adapt = self.calculate_adaptive_lambda()
        self.kdl_solver_vel.setLambda(lambda_adapt)
        # ----------------------------------------

        target_pose_kdl = pm.fromMsg(pose_stamped_msg.pose)
        q_out = kdl.JntArray(self.num_joints)
        
        result = self.kdl_solver_pos.CartToJnt(self.q_init, target_pose_kdl, q_out)
        
        if result >= 0: 
            self._publish_joint_command(q_out)
            return True
        else:
            # Com o lambda adaptativo, erros graves de IK serão muito raros
            rospy.logwarn_throttle(1, "Alvo fora de alcance. Código IK: %d (Lambda: %.3f)", result, lambda_adapt)
            return False

    def pose_callback(self, data):
        self.process_pose(data)

    def _publish_joint_command(self, joint_positions):
        joint_positions_list = list(joint_positions)
        
        traj_msg = JointTrajectory()
        traj_msg.joint_names = self.JOINT_NAMES
        
        point = JointTrajectoryPoint()
        point.positions = joint_positions_list
        point.time_from_start = rospy.Duration(0.01) 
        
        traj_msg.points.append(point)
        self.command_pub.publish(traj_msg)

def main():
    # ... (O restante da função main() permanece inalterado) ...
    rospy.init_node('kdl_teleop_solver_node', anonymous=True)
    solver = KDLTeleopSolver()
    
    if TEST_MODE:
        rospy.loginfo("--- INICIANDO ROTINA DE TESTE NO GAZEBO ---")
        
        pos = [0.3, 0.2, 0.5]
        orient = [0.0, 1.57, 0.0] 
        
        import tf.transformations
        quat = tf.transformations.quaternion_from_euler(orient[0], orient[1], orient[2])
        
        test_pose = Pose()
        test_pose.position.x = pos[0]
        test_pose.position.y = pos[1]
        test_pose.position.z = pos[2]
        
        test_pose.orientation.x = quat[0]
        test_pose.orientation.y = quat[1]
        test_pose.orientation.z = quat[2]
        test_pose.orientation.w = quat[3]
        
        rate = rospy.Rate(1.0)
        while not rospy.is_shutdown():
            test_stamped = PoseStamped(header=Header(frame_id=solver.BASE_LINK, stamp=rospy.Time.now()), pose=test_pose)
            solver.process_pose(test_stamped)
            rate.sleep()
            
    else:
        rospy.spin()

if __name__ == '__main__':
    main()
