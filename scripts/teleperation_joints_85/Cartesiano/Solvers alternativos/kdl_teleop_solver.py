#!/usr/bin/env python3

import rospy
import PyKDL as kdl
import tf_conversions.posemath as pm
from kdl_parser_py.urdf import treeFromUrdfModel
from urdf_parser_py.urdf import URDF
from geometry_msgs.msg import Pose, PoseStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Header # Necessário para o modo de teste

# --- VARIÁVEL GLOBAL PARA MODO DE TESTE ---
# Altere para False para usar a entrada do Unity (Padrão)
TEST_MODE = False
# ----------------------------------------


class KDLTeleopSolver:
    """
    Servidor de cinemática inversa usando KDL para controle do UR5.
    """
    
    # Parâmetros de Configuração
    BASE_LINK = 'base_link'
    EE_LINK = 'tool0'
    POSE_TOPIC = 'unity/target_pose'
    COMMAND_TOPIC = '/ur5/eff_joint_traj_controller/command'
    
    # Nomes das juntas do UR5 na ordem correta
    JOINT_NAMES = [
        'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
        'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
    ]

    def __init__(self):
        """Inicializa o solucionador KDL e os componentes ROS."""
        rospy.loginfo("Iniciando KDL Teleop Solver...")
        
        self._load_robot_model()
        self._initialize_kdl_solvers()
        self._setup_ros_communication()
        
        if not TEST_MODE:
            rospy.loginfo("Modo ATIVO: Recebendo poses do Unity em '%s'", self.POSE_TOPIC)
        else:
            rospy.loginfo("Modo ATIVO: Teste Manual. Publicando pose de teste.")

    # ... (Os métodos _load_robot_model e _initialize_kdl_solvers permanecem os mesmos) ...

    def _load_robot_model(self):
        """Carrega o modelo URDF e constrói a árvore KDL."""
        # ... (O método _load_robot_model permanece o mesmo) ...
        try:
            self.robot = URDF.from_parameter_server()
            success, kdl_tree_object = treeFromUrdfModel(self.robot)
            
            if not success:
                rospy.logerr("Falha Crítica: KDL não conseguiu construir a árvore a partir do URDF.")
                raise Exception("Falha ao construir a árvore KDL.")
                
            self.kdl_tree = kdl_tree_object
            self.chain = self.kdl_tree.getChain(self.BASE_LINK, self.EE_LINK)
            self.num_joints = self.chain.getNrOfJoints()
            
        except Exception as e:
            rospy.logerr("Erro ao carregar modelo do robô: %s", str(e))
            raise

    def _initialize_kdl_solvers(self):
        """Inicializa os solucionadores de cinemática direta e inversa."""
        # ... (O método _initialize_kdl_solvers permanece o mesmo) ...
        self.kdl_solver_fk = kdl.ChainFkSolverPos_recursive(self.chain)
        self.kdl_solver_vel = kdl.ChainIkSolverVel_pinv(self.chain)
        self.kdl_solver_pos = kdl.ChainIkSolverPos_NR(
            self.chain, 
            self.kdl_solver_fk, 
            self.kdl_solver_vel,
            maxiter=200,
            eps=1e-5
        )
        self.q_init = kdl.JntArray(self.num_joints)

    def _setup_ros_communication(self):
        """Configura publishers e subscribers ROS."""
        self.command_pub = rospy.Publisher(self.COMMAND_TOPIC, JointTrajectory, queue_size=1)
        
        # Só se inscreve no Unity se não estiver no modo de teste manual
        if not TEST_MODE:
            rospy.Subscriber(self.POSE_TOPIC, PoseStamped, self.pose_callback, queue_size=1)

    def process_pose(self, pose_stamped_msg):
        """
        Função de processamento central para ser usada tanto pelo callback quanto pelo teste.
        """
        # Converte a Pose do ROS para o formato Frame do KDL
        target_pose_kdl = pm.fromMsg(pose_stamped_msg.pose)
        
        # Tenta resolver o IK
        q_out = kdl.JntArray(self.num_joints)
        result = self.kdl_solver_pos.CartToJnt(self.q_init, target_pose_kdl, q_out)
        
        rospy.logdebug("Solução das juntas: %s", list(q_out))
        
        if result >= 0:  # IK Resolvido com Sucesso
            self._publish_joint_command(q_out)
            self.q_init = q_out  # Atualiza estado inicial para próxima iteração
            return True
        else:
            rospy.logwarn_throttle(2, "KDL falhou. Alvo pode ser inatingível. Código: %d", result)
            return False

    def pose_callback(self, data):
        """Callback chamado quando a pose do Unity é recebida."""
        self.process_pose(data)
        rospy.loginfo("Pose do Unity processada: %s", data.pose)

    def _publish_joint_command(self, joint_positions):
        """Publica comando de trajetória de juntas para o Gazebo."""
        # ... (O método _publish_joint_command permanece o mesmo) ...
        joint_positions_list = list(joint_positions)
        
        traj_msg = JointTrajectory()
        traj_msg.joint_names = self.JOINT_NAMES
        
        point = JointTrajectoryPoint()
        point.positions = joint_positions_list
        point.time_from_start = rospy.Duration(0.01)
        
        traj_msg.points.append(point)
        
        self.command_pub.publish(traj_msg)
        rospy.loginfo("Comando de junta publicado: %s", joint_positions_list)


def main():
    """Função principal."""
    rospy.init_node('kdl_teleop_solver_node', anonymous=True)
    solver = KDLTeleopSolver()
    
    if TEST_MODE:
        rospy.loginfo("INICIANDO MODO DE TESTE MANUAL...")
        pos = [0.002, 0.191, 0.739] # x, y, z em metros
        orient = [-1.633, -1.560, 0.061]  # Roll, Pitch, Yaw em radianos
        # Converte orientação Euler para Quaternion
        import tf.transformations
        quat = tf.transformations.quaternion_from_euler(orient[0], orient[1], orient[2])
        

        # --- DEFINA SUA POSE DE TESTE AQUI (Posição e Orientação) ---
        test_pose = Pose()
        test_pose.position.x = pos[0]    # 0.5m na frente do robô
        test_pose.position.y = pos[1]    # 0.0m de lado
        test_pose.position.z = pos[2]    # 0.5m de altura
        
        # Orientação Neutra (identidade, sem rotação)
        test_pose.orientation.x = quat[0]
        test_pose.orientation.y = quat[1]
        test_pose.orientation.z = quat[2]
        test_pose.orientation.w = quat[3]
        
        # Empacota no formato PoseStamped (necessário para o KDL)
        test_stamped = PoseStamped(header=Header(frame_id=solver.BASE_LINK), pose=test_pose)
        
        # Loop de teste (publica a pose uma vez a cada 2 segundos)
        rate = rospy.Rate(0.5) # 0.5 Hz (uma vez a cada 2 segundos)
        while not rospy.is_shutdown():
            solver.process_pose(test_stamped)
            rate.sleep()
        # --- FIM DO MODO DE TESTE MANUAL ---
        
    else:
        # Modo Padrão: Espera pelo Unity
        rospy.spin()


if __name__ == '__main__':
    main()