import numpy as np
import sapien.core as sapien
from sapien.core import Pose

from mani_skill2_real2sim.agents.base_agent import BaseAgent
from mani_skill2_real2sim.agents.configs.jaco import defaults
from mani_skill2_real2sim.utils.common import compute_angle_between
from mani_skill2_real2sim.utils.sapien_utils import (
    get_entity_by_name,
    get_pairwise_contact_impulse,
)


class Jaco(BaseAgent):
    _config: defaults.JacoDefaultConfig

    """
        Jaco 6DoF robot
        links:
            [Actor(name="world", id="2"), 
            Actor(name="root", id="3"), 
            Actor(name="j2n6s300_link_base", id="4"), 
            Actor(name="j2n6s300_link_1", id="5"), 
            Actor(name="j2n6s300_link_2", id="6"), 
            Actor(name="j2n6s300_link_3", id="7"), 
            Actor(name="j2n6s300_link_4", id="8"), 
            Actor(name="j2n6s300_link_5", id="9"),
            Actor(name="j2n6s300_link_6", id="10"),
            Actor(name="j2n6s300_end_effector", id="17"), 
            Actor(name="j2n6s300_link_finger_1", id="15"), 
            Actor(name="j2n6s300_link_finger_tip_1", id="16"), 
            Actor(name="j2n6s300_link_finger_2", id="13"), 
            Actor(name="j2n6s300_link_finger_tip_2", id="14"), 
            Actor(name="j2n6s300_link_finger_3", id="11"), 
            Actor(name="j2n6s300_link_finger_tip_3", id="12")]

        active_joints: 
            ['j2n6s300_joint_1', 'j2n6s300_joint_2', 'j2n6s300_joint_3', 'j2n6s300_joint_4', 'j2n6s300_joint_5', 
            'j2n6s300_joint_6', 'j2n6s300_joint_finger_1', 'j2n6s300_joint_finger_2', 'j2n6s300_joint_finger_3']
        joint_limits:

            [[-inf,  inf],  
            [0.82030475, 5.4628806 ], 
            [0.33161256, 5.951573  ], 
            [-inf,  inf], 
            [-inf,  inf], 
            [-inf,  inf], 
            [0., 2.], 
            [0., 2.], 
            [0., 2.]]

    """

    """
        Jaco Gym 7DoF robot
        links:
            [Actor(name="root", id="2"), 
            Actor(name="j2s7s300_link_base", id="3"), 
            Actor(name="j2s7s300_link_1", id="4"), 
            Actor(name="j2s7s300_link_2", id="5"), 
            Actor(name="j2s7s300_link_3", id="6"), 
            Actor(name="j2s7s300_link_4", id="7"), 
            Actor(name="j2s7s300_link_5", id="8"), 
            Actor(name="j2s7s300_link_6", id="9"), 
            Actor(name="j2s7s300_link_7", id="10"), 
            Actor(name="j2s7s300_end_effector", id="17"), 
            Actor(name="j2s7s300_link_finger_1", id="15"), 
            Actor(name="j2s7s300_link_finger_tip_1", id="16"), 
            Actor(name="j2s7s300_link_finger_2", id="13"), 
            Actor(name="j2s7s300_link_finger_tip_2", id="14"), 
            Actor(name="j2s7s300_link_finger_3", id="11"), 
            Actor(name="j2s7s300_link_finger_tip_3", id="12")]


        active_joints: 
            ['j2s7s300_joint_1', 'j2s7s300_joint_2', 'j2s7s300_joint_3', 
            'j2s7s300_joint_4', 'j2s7s300_joint_5', 'j2s7s300_joint_6', 
            'j2s7s300_joint_7', 'j2s7s300_joint_finger_1', 
            'j2s7s300_joint_finger_2', 'j2s7s300_joint_finger_3']
        joint_limits:

            [[-inf,  inf],  
            [0.82030475, 5.4628806 ], 
            [0.33161256, 5.951573  ], 
            [-inf,  inf], 
            [-inf,  inf], 
            [-inf,  inf], 
            [0., 2.], 
            [0., 2.], 
            [0., 2.]]

    """


    @classmethod
    def get_default_config(cls):
        return defaults.JacoDefaultConfig()

    def __init__(
        self, scene, control_freq, control_mode=None, fix_root_link=True, config=None
    ):
        
        if control_mode is None:  # if user did not specify a control_mode
            control_mode = "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos"
        super().__init__(
            scene,
            control_freq,
            control_mode=control_mode,
            fix_root_link=fix_root_link,
            config=config,
        )

    def _after_init(self):
        super()._after_init()

        self.base_link = [x for x in self.robot.get_links() if x.name == f"{self.config.robot_name}_link_base"][0]
        self.ee_link = [x for x in self.robot.get_links() if x.name == f"{self.config.robot_name}_end_effector"][0]

        # Joints
        self.joint_finger_1 = get_entity_by_name(
            self.robot.get_joints(), f"{self.config.robot_name}_joint_finger_1"
        )
        self.joint_finger_2 = get_entity_by_name(
            self.robot.get_joints(), f"{self.config.robot_name}_joint_finger_2"
        )
        self.joint_finger_3 = get_entity_by_name(
            self.robot.get_joints(), f"{self.config.robot_name}_joint_finger_3"
        )



        self.link_finger_1 = get_entity_by_name(
            self.robot.get_links(), f"{self.config.robot_name}_link_finger_1"
        )
        self.link_finger_tip_1 = get_entity_by_name(
            self.robot.get_links(), f"{self.config.robot_name}_link_finger_tip_1"
        )
        self.link_finger_2 = get_entity_by_name(
            self.robot.get_links(), f"{self.config.robot_name}_link_finger_2"
        )
        self.link_finger_tip_2 = get_entity_by_name(
            self.robot.get_links(), f"{self.config.robot_name}_link_finger_tip_2"
        )
        self.link_finger_3 = get_entity_by_name(
            self.robot.get_links(), f"{self.config.robot_name}_link_finger_3"
        )
        self.link_finger_tip_3 = get_entity_by_name(
            self.robot.get_links(), f"{self.config.robot_name}_link_finger_tip_3"
        )

    @property
    def gripper_closedness(self):
        finger_qpos = self.robot.get_qpos()[-3:]
        finger_qlim = self.robot.get_qlimits()[-3:]
        closedness_left = (finger_qlim[0, 1] - finger_qpos[0]) / (
            finger_qlim[0, 1] - finger_qlim[0, 0]
        )
        closedness_right = (finger_qlim[1, 1] - finger_qpos[1]) / (
            finger_qlim[1, 1] - finger_qlim[1, 0]
        )
        closedness_middle = (finger_qlim[2, 1] - finger_qpos[2]) / (
            finger_qlim[2, 1] - finger_qlim[2, 0]
        )
        return np.maximum(np.mean([closedness_left, closedness_right, closedness_middle]), 0.0)

    def get_fingers_info(self):
        finger_1_pos = self.link_finger_1.get_global_pose().p
        finger_2_pos = self.link_finger_2.get_global_pose().p
        finger_3_pos = self.link_finger_3.get_global_pose().p

        finger_1_vel = self.link_finger_1.get_velocity()
        finger_2_vel = self.link_finger_2.get_velocity()
        finger_3_vel = self.link_finger_3.get_velocity()

        return {
            "finger_1_pos": finger_1_pos,
            "finger_2_pos": finger_2_pos,
            "finger_3_pos": finger_3_pos,
            "finger_1_vel": finger_1_vel,
            "finger_2_vel": finger_2_vel,
            "finger_3_vel": finger_3_vel,
        }

    def check_grasp(self, actor: sapien.ActorBase, min_impulse=1e-6, max_angle=60):
        assert isinstance(actor, sapien.ActorBase), type(actor)
        contacts = self.scene.get_contacts()

        impulse_finger_1 = get_pairwise_contact_impulse(
            contacts, self.link_finger_1, actor
        )
        impulse_finger_2 = get_pairwise_contact_impulse(
            contacts, self.link_finger_2, actor
        )
        impulse_finger_3 = get_pairwise_contact_impulse(
            contacts, self.link_finger_3, actor
        )


        # direction to open the gripper
        direction_finger_1 = self.link_finger_1.pose.to_transformation_matrix()[:3, 1]
        direction_finger_2 = self.link_finger_2.pose.to_transformation_matrix()[:3, 1]
        direction_finger_3 = self.link_finger_3.pose.to_transformation_matrix()[:3, 1]

        # angle between impulse and open direction
        # TODO: check if this is correct. When adding things like the 3rd finger, the direction can change
        angle1 = compute_angle_between(direction_finger_1, impulse_finger_1)
        angle2 = compute_angle_between(-direction_finger_2, impulse_finger_2)
        angle3 = compute_angle_between(-direction_finger_3, impulse_finger_3)

        lflag = (np.linalg.norm(impulse_finger_1) >= min_impulse) and np.rad2deg(
            angle1
        ) <= max_angle
        rflag = (np.linalg.norm(impulse_finger_2) >= min_impulse) and np.rad2deg(
            angle2
        ) <= max_angle
        mflag = (np.linalg.norm(impulse_finger_3) >= min_impulse) and np.rad2deg(
            angle3
        ) <= max_angle

        return all([lflag, rflag, mflag])

    def check_contact_fingers(self, actor: sapien.ActorBase, min_impulse=1e-6):
        assert isinstance(actor, sapien.ActorBase), type(actor)
        contacts = self.scene.get_contacts()

        impulse_finger_1 = get_pairwise_contact_impulse(
            contacts, self.link_finger_1, actor
        )
        impulse_finger_2 = get_pairwise_contact_impulse(
            contacts, self.link_finger_2, actor
        )
        impulse_finger_3 = get_pairwise_contact_impulse(
            contacts, self.link_finger_3, actor
        )


        return (
            np.linalg.norm(impulse_finger_1) >= min_impulse,
            np.linalg.norm(impulse_finger_2) >= min_impulse,
            np.linalg.norm(impulse_finger_3) >= min_impulse,
        )

    @staticmethod
    def build_grasp_pose(approaching, closing, center):
        """
            Build a grasp pose (WidowX gripper).
            From link_gripper's frame, x=approaching, -y=closing
        """
        assert np.abs(1 - np.linalg.norm(approaching)) < 1e-3
        assert np.abs(1 - np.linalg.norm(closing)) < 1e-3
        assert np.abs(approaching @ closing) <= 1e-3
        ortho = np.cross(approaching, closing)
        T = np.eye(4)
        T[:3, :3] = np.stack([approaching, closing, ortho], axis=1)
        T[:3, 3] = center
        return Pose.from_transformation_matrix(T)

    @property
    def base_pose(self):
        return self.base_link.get_pose()
    
    @property
    def ee_pose(self):
        return self.ee_link.get_pose()


class JacoBridgeDatasetCameraSetup(Jaco):
    _config: defaults.JacoBridgeDatasetCameraSetupConfig

    @classmethod
    def get_default_config(cls):
        return defaults.JacoBridgeDatasetCameraSetupConfig()


class JacoSinkCameraSetup(Jaco):
    _config: defaults.JacoSinkCameraSetupConfig

    @classmethod
    def get_default_config(cls):
        return defaults.JacoSinkCameraSetupConfig()