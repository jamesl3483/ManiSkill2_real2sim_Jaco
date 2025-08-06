"""
Example)



Visualize and manually control the robot in an environment, useful for debugging purposes.

Controls:
xyz: "i": +x, "k": -x, "j": +y, "l": -y, "u": +z, "o": -z
rotation rpy: "1": +r, "2": -r, "3": +p, "4": -p, "5": +y, "6": -y
reset environment: "r"
gripper open: "f", gripper close: "g"

If --enable-sapien-viewer, press "0" to switch to sapien viewer; 
    In the viewer, press "coordinate axes" option on the left twice to activate it;
    You can click on an object / articulation link and press "f" to focus on it; 
    Use right mouse button to rotate; middle-mouth-button + shift to translate;
    Under "scene hierarchy" on the bottom left, you can select different actors and articulation links;
    When an articulated object is selected (e.g., robot / cabinet), then under "articulation" on the bottom right, you can move the scrollbar to change each of its joint positions / angles; 
    Press "pause" on the top left to pause the simulation;
    Press "g" to grab object; "g" + "x"/"y"/"z" to move object along x/y/z axis;
    
If rgb_overlay_path is given, Press "v" to visualize the "greenscreened" image overlayed on top of simulation observation;
    this visualization can be used to debug e.g., the alignment of the real table and the simulation proxy table

To debug an environment, you can modify the "env_reset_options" in the main function to change the initial state of the environment.


**Example Scripts:**

# Google Robot
python mani_skill2_real2sim/examples/jaco_get_water.py -e GetWaterCustomInScene-v0 -c\
    arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_pos  --enable-sapien-viewer \
    -o rgbd robot google_robot_static sim_freq @501 control_freq @3 scene_name frl_apartment_stage_simple \
    rgb_overlay_mode debug rgb_overlay_path data/real_inpainting/open_drawer_b0.png rgb_overlay_cameras \
    overhead_camera model_ids a_cups


# JACO
python mani_skill2_real2sim/examples/jaco_get_water.py -e GetWaterCustomInScene-v0 -c \
    arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_pos  --enable-sapien-viewer \
    -o rgbd robot jaco sim_freq @501 control_freq @3 scene_name frl_apartment_stage_simple rgb_overlay_mode debug \
    rgb_overlay_path data/real_inpainting/open_drawer_b0.png rgb_overlay_cameras 3rd_view_camera model_ids a_cups


"""

import argparse
from operator import pos
import sys
import ast


import gymnasium as gym
import numpy as np
from transforms3d.quaternions import qmult
import time

from mani_skill2_real2sim.envs.sapien_env import BaseEnv
from mani_skill2_real2sim.utils.visualization.cv2_utils import OpenCVViewer
from mani_skill2_real2sim.utils.sapien_utils import look_at, normalize_vector
from sapien.core import Pose

MS1_ENV_IDS = [
    "OpenCabinetDoor-v1",
    "OpenCabinetDrawer-v1",
    "PushChair-v1",
    "MoveBucket-v1",
]


def parse_list_argument(arg_str):
    """
    Convert a string like "[1.0, 2.0, 3.0]" or "1.0 2.0 3.0" to a list of floats.
    """
    try:
        # Try parsing as Python literal (e.g., "[1.0, 2.0]")
        return ast.literal_eval(arg_str)
    except:
        # Fallback to space-separated float list
        return [float(x) for x in arg_str.split()]

def parse_bool_list(arg_str):
    try:
        return ast.literal_eval(arg_str)
    except:
        return [x.lower() == "true" for x in arg_str.split()]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, required=True)
    parser.add_argument("-o", "--obs-mode", type=str)
    parser.add_argument("--reward-mode", type=str)
    parser.add_argument("-c", "--control-mode", type=str, default="pd_ee_delta_pose")
    parser.add_argument("--render-mode", type=str, default="cameras")
    parser.add_argument("--add-segmentation", action="store_true")
    parser.add_argument("--enable-sapien-viewer", action="store_true")
    parser.add_argument("--model_ids", default=["a_cups"])
    parser.add_argument("--model_scales", default=[1.0], help="List of scales corresponding to each model")
    parser.add_argument("--init_xys",  default=[-0.09, 0.0], help="Flat list of x, y values for each object's initial position, e.g. --init_xys -0.09 0.0 -0.09 0.1")
    parser.add_argument("--init_rots",  default=[1, 0, 0, 0], help="Flat list of quaternion rotations for each object (e.g. 3 objects → 12 values)")
    parser.add_argument("--rand_rot_z", default=[False], help="Randomize Z-axis rotation for each object (e.g. --rand_rot_z False False True)")
    parser.add_argument("--rand_axis_rot_range",  default=[0.0], help="Random axis rotation range for each object")

    args, opts = parser.parse_known_args()

    print("args.init_xys:", args.init_xys)
    print("args.init_rots:", args.init_rots)
    print("args.rand_rot_z:", args.rand_rot_z)
    print("args.rand_axis_rot_range:", args.rand_axis_rot_range)
    print("args.model_ids:", args.model_ids)
    print("args.model_scales:", args.model_scales)


    # Parse env kwargs
    print("opts:", opts)
    eval_str = lambda x: eval(x[1:]) if x.startswith("@") else x
    env_kwargs = dict((x, eval_str(y)) for x, y in zip(opts[0::2], opts[1::2]))
    print("env_kwargs:", env_kwargs)
    args.env_kwargs = env_kwargs

    # Convert string arguments to lists
    args.model_ids = parse_list_argument(args.model_ids)
    args.model_scales = parse_list_argument(args.model_scales)
    args.init_xys = parse_list_argument(args.init_xys)
    args.init_rots = parse_list_argument(args.init_rots)
    args.rand_rot_z = parse_bool_list(args.rand_rot_z)
    args.rand_axis_rot_range = parse_list_argument(args.rand_axis_rot_range)
    num_models = len(args.model_ids)
    # --- Check all lists are the same length (number of models) ---
    expected_lengths = {
        "model_scales": len(args.model_scales),
        "init_xys": len(args.init_xys),
        "init_rots": len(args.init_rots),
        "rand_rot_z": len(args.rand_rot_z),
        "rand_axis_rot_range": len(args.rand_axis_rot_range),
    }

    for name, length in expected_lengths.items():
        if length != num_models:
            raise ValueError(
                f"Length mismatch: {name} has {length} entries, but model_ids has {num_models}."
            )


    return args


def main():
    np.set_printoptions(suppress=True, precision=3)
    args = parse_args()

    if args.env_id in MS1_ENV_IDS:
        if args.control_mode is not None and not args.control_mode.startswith("base"):
            args.control_mode = "base_pd_joint_vel_arm_" + args.control_mode

    if "robot" in args.env_kwargs:
        if "google_robot" in args.env_kwargs["robot"]:
            pose = look_at([1.0, 1.0, 2.0], [0.0, 0.0, 0.7])
            args.env_kwargs["render_camera_cfgs"] = {
                "render_camera": dict(p=pose.p, q=pose.q)
            }
        elif "jaco" in args.env_kwargs["robot"]:
            pose = look_at([1.0, 1.0, 2.0], [0.0, 0.0, 0.7])
            args.env_kwargs["render_camera_cfgs"] = {
                "render_camera": dict(p=pose.p, q=pose.q)
            }


    env: BaseEnv = gym.make( # create the environment with incorrect arguments
        args.env_id,
        obs_mode=args.obs_mode,
        reward_mode=args.reward_mode,
        control_mode=args.control_mode,
        render_mode=args.render_mode,
        camera_cfgs={"add_segmentation": args.add_segmentation},
        **args.env_kwargs
    )

    print("Observation space", env.observation_space)
    print("Action space", env.action_space)
    print("Control mode", env.control_mode)
    print("Reward mode", env.reward_mode)

    env_reset_options = {}
    if (not hasattr(env, "prepackaged_config")) or (not env.prepackaged_config):
        """
        Change the following reset options as you want to debug the environment
        """
        names_in_env_id_fxn = lambda name_list: any(
            name in args.env_id for name in name_list
        )
        init_rot_quat = [0, 0, 0, 1]
        init_xy = [0.452, -0.609] # Change the robot's position
        #TODO: Change so that this is a list of a dictionary

        env_reset_options = { # full initialization options
            "model_id": args.model_ids,  # Change the object model ids
            "model_scale": args.model_scales,  # Change the object scale
            "obj_init_options": {"init_xy": args.init_xys,
                                 "init_rot_quat": args.init_rots,
                                 "init_rand_rot_z": args.rand_rot_z,
                                 "init_rand_axis_rot_range": args.rand_axis_rot_range,
                                 },
            "robot_init_options": {
                "init_xy": init_xy,
                "init_rot_quat": init_rot_quat,
            }}
    print("environment:", env)
    print("env_reset_options:", env_reset_options)
    obs, info = env.reset(options=env_reset_options) # reset the environment with the options specified above
    print("Reset info:", info)
    print("Instruction:", env.get_language_instruction())
    after_reset = True

    if "google_robot" in env.agent.robot.name:
        print(
            "overhead camera pose",
            env.unwrapped._cameras["overhead_camera"].camera.pose,
        )
        print(
            "overhead camera pose wrt robot base",
            env.agent.robot.pose.inv()
            * env.unwrapped._cameras["overhead_camera"].camera.pose,
        )
    elif "jaco" in env.agent.robot.name:
        print(
            "3rd view camera pose",
            env.unwrapped._cameras["3rd_view_camera"].camera.pose,
        )
        print(
            "3rd view camera pose wrt robot base",
            env.agent.robot.pose.inv()
            * env.unwrapped._cameras["3rd_view_camera"].camera.pose,
        )


    print("robot pose", env.agent.robot.pose)
    # env.obj.get_collision_shapes()[0].get_physical_material().static_friction / dynamic_friction / restitution # object material properties

    # Viewer
    if args.enable_sapien_viewer:
        env.render_human()
    opencv_viewer = OpenCVViewer(exit_on_esc=False)

    def render_wait():
        if not args.enable_sapien_viewer:
            return
        while True:
            env.render_human()
            sapien_viewer = env.viewer
            if sapien_viewer.window.key_down("0"):
                break

    # Embodiment
    has_base = "base" in env.agent.controller.configs
    num_arms = sum("arm" in x for x in env.agent.controller.configs)
    has_gripper = any("gripper" in x for x in env.agent.controller.configs)
    is_google_robot = "google_robot" in env.agent.robot.name
    is_widowx = "wx250s" in env.agent.robot.name
    is_gripper_delta_target_control = (
        env.agent.controller.controllers["gripper"].config.use_target
        and env.agent.controller.controllers["gripper"].config.use_delta
    )

    def get_reset_gripper_action():
        # open gripper at initialization
        if not is_google_robot:
            return 1
        else:
            # for google robot, open-and-close actions are reversed
            return -1

    gripper_action = get_reset_gripper_action()

    EE_ACTION = (
        0.1 if not (is_google_robot or is_widowx) else 0.02
    )  # google robot and widowx use unnormalized action space
    EE_ROT_ACTION = (
        1.0 if not (is_google_robot or is_widowx) else 0.1
    )  # google robot and widowx use unnormalized action space

    # print("obj pose", env.obj.pose, "tcp pose", env.tcp.pose)
    print("qpos", env.agent.robot.get_qpos())
    first_pose = env.tcp.pose   # save the home pose of the end-effector

    # velocity control
    velocity_control_pos = [0,0,0]
    velocity_control_rot = [0,0,0]


    while True:
        time.sleep(0.01) 
        # -------------------------------------------------------------------------- #
        # Visualization
        # -------------------------------------------------------------------------- #
        if args.enable_sapien_viewer:
            env.render_human()

        render_frame = env.render()

        if after_reset:
            after_reset = False
            # Re-focus on opencv viewer
            if args.enable_sapien_viewer:
                opencv_viewer.close()
                opencv_viewer = OpenCVViewer(exit_on_esc=False)

        # -------------------------------------------------------------------------- #
        # Interaction
        # -------------------------------------------------------------------------- #
        # Input
        # key = opencv_viewer.imshow(image=render_frame, delay=1, non_blocking=True)
        key = opencv_viewer.imshow(image=render_frame, delay=10)
        # key = opencv_viewer.imshow(render_frame)

        if has_base:
            base_action = np.zeros([4])  # hardcoded
        else:
            base_action = np.zeros([0])

        # Parse end-effector action
        if (
            "pd_ee_delta_pose" in args.control_mode
            or "pd_ee_target_delta_pose" in args.control_mode
        ):
            ee_action = np.zeros([6])
        elif (
            "pd_ee_delta_pos" in args.control_mode
            or "pd_ee_target_delta_pos" in args.control_mode
        ):
            ee_action = np.zeros([3])
        else:
            raise NotImplementedError(args.control_mode)

        velocity_active = True

        # End-effector
        if num_arms > 0:

            # if velocity_active is true, they you change the velocity array that will be applied to the end-effector position at the end
            # Position
            if key == "1":  # +x
                if velocity_active:
                    velocity_control_pos[0] = EE_ACTION 
                else:
                    ee_action[0] = EE_ACTION
                
            elif key == "3":  # -x
                if velocity_active:
                    velocity_control_pos[0] = -EE_ACTION
                else:
                    ee_action[0] = -EE_ACTION
            elif key == "2":  # +y
                if velocity_active:
                    velocity_control_pos[1] = EE_ACTION 
                else:
                    ee_action[1] = EE_ACTION
            elif key == "4":  # -y
                if velocity_active:
                    velocity_control_pos[1] = -EE_ACTION 
                else:
                    ee_action[1] = -EE_ACTION
            elif key == "5":  # +z
                if velocity_active:
                    velocity_control_pos[2] = EE_ACTION 
                else:
                    ee_action[2] = EE_ACTION
            elif key == "6":  # -z
                if velocity_active:
                    velocity_control_pos[2] = -EE_ACTION 
                else:
                    ee_action[2] = -EE_ACTION

            if velocity_active:
                if velocity_control_pos[0] > 0:
                    velocity_control_pos[0] -= EE_ACTION/4.0
                elif velocity_control_pos[0] < 0:
                    velocity_control_pos[0] += EE_ACTION/4.0
                if velocity_control_pos[1] > 0:
                    velocity_control_pos[1] -= EE_ACTION/4.0
                elif velocity_control_pos[1] < 0:
                    velocity_control_pos[1] += EE_ACTION/4.0
                if velocity_control_pos[2] > 0:
                    velocity_control_pos[2] -= EE_ACTION/4.0
                elif velocity_control_pos[2] < 0:
                    velocity_control_pos[2] += EE_ACTION/4.0

                VEL_THRESHOLD = 1e-4
                for i in range(3):  # x, y, z
                    if abs(velocity_control_pos[i]) < VEL_THRESHOLD:
                        velocity_control_pos[i] = 0.0

                ee_action[0:3] = velocity_control_pos


            # Rotation (axis-angle)
            if key == "q":
                ee_action[3:6] = (EE_ROT_ACTION, 0, 0)
            elif key == "e":
                ee_action[3:6] = (-EE_ROT_ACTION, 0, 0)
            elif key == "w":
                ee_action[3:6] = (0, EE_ROT_ACTION, 0)
            elif key == "s":
                ee_action[3:6] = (0, -EE_ROT_ACTION, 0)
            elif key == "a":
                ee_action[3:6] = (0, 0, EE_ROT_ACTION)
            elif key == "d":
                ee_action[3:6] = (0, 0, -EE_ROT_ACTION)

        # Gripper
        if has_gripper:
            if not is_google_robot:
                if key == "f":  # open gripper
                    gripper_action = 1
                elif key == "g":  # close gripper
                    gripper_action = -1
            else:
                if key == "f":  # open gripper
                    gripper_action = -1
                elif key == "g":  # close gripper
                    gripper_action = 1

        # Other functions
        if key == "0":  # switch to SAPIEN viewer
            render_wait()
        elif key == "r":  # reset env
            obs, info = env.reset(options=env_reset_options)
            print("Reset info:", info)
            print("Instruction:", env.get_language_instruction())
            gripper_action = get_reset_gripper_action()
            after_reset = True
            continue
        elif key == None:  # exit
            # print("Exit")
            if velocity_active:
                pass
            else:
                continue

        elif key == "t": # home position
            from transforms3d.euler import quat2euler

            from scipy.spatial.transform import Rotation as R

            # Target
            goal_pos = np.array([0.0126156, -0.390583, 1.04092])
            goal_rot = [0.0324987, -0.780173, -0.603852, 0.160115]

            home_pose = Pose(p=goal_pos, q=goal_rot)

            delta_pose = env.tcp.pose * first_pose.inv()
            # rot = [delta_pose.q[3], delta_pose.q[0], delta_pose.q[1], delta_pose.q[2]]
            rot = delta_pose.q
            rot = R.from_quat(rot).as_rotvec()
            min_rot = [min(abs(x), abs(np.pi-x)) for x in rot]
            # print("changes", delta_pose )
            # print("changes", min_rot )
            # print("sum of rotation", np.sum(np.abs(np.array(min_rot))))
            # print("delta pose", np.sum(np.abs(np.array(delta_pose.p))))


            # print("rotation starting ............")
            norm_pos = np.zeros(3)
            norm_rot = np.zeros(3)

            if np.linalg.norm(min_rot) > 0.2:  # overall rotation threshold

                # Determine the axis with the largest absolute error
                max_axis = np.argmax(np.abs(min_rot))  # 0 = roll, 1 = pitch, 2 = yaw
                axis_error = rot[max_axis]

                # Apply EE_ACTION only in the direction of that axis

                if abs(axis_error) > 0.1:  # per-axis minimum threshold
                    if max_axis == 0:
                        norm_rot[2] = -1 * np.sign(axis_error) * EE_ACTION
                    elif max_axis == 1:
                        norm_rot[0] = np.sign(axis_error) * EE_ACTION
                    elif max_axis == 2:
                        norm_rot[1] = np.sign(axis_error) * EE_ACTION

                # ee_action = np.concatenate([[0, 0, 0], norm_rot])

            # print("translation starting ............") 
            if abs(np.abs(delta_pose.p[0])) > 0.05:
                norm_pos[0] =  delta_pose.p[0] / np.linalg.norm(delta_pose.p[1]) * EE_ACTION
                # ee_action = np.concatenate([norm_pos, [0, 0, 0]])
            elif abs(np.abs(delta_pose.p[1])) > 0.05:
                norm_pos[1] =  delta_pose.p[1] / np.linalg.norm(delta_pose.p[1]) * EE_ACTION
                # ee_action = np.concatenate([norm_pos, [0, 0, 0]])
            elif abs(np.abs(delta_pose.p[2])) > 0.05:
                norm_pos[2] =  delta_pose.p[2] / np.linalg.norm(delta_pose.p[2]) * EE_ACTION
                # ee_action = np.concatenate([norm_pos, [0, 0, 0]])

            ee_action = np.concatenate([norm_pos, norm_rot])

            print("delta pose", delta_pose)
            print("ee_action:", ee_action)


        # Visualize observation
        if key == "v":
            if "rgbd" in env.obs_mode:
                from itertools import chain

                from mani_skill2_real2sim.utils.visualization.misc import (
                    observations_to_images,
                    tile_images,
                )

                images = list(
                    chain(*[observations_to_images(x) for x in obs["image"].values()])
                )
                render_frame = tile_images(images)
                opencv_viewer.imshow(render_frame)
            elif "pointcloud" in env.obs_mode:
                import trimesh

                xyzw = obs["pointcloud"]["xyzw"]
                mask = xyzw[..., 3] > 0
                rgb = obs["pointcloud"]["rgb"]
                if "robot_seg" in obs["pointcloud"]:
                    robot_seg = obs["pointcloud"]["robot_seg"]
                    rgb = np.uint8(robot_seg * [11, 61, 127])
                trimesh.PointCloud(xyzw[mask, :3], rgb[mask]).show()

        # -------------------------------------------------------------------------- #
        # Post-process action
        # -------------------------------------------------------------------------- #
        # print("starting post-process action")
        action_dict = dict(base=base_action, arm=ee_action)
        if has_gripper:
            action_dict["gripper"] = gripper_action
        action = env.agent.controller.from_action_dict(action_dict)

        # print("action", action)
        obs, reward, terminated, truncated, info = env.step(action)

        if is_gripper_delta_target_control:
            gripper_action = 0

        # print("obj pose", env.obj.pose, "tcp pose", env.tcp.pose)
        # # print("tcp pose wrt robot base", env.agent.robot.pose.inv() * env.tcp.pose)
        # print("qpos", env.agent.robot.get_qpos())
        # # print("reward", reward)
        # print("terminated", terminated, "truncated", truncated)
        # print("info", info)

    env.close()


if __name__ == "__main__":
    main()
