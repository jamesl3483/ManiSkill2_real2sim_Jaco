#!/bin/bash

# ------------------------------------------------------------------------------
# Run the GetWaterCustomInScene-v0 environment using ManiSkill2 Real2Sim
# This script wraps a long CLI command for easier readability and reuse.
# ------------------------------------------------------------------------------

# Activate your environment if needed
# source ~/miniconda3/etc/profile.d/conda.sh
conda activate sapien

# Set environment variables (optional)
# export PYTHONPATH=/path/to/your/project:$PYTHONPATH


# ========= User Options =========
# Environment and scene
# - Options: GetWaterCustomInScene-v0, OpenDrawerCustomInScene-v0, GraspSingleOpenedCokeCanInScene-v0, ...
# - For custom scenes, ensure the scene is correctly set up in mani_skill2_real2sim/envs/custom_scenes
ENV_ID="GetWaterCustomInScene-v0"                   


# Scene configuration
# - Options: frl_apartment_stage_simple, modermodern_office, modern_bedroom, ...
# - For custom scenes, ensure the scene is correctly set up in mani_skill2_real2sim/envs/custom_scenes
# - For empty background, use "dummy", "dummy_drawer", "dummy_tabletop"
SCENE_NAME="frl_apartment_stage_simple"             


# Robot configuration
# Options: google_robot_static, google_robot_grip_camera, jaco, widowx, jaco_bridge_dataset_camera_setup, jaco_sink_camera_setup
# - For custom robots, ensure the robot is correctly set up in mani_skill2_real2sim/agents/configs/robots
# - google_robot_grip_camera adds a hand camera to the robot
# - google_robot_static is the default static robot with a base camera
# - Jaco is the kinova robot added to the environment
ROBOT_MODEL="google_robot_grip_camera"         

# Control mode
# - There are a variety of control modes available: if you decide to experiment, go at your own risk!
# - if it does not exist for the robot you are using, it will list the available control modes
CONTROL_MODE="arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_pos"

# Observation & camera
# - I dont ever touch this, but you can if you want
OBS_MODE="rgbd"                                            # Options: rgb, depth, rgbd
ENABLE_VIEWER="--enable-sapien-viewer"                     # Add or remove this to toggle GUI
OVERLAY_MODE="debug"                                       # Options: debug, none
OVERLAY_PATH="data/real_inpainting/open_drawer_b0.png"     # Path to overlay image

# Frequencies
SIM_FREQ="@501"                                            # Simulation frequency
CONTROL_FREQ="@3"                                          # Control frequency

# Object models and their configurations
# - These are the objects you want to interact with in the scene
# - Find more models in data/custom/models
# - For custom models, there are more models outside the folder
#     - ensure the textured file is an glb
#     - there is a script to convert from obj to glb in data/custom/convert_obj_to_glb.py
MODEL_ID="['a_cups','a_cups','003_cracker_box']" # Name of the model(s) to insert
MODEL_SCALES="[1.0, 1.0, 1.0]" # relative scale of the model(s) to insert
INIT_XYS="[[-0.09, 0.0], [-0.09, 0.1], [-0.09, 0.2]]" # Positions in the world, the z axis is automated to be above the table

# These are less important, but u need to make sure they are the same length as MODEL_ID and MODEL_SCALES
INIT_ROTS="[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]"
RAND_ROT_Z="[False, False, False]"
RAND_AXIS_ROT_RANGE="[0.0, 0.0, 0.0]"





# ========= Command Execution =========

python mani_skill2_real2sim/examples/jaco_get_water.py \
  -e "$ENV_ID" \
  -c "$CONTROL_MODE" \
  $ENABLE_VIEWER \
  -o "$OBS_MODE" \
  robot "$ROBOT_MODEL" \
  sim_freq "$SIM_FREQ" \
  control_freq "$CONTROL_FREQ" \
  scene_name "$SCENE_NAME" \
  rgb_overlay_mode "$OVERLAY_MODE" \
  rgb_overlay_path "$OVERLAY_PATH" \
  --model_ids "$MODEL_ID" \
  --model_scales "$MODEL_SCALES" \
  --init_xys "$INIT_XYS" \
  --init_rots "$INIT_ROTS" \
  --rand_rot_z "$RAND_ROT_Z" \
  --rand_axis_rot_range "$RAND_AXIS_ROT_RANGE"




# Example) Run the GetWaterCustomInScene environment
# python mani_skill2_real2sim/examples/jaco_get_water.py \
#   -e GetWaterCustomInScene-v0 \
#   -c arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_pos \  
#   --enable-sapien-viewer \                                     
#   -o rgbd \                                                    
#   robot google_robot_grip_camera \                                  
#   sim_freq @501 \                                              
#   control_freq @3 \                                            
#   scene_name frl_apartment_stage_simple \                     
#   rgb_overlay_mode debug \                                     
#   rgb_overlay_path data/real_inpainting/open_drawer_b0.png \   
