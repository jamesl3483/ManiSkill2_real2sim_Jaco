"""
Multi-object version of GetWaterInSceneEnv.

Key additions
-------------
* `self.objs` (list of actors) instead of a single `self.obj`
* `_set_models`, `_load_models`, `_initialize_actors` updated for N objects
* Evaluation checks contact for *all* objects
* Reset options:
    options = {
        "model_ids": ["cup", "mug", "bottle"],      # list of IDs in your model DB
        "model_scales": {"cup":1.0, "mug":0.8},     # per-object scale (optional)
        "obj_init_options": {                       # optional per-object init opts
            "multi_obj_init": [
                {"init_xy": [0.10, 0.10]},
                {"init_xy": [0.20, 0.05]},
                {"init_xy": [0.30,-0.02], "init_rand_rot_z":True},
            ]
        }
    }
"""

from collections import OrderedDict
from typing import List, Optional

import numpy as np
import cv2
import sapien.core as sapien
from mani_skill2_real2sim import ASSET_DIR
from mani_skill2_real2sim.utils.registration import register_env
from mani_skill2_real2sim.utils.sapien_utils import get_entity_by_name
from transforms3d.euler import euler2quat
from mani_skill2_real2sim.utils.common import random_choice
from transforms3d.euler import euler2quat
from transforms3d.quaternions import axangle2quat, qmult
from mani_skill2_real2sim.utils.sapien_utils import (
    get_pairwise_contacts,
    compute_total_impulse,
)
from mani_skill2_real2sim.sensors.camera import CameraConfig

from .base_env import CustomOtherObjectsInSceneEnv, CustomSceneEnv
from .open_door_in_scene import OpenDoorInSceneEnv

# ---- import your base classes ----------------
from .open_door_in_scene import OpenDoorInSceneEnv


class GetWaterInSceneEnv(OpenDoorInSceneEnv):
    """Get-Water environment that can spawn multiple random objects."""

    # ------------- INITIALISATION -------------------------------------------------
    def __init__(
        self,
        force_advance_subtask_time_steps: int = 100,
        **kwargs,
    ):
        # --- per-object state ---
        self.objs: List[sapien.Actor] = []
        self.model_ids: List[str] = []
        self.model_scales: List[float] = []
        self.model_bbox_sizes: List[Optional[np.ndarray]] = []
        self.obj_init_options: dict[str, Any] = {}
        self.obj_init_options_list: List[dict[str, Any]] = []

        # misc
        self.force_advance_subtask_time_steps = force_advance_subtask_time_steps
        super().__init__(**kwargs)

    # ------------- SCENE CONFIG ---------------------------------------------------
    def _get_default_scene_config(self):
        cfg = super()._get_default_scene_config()
        cfg.contact_offset = 0.005  # avoid “false-positive” collisions
        return cfg

    # ------------- MODEL SELECTION ------------------------------------------------
    def _set_model(
        self,
        model_ids: Optional[List[str]],
        model_scales: Optional[dict[str, float]],
    ) -> bool:
        """
        Decide which models & scales to use this episode.
        Returns True if we need to recreate actors (reconfigure).
        """
        if model_ids is None:
            # pick one random object if user didn’t specify anything
            model_ids = [random_choice(list(self.model_db.keys()), self._episode_rng)]

        reconfigure = (model_ids != self.model_ids)
        self.model_ids = model_ids
        self.model_scales = []
        self.model_bbox_sizes = []

        for mid in self.model_ids:
            # scale
            scale = None
            if model_scales and mid in model_scales:
                scale = model_scales[mid]
            else:
                avail = self.model_db[mid].get("scales")
                scale = random_choice(avail, self._episode_rng) if avail else 1.0
            self.model_scales.append(scale)

            # bbox size
            if "bbox" in self.model_db[mid]:
                bbox = self.model_db[mid]["bbox"]
                size = (np.array(bbox["max"]) - np.array(bbox["min"])) * scale
                self.model_bbox_sizes.append(size)
            else:
                self.model_bbox_sizes.append(None)

        return reconfigure

    # ------------- MODEL CREATION -------------------------------------------------
    def _load_model(self):
        """Create SAPIEN actors for every model in self.model_ids."""
        self.objs.clear()
        # print("Loading objects:", self.model_ids)
        for mid, scale in zip(self.model_ids, self.model_scales):
            density = self.model_db[mid].get("density", 1000)
            obj = self._build_actor_helper(
                mid,
                self._scene,
                scale=scale,
                density=density,
                physical_material=self._scene.create_physical_material(
                    static_friction=self.obj_static_friction,
                    dynamic_friction=self.obj_dynamic_friction,
                    restitution=0.0,
                ),
                root_dir=self.asset_root,
            )
            obj.name = mid

            # print("Loaded object:", mid, "with scale", scale)
            obj.set_damping(0.1, 0.1)
            self.objs.append(obj)

    # ------------- LOAD ACTORS ----------------------------------------------------
    def _load_actors(self):
        super()._load_actors()      # loads robot + drawer
        # print("GetWaterInSceneEnv._load_actors")
        self._load_model()         # loads multiple objects

    # ------------- INITIALISE ACTORS ---------------------------------------------
    def _initialize_actors(self):
        """
        * Randomly places each object above the table.
        * Robot moved away; actors settled.
        """
        # initialise parent class first (moves robot etc.)
        super()._initialize_actors()

        # -------- per-object initialisation --------
        xy_list = self.obj_init_options.get(
            "init_xy",
            [[0.0, 0.0]] * len(self.objs),
        )
        # print("GetWaterInSceneEnv._initialize_actors: placing objects at", xy_list)
        quat_list = self.obj_init_options.get("init_rot_quat", [[1, 0, 0, 0]] * len(self.objs))
        rand_rot_z_list = self.obj_init_options.get("init_rand_rot_z", [False, False, False])
        rand_axis_rot_range_list = self.obj_init_options.get("init_rand_axis_rot_range", [0.0, 0.0, 0.0])

        for i, obj in enumerate(self.objs):
            # position
            xy = xy_list[i]
            # print("GetWaterInSceneEnv._initialize_actors: placing object at", xy)
            z = self.obj_init_options.get("init_z", self.scene_table_height) + 0.5
            # print("object z-height:", z)
            quat = quat_list[i]
            p = np.hstack([xy, z])
            q = quat

            # random z-rotation
            if rand_rot_z_list[i]:
                ori = self._episode_rng.uniform(0, 2 * np.pi)
                q = qmult(euler2quat(0, 0, ori), q)

            # random small axis rotation
            rot_rng = rand_axis_rot_range_list[i]
            if rot_rng > 0:
                axis = self._episode_rng.uniform(-1, 1, 3)
                axis /= max(np.linalg.norm(axis), 1e-6)
                angle = self._episode_rng.uniform(0, rot_rng)
                q = qmult(q, axangle2quat(axis, angle, True))

            obj.set_pose(sapien.Pose(p, q))
            obj.lock_motion(0, 0, 0, 1, 1, 0)  # lock rot X/Y

        # move robot far away; let objects settle
        self.agent.robot.set_pose(sapien.Pose([-10, 0, 0]))
        self._settle(0.5)

        # unlock and settle again
        for obj in self.objs:
            obj.lock_motion(0, 0, 0, 0, 0, 0)
            obj.set_velocity(np.zeros(3))
            obj.set_angular_velocity(np.zeros(3))
        self._settle(0.5)

    # ------------- RESET ----------------------------------------------------------
    def reset(self, seed=None, options=None):
        if options is None:
            options = {}
        options = options.copy()
        self.set_episode_rng(seed)
        # print("options:", options)

        # parse options
        self.obj_init_options = options.get("obj_init_options", {})
        # print("GetWaterInSceneEnv.reset: obj_init_options", self.obj_init_options)
        model_ids = options.get("model_id")
        # print("GetWaterInSceneEnv.reset: model_id", model_ids)
        model_scales = options.get("model_scales")

        reconfigure = options.get("reconfigure", False)
        if self._set_model(model_ids, model_scales):
            reconfigure = True
        options["reconfigure"] = reconfigure

        # run parent reset (loads/initialises everything)
        obs, info = super().reset(seed=self._episode_seed, options=options)

        self.drawer_link = get_entity_by_name(
            self.art_obj.get_links(), f"simple_{self.drawer_id}"
        )
        self.drawer_collision = self.drawer_link.get_collision_shapes()[0]
        # self._cameras()
        return obs, info

    # ------------- EPISODE STATS / EVAL ------------------------------------------
    def _initialize_episode_stats(self):
        self.cur_subtask_id = 0  # 0=open drawer, 1=place objs
        self.episode_stats = OrderedDict(
            qpos=0.0, is_drawer_open=False, has_contact=0
        )

    def evaluate(self, **kwargs):
        # drawer progress
        qpos = self.art_obj.get_qpos()[self.joint_idx]
        self.episode_stats["qpos"] = qpos
        self.episode_stats["is_drawer_open"] |= qpos >= 0.15

        # contact check for every object
        for obj in self.objs:
            infos = get_pairwise_contacts(
                self._scene.get_contacts(),
                obj,
                self.drawer_link,
                collision_shape1=self.drawer_collision,
            )
            if np.linalg.norm(compute_total_impulse(infos)) > 1e-6:
                self.episode_stats["has_contact"] += 1

        success = (
            self.cur_subtask_id == 1
            and qpos >= 0.05
            and self.episode_stats["has_contact"] >= len(self.objs)
        )
        return dict(success=success, episode_stats=self.episode_stats)

    # ------------- SUBTASK LOGIC --------------------------------------------------
    def advance_to_next_subtask(self):
        self.cur_subtask_id = 1

    def step(self, action):
        # force advance after certain steps
        if self._elapsed_steps >= self.force_advance_subtask_time_steps:
            self.advance_to_next_subtask()
        return super().step(action)

    def get_language_instruction(self, **kwargs):
        if self.cur_subtask_id == 0:
            return f"open {self.drawer_id} drawer"
        else:
            names = [self._get_instruction_obj_name(mid) for mid in self.model_ids]
            obj_str = ", ".join(names)
            return f"place {obj_str} into {self.drawer_id} drawer"

    def is_final_subtask(self):
        return self.cur_subtask_id == 1

    @property
    def cameras(self):
        print("GetWaterInSceneEnv.cameras: adding gripper camera")
        cams = super().cameras          # get existing camera list
        cams.append(                    # add a new one that follows the gripper
            CameraConfig(
                uid        = "gripper_cam",
                p          = [0.0, 0.0, 0.05],          # 5 cm in front of link origin
                q          = [0.7071, 0.0, 0.7071, 0.0],# face forward
                width      = 640,
                height     = 512,
                actor_uid  = "link_gripper",                 # ← name of the robot link
                intrinsic  = np.array(
                    [[425.0, 0, 320.0],
                     [0, 425.0, 256.0],
                     [0,   0,   1  ]]),
            )
        )
        return cams



# ------------------ REGISTRATION EXAMPLE ----------------------------------------
@register_env("GetWaterInScene-v0", max_episode_steps=200)
class GetWaterTopInSceneEnv(GetWaterInSceneEnv, CustomOtherObjectsInSceneEnv):
    DEFAULT_MODEL_JSON = "info_pick_custom_baked_tex_v1.json"
    drawer_ids = ["top", "middle", "bottom"]


@register_env("GetWaterCustomInScene-v0", max_episode_steps=200)
class GetWaterCustomInSceneEnv(GetWaterTopInSceneEnv):
    drawer_ids = ["cabinet"]

