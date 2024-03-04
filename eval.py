"""
Example script demo'ing robot primitive to solve a task
"""

import os
import yaml
import numpy as np
import omnigibson as og
from omnigibson import object_states
from PIL import Image
from omnigibson.macros import gm

# from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives, StarterSemanticActionPrimitiveSet
# Don't use GPU dynamics and use flatcache for performance boost
gm.USE_GPU_DYNAMICS = True
# gm.ENABLE_FLATCACHE = True
from omnigibson.action_primitives.starter_semantic_action_primitives import (
    StarterSemanticActionPrimitives,
    StarterSemanticActionPrimitiveSet,
    RobotCopy,
    PlanningContext,
)
import transforms3d as tf
import quaternion as quat
from omnigibson.utils.object_state_utils import sample_cuboid_for_predicate
import omnigibson.utils.transform_utils as T
from omnigibson.utils.motion_planning_utils import (
    plan_base_motion,
    plan_arm_motion,
    plan_arm_motion_ik,
    set_base_and_detect_collision,
    detect_robot_collision_in_sim,
)
from omnigibson.object_states import *
from omnigibson.action_primitives.action_primitive_set_base import ActionPrimitiveError
from omnigibson.utils.bddl_utils import (
    OmniGibsonBDDLBackend,
    BDDLEntity,
    BEHAVIOR_ACTIVITIES,
    BDDLSampler,
)
from omnigibson.systems import (
    get_system,
    is_physical_particle_system,
    is_visual_particle_system,
)

PREDICATE_SAMPLING_Z_OFFSET = 0.02
MAX_ATTEMPTS_FOR_SAMPLING_POSE_WITH_OBJECT_AND_PREDICATE = 1000
MAX_ATTEMPTS_FOR_SAMPLING_POSE_NEAR_OBJECT = 100000
PICK_OBJ_HEIGHT = 2
obj_held = None
filled = None


def yaw_to_quaternion(yaw):
    """
    Convert yaw (rotation around the vertical axis) to quaternion.

    Parameters:
    - yaw: Yaw angle in radians.

    Returns:
    - quaternion: A 4-element numpy array representing the quaternion.
    """
    # Calculate half angles
    yaw_half = 0.5 * yaw

    # Quaternion components
    w = np.cos(yaw_half)
    x = 0.0
    y = 0.0
    z = np.sin(yaw_half)

    # Construct quaternion
    quaternion = np.array([x, y, z, w])

    return quaternion


def rotate_x(initial_quaternion, angle_degrees):
    # Convert the angle to radians
    angle_radians = np.radians(angle_degrees)

    # Create a quaternion representing the rotation
    rotation_quaternion = np.quaternion(
        np.cos(angle_radians / 2), np.sin(angle_radians / 2), 0, 0
    )

    # Perform quaternion multiplication to apply the rotation
    new_quaternion = rotation_quaternion * quat.as_quat_array(initial_quaternion)

    return quat.as_float_array(new_quaternion)


def rotate_z(initial_quaternion, angle_degrees):
    # Convert the angle to radians
    angle_radians = np.radians(angle_degrees)

    rotation_quaternion = np.quaternion(
        np.cos(angle_radians / 2), 0, 0, np.sin(angle_radians / 2)
    )

    # Perform quaternion multiplication to apply the rotation
    new_quaternion = rotation_quaternion * quat.as_quat_array(initial_quaternion)

    return quat.as_float_array(new_quaternion)


def inspect():
    while True:
        og.sim.step()


def run_sim(step=20):
    for _ in range(step):
        og.sim.step()
        # Test: object in hand might be
    dummy_action = np.zeros((11))
    env.step(dummy_action)
    global sim_counter
    Image.fromarray(og.sim.viewer_camera.get_obs()["rgb"], "RGBA").save(
        os.path.join(debug_path, str(sim_counter) + ".png")
    )
    sim_counter += 1


def goto(obj_name="can_of_soda_89"):
    # obj = scene.object_registry("name", obj_name)
    obj = env.task.object_scope[obj_name].wrapped_obj
    xyt = sample_teleport_pose_near_object(ap, obj)
    # pos, orien = obj.get_position_orientation()
    pos = np.empty([3])
    pos[2] = robot_init_z
    pos[1] = xyt[1]
    pos[0] = xyt[0]
    orien = yaw_to_quaternion(xyt[2])
    robot.set_position_orientation(pos, orien)
    global obj_held
    if obj_held:
        print("there is an object in hand -- moving this object as well")
        obj_held.set_position([pos[0], pos[1], PICK_OBJ_HEIGHT])
    watch_robot()
    lookat(obj_name)
    robot.tuck()
    run_sim()


def turnon(obj):
    env.task.object_scope[obj].states[ToggledOn].set_value(True)
    run_sim(step=100)


def lookat(obj_name="can_of_soda_89"):
    obj = env.task.object_scope[obj_name].wrapped_obj
    target_obj_pose = obj.get_position_orientation()
    # ap = StarterSemanticActionPrimitives(env)
    head_q = ap._get_head_goal_q(target_obj_pose)
    head_action = ap.robot.get_joint_positions()
    for qid, cid in enumerate(ap.robot.camera_control_idx):
        head_action[cid] = -head_q[qid]

    robot.set_joint_positions(head_action)


def watch_robot():
    og.sim.viewer_camera.set_position_orientation(
        position=np.array([0, 0, 3.5]) + robot.get_position(),
        orientation=rotate_z(rotate_x(robot.get_orientation(), 90), -15),
    )


def fill_sink(sink):
    env.task.object_scope[sink].states[ToggledOn].set_value(True)
    run_sim(step=40)
    env.task.object_scope[sink].states[ToggledOn].set_value(False)
    run_sim()


def grasp(obj_name="can_of_soda_89"):
    global obj_held
    if obj_held:
        print("there is already an object in hand")
        raise RuntimeError
        # TODO: return False ?

    # check if object is in the field of view -- assuming vision-based grasping

    obj = env.task.object_scope[obj_name].wrapped_obj
    # obj_held = env.task.object_scope['mug.n.04_1'].wrapped_obj
    robot_pose = robot.get_position()
    obj.set_position_orientation(
        position=[robot_pose[0], robot_pose[1], PICK_OBJ_HEIGHT],
        orientation=[0, 0, 0, 1],
    )
    # obj.set_orientation([0,0,0,1])
    obj_held = obj
    obj.disable_gravity()
    # global
    # if filled
    # system.remove_all_particles()
    run_sim()
    return True


def fill(container, source, liquid="water"):
    global filled
    if not obj_held:
        print("you have to hold a container to fill a liquid")
        raise RuntimeError
    if filled:
        print("there are already something in the container")
        raise RuntimeError
    system = get_system(liquid)

    container_obj = obj_held
    place_with_predicate(container, source, Inside)
    assert container_obj.states[Filled].set_value(system, True)
    # obj_held.states[Filled].get_value(get_system('water'))
    run_sim()
    # import pdb; pdb.set_trace()
    # TODO: something wrong here
    # assert container_obj.states[Filled].get_value(system)
    # import pdb; pdb.set_trace()
    filled = system
    # container_obj.states[Filled].get_value(system)
    # import pdb; pdb.set_trace()
    system.remove_all_particles()
    grasp(container)


def openit(obj):
    env.task.object_scope[obj].states[Open].set_value(True)
    run_sim()


def closeit(obj):
    env.task.object_scope[obj].states[Open].set_value(False)
    run_sim()


def sample_teleport_pose_near_object(ap, obj, pose_on_obj=None, **kwargs):
    with PlanningContext(ap.robot, ap.robot_copy, "simplified") as context:
        for _ in range(MAX_ATTEMPTS_FOR_SAMPLING_POSE_NEAR_OBJECT):
            # if pose_on_obj is None:
            pos_on_obj = ap._sample_position_on_aabb_side(obj)
            pose_on_obj = [pos_on_obj, np.array([0, 0, 0, 1])]

            distance = np.random.uniform(1.0, 2.0)
            yaw = np.random.uniform(-np.pi, np.pi)
            avg_arm_workspace_range = np.mean(ap.robot.arm_workspace_range[ap.arm])
            pose_2d = np.array(
                [
                    pose_on_obj[0][0] + distance * np.cos(yaw),
                    pose_on_obj[0][1] + distance * np.sin(yaw),
                    yaw + np.pi - avg_arm_workspace_range,
                ]
            )
            # Check room

            obj_rooms = (
                obj.in_rooms
                if obj.in_rooms
                else [
                    ap.env.scene._seg_map.get_room_instance_by_point(pose_on_obj[0][:2])
                ]
            )

            if obj_rooms == [None]:
                print("object not in any room.")
                continue
            if (
                ap.env.scene._seg_map.get_room_instance_by_point(pose_2d[:2])
                not in obj_rooms
            ):
                print("Candidate position is in the wrong room.")
                continue
            if set_base_and_detect_collision(
                context, ap._get_robot_pose_from_2d_pose(pose_2d)
            ):
                print("Candidate position failed collision test.")
                continue
            # import pdb; pdb.set_trace()
            return pose_2d
        print("Could not find valid position near object.")
        raise ActionPrimitiveError(
            ActionPrimitiveError.Reason.SAMPLING_ERROR,
            "Could not find valid position near object.",
            {
                "target object": obj.name,
                "target pos": obj.get_position(),
                "pose on target": pose_on_obj,
            },
        )


def _sample_pose_with_object_and_predicate(
    predicate, held_obj, target_obj, near_poses=None, near_poses_threshold=None
):
    """
    Returns a pose for the held object relative to the target object that satisfies the predicate

    Args:
        predicate (object_states.OnTop or object_states.Inside): Relation between held object and the target object
        held_obj (StatefulObject): Object held by the robot
        target_obj (StatefulObject): Object to sample a pose relative to
        near_poses (Iterable of arrays): Poses in the world frame to sample near
        near_poses_threshold (float): The distance threshold to check if the sampled pose is near the poses in near_poses

    Returns:
        2-tuple:
            - 3-array: (x,y,z) Position in the world frame
            - 4-array: (x,y,z,w) Quaternion orientation in the world frame
    """
    pred_map = {object_states.OnTop: "onTop", object_states.Inside: "inside"}

    for _ in range(MAX_ATTEMPTS_FOR_SAMPLING_POSE_WITH_OBJECT_AND_PREDICATE):
        _, _, bb_extents, _ = held_obj.get_base_aligned_bbox()
        sampling_results = sample_cuboid_for_predicate(
            pred_map[predicate], target_obj, bb_extents
        )
        if sampling_results[0][0] is None:
            continue
        sampled_bb_center = sampling_results[0][0] + np.array(
            [0, 0, PREDICATE_SAMPLING_Z_OFFSET]
        )
        sampled_bb_orn = sampling_results[0][2]

        # Get the object pose by subtracting the offset
        # sampled_obj_pose = T.pose2mat((sampled_bb_center, sampled_bb_orn)) @ T.pose_inv(T.pose2mat((bb_center_in_base, [0, 0, 0, 1])))
        sampled_obj_pose = T.pose2mat((sampled_bb_center, sampled_bb_orn))
        # Check that the pose is near one of the poses in the near_poses list if provided.
        # if near_poses:
        #     sampled_pos = np.array([sampled_obj_pose[0]])
        #     if not np.any(np.linalg.norm(near_poses - sampled_pos, axis=1) < near_poses_threshold):
        #         continue

        # Return the pose
        return T.mat2pose(sampled_obj_pose)

    # If we get here, sampling failed.
    print(
        "Could not find a position to put this object in the desired relation to the target object"
    )


def place_with_predicate(
    obj_in_hand_name="can_of_soda_89", obj_name="trash_can_85", predicate=OnTop
):
    """
    Yields action for the robot to navigate to the object if needed, then to place it

    Args:
        obj (StatefulObject): Object for robot to place the object in its hand on
        predicate (object_states.OnTop or object_states.Inside): Determines whether to place on top or inside

    Returns:
        np.array or None: Action array for one step for the robot to place or None if place completed
    """
    # Update the tracking to track the object.
    # self._tracking_object = obj
    obj = env.task.object_scope[obj_name].wrapped_obj
    # ap = StarterSemanticActionPrimitives(env)
    obj_in_hand = env.task.object_scope[obj_in_hand_name].wrapped_obj
    # TODO: verify it here
    global obj_held
    if obj_in_hand != obj_held:
        print("You need to be grasping the object first to place it somewhere.")
        raise ActionPrimitiveError(
            ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
            "You need to be grasping the object first to place it somewhere.",
        )
    # Sample location to place object
    obj_pose = _sample_pose_with_object_and_predicate(predicate, obj_in_hand, obj)
    # hand_pose = ap._get_hand_pose_for_object_pose(obj_pose)
    # obj_in_hand.wake()
    obj_in_hand.set_position_orientation(obj_pose[0], obj_pose[1])
    obj_in_hand.enable_gravity()
    # obj_in_hand.sleep()
    # yield from self._navigate_if_needed(obj, pose_on_obj=hand_pose)
    # yield from self._move_hand(hand_pose)
    # yield from self._execute_release()

    # if self._get_obj_in_hand() is not None:
    #     raise ActionPrimitiveError(
    #         ActionPrimitiveError.Reason.EXECUTION_ERROR,
    #         "Could not release object - the object is still in your hand",
    #         {"object": self._get_obj_in_hand().name}
    #     )

    # if not obj_in_hand.states[predicate].get_value(obj):
    #     raise ActionPrimitiveError(
    #         ActionPrimitiveError.Reason.EXECUTION_ERROR,
    #         "Failed to place object at the desired place (probably dropped). The object was still released, so you need to grasp it again to continue",
    #         {"dropped object": obj_in_hand.name, "target object": obj.name}
    #     )
    # run_sim()
    global filled
    run_sim()
    if filled:
        assert obj_held.states[Filled].set_value(filled, True)
        # import pdb; pdb.set_trace()
        # obj_held.states[Filled].get_value(filled)
        # obj_held.states[Filled].set_value(filled, False)
        filled = None
    obj_held = None

    # yield from self._move_hand_upward()


# Load the config
# config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
config_filename = os.path.join(og.example_config_path, "fetch_behavior.yaml")
config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

# Update it to run a grocery shopping task
config["scene"]["scene_model"] = "Beechwood_0_int"
config["scene"]["load_task_relevant_only"] = True
config["scene"]["not_load_object_categories"] = ["ceilings"]

a_name = "boil_water_in_the_microwave"
# find task by name
# for scenes in os.listdir('/home/xiaohan/OmniGibson/omnigibson/data/og_dataset/scenes/'):
#     for js in os.listdir(os.path.join('/home/xiaohan/OmniGibson/omnigibson/data/og_dataset/scenes/', scenes, 'json')):
#         if a_name in js:
#             print (scenes)
#             print (js)

config["task"] = {
    "type": "BehaviorTask",
    "activity_name": a_name,
    "activity_definition_id": 0,
    "activity_instance_id": 0,
    "predefined_problem": None,
    "online_object_sampling": False,
}

# initialize temp dump dir for visualization
import shutil
import os

debug_path = "datadump/third_person"
shutil.rmtree(debug_path, ignore_errors=True)
os.makedirs(debug_path, exist_ok=True)
sim_counter = 0

# Load the environment
env = og.Environment(configs=config)
scene = env.scene
robot = env.robots[0]

robot_init_z = robot.get_position()[2]
# Allow user to move camera more easily
og.sim.enable_viewer_camera_teleoperation()
# run_sim_inf()
watch_robot()
run_sim()
ap = StarterSemanticActionPrimitives(env)
# goto()
# grasp()
# goto("trash_can_85")
# place_with_predicate()
# import pdb; pdb.set_trace()
# inspect()
goto("cabinet.n.01_1")

openit("cabinet.n.01_1")

goto("mug.n.04_1")
# fill('mug.n.04_1', 'sink.n.01_1')

grasp("mug.n.04_1")
goto("sink.n.01_1")
# place_with_predicate('mug.n.04_1', 'sink.n.01_1', Inside)

fill_sink("sink.n.01_1")

fill("mug.n.04_1", "sink.n.01_1")
# env.task.object_scope['water.n.06_1'].disable_gravity()
# env.task.object_scope['sink.n.01_1'].states[ToggledOn].set_value(False)
# run_sim()

# grasp('mug.n.04_1')
goto("microwave.n.02_1")

openit("microwave.n.02_1")


place_with_predicate("mug.n.04_1", "microwave.n.02_1", Inside)

closeit("microwave.n.02_1")

turnon("microwave.n.02_1")


# env.task.object_scope['mug.n.04_1'].states[Filled].get_value(get_system('water'))
# inspect()
