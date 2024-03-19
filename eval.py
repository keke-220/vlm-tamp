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
from omnigibson.utils.constants import CLASS_NAME_TO_CLASS_ID

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
PICK_OBJ_HEIGHT = 1.4
obj_held = None
filled = None
inside_relationships = []


def get_class_name_from_class_id(target_class_id):
    for class_name, class_id in CLASS_NAME_TO_CLASS_ID.items():
        if class_id == target_class_id:
            return class_name
    return None


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
        # raise ActionPrimitiveError(
        #     ActionPrimitiveError.Reason.SAMPLING_ERROR,
        #     "Could not find valid position near object.",
        #     {
        #         "target object": obj.name,
        #         "target pos": obj.get_position(),
        #         "pose on target": pose_on_obj,
        #     },
        # )
        return None

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


def get_fpv_rgb():
    return robot.get_obs()["fetch:eyes_Camera_sensor"]["rgb"]


def get_seg_semantic():
    return robot.get_obs()["fetch:eyes_Camera_sensor"]["seg_semantic"]


def get_seg_instance():
    return robot.get_obs()["fetch:eyes_Camera_sensor"]["seg_instance"]


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


def inview(obj_name):
    obj = env.task.object_scope[obj_name].wrapped_obj
    seg = get_seg_instance()
    instances = np.unique(seg)
    for instance_id in instances:
        if obj == scene.objects[instance_id - 1]:
            print("object in view!!!!!")
            return True
    print("object not in view")
    return False


def inspect():
    try:
        while True:
            og.sim.step()
    except KeyboardInterrupt:
        print("Exiting ...")


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


def goto(obj_name="can_of_soda_89", oracle=False):

    # preconditions
    # TODO:the object and robot should be in the same room?

    # execution
    obj = env.task.object_scope[obj_name].wrapped_obj
    xyt = sample_teleport_pose_near_object(ap, obj)
    if xyt is None:
        print ("there is no free space near the object -- action failed")
        return
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
    return


def turnon(obj, oracle=False):
    if not oracle:
        if not inview(obj):
            # breakpoint()
            print("turn on object failed -- unsatisfied hidden preconditions")
            return
    env.task.object_scope[obj].states[ToggledOn].set_value(True)
    run_sim(step=100)


def fill_sink(sink, oracle=False):
    if not oracle:
        if not inview(sink):
            # breakpoint()
            print("filling sink failed -- unsatisfied hidden preconditions")
            return
    env.task.object_scope[sink].states[ToggledOn].set_value(True)
    run_sim(step=40)
    env.task.object_scope[sink].states[ToggledOn].set_value(False)
    run_sim()


def grasp(obj_name="can_of_soda_89", oracle=False):

    # preconditions

    # check if hand is empty
    global obj_held
    if not oracle:
        if obj_held:
            print(
                "there is already an object in hand -- unsatisfied hidden preconditions"
            )
            return
            # TODO: return False ?

        # check if object is in the field of view -- assuming vision-based grasping
        if not inview(obj_name):
            # breakpoint()
            print(
                "grasp object failed because of the object not in the view -- unsatisfied hidden preconditions"
            )
            return

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

    # manually remove this if it exists in our relationship tracker
    global inside_relationships
    for in_obj, recep in inside_relationships:
        if in_obj == obj_name:
            inside_relationships.remove((in_obj, recep))
            break

    return


def fill(container, sink, liquid="water", oracle=False):
    global filled
    if not oracle:
        if not obj_held:
            print(
                "you have to hold a container to fill a liquid -- unsatisfied hidden preconditions"
            )
            return
        if filled:
            print(
                "there are already something in the container -- unsatisfied hidden preconditions"
            )
            return
        if not inview(sink):
            # breakpoint()
            print(
                "filling container near source failed -- unsatisfied hidden preconditions"
            )
            return
    # system = get_system(liquid)

    container_obj = obj_held
    place_with_predicate(container, sink, OnTop, oracle=True)
    while not container_obj.states[Filled].get_value(get_system(liquid)):
        assert container_obj.states[Filled].set_value(get_system(liquid), True)
        run_sim()
    # assert container_obj.states[Filled].get_value(get_system(liquid))
    # obj_held.states[Filled].get_value(get_system('water'))
    run_sim()
    # import pdb; pdb.set_trace()
    # TODO: something wrong here
    # assert container_obj.states[Filled].get_value(system)
    # import pdb; pdb.set_trace()
    filled = liquid
    # container_obj.states[Filled].get_value(system)
    # import pdb; pdb.set_trace()
    get_system(liquid).remove_all_particles()
    grasp(container, oracle=True)


def openit(obj, oracle=False):

    # check if object is in the field of view -- effects
    if not oracle:
        if not inview(obj):
            # breakpoint()
            print("opening object failed -- unsatisfied hidden preconditions")
            return

    env.task.object_scope[obj].states[Open].set_value(True)
    run_sim()


def closeit(obj, oracle=False):
    if not oracle:
        if not inview(obj):
            # breakpoint()
            print("closing object failed -- unsatisfied hidden preconditions")
            return
    env.task.object_scope[obj].states[Open].set_value(False)
    run_sim()


def place_with_predicate(
    obj_in_hand_name="can_of_soda_89",
    obj_name="trash_can_85",
    predicate=OnTop,
    oracle=False,
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
    if not oracle:
        if obj_in_hand != obj_held:
            print(
                "You need to be grasping the object first to place it somewhere. -- unsatified hidden preconditions"
            )
            return
        if not inview(obj_name):
            # breakpoint()
            print("placing object failed -- unsatisfied hidden preconditions")
            return
        if (
            predicate == Inside
            and not env.task.object_scope[obj_name].states[Open].get_value()
        ):
            print(
                "placing object failed because the receptacle is closed-- unsatisfied hidden preconditions"
            )
            return
    # Sample location to place object
    # while (
    #     env.task.object_scope["mug.n.04_1"]
    #     .states[Inside]
    #     .get_value(env.task.object_scope["microwave.n.02_1"])
    # ):
    obj_pose = _sample_pose_with_object_and_predicate(predicate, obj_in_hand, obj)
    obj_in_hand.set_position_orientation(obj_pose[0], obj_pose[1])
    # TODO: this line manually set the object relationships -- which should not happen
    obj_in_hand.enable_gravity()
    run_sim()

    if predicate == Inside:
        # manually add this into our inside relationship tracker due to issues with the simulator
        global inside_relationships
        inside_relationships.append((obj_in_hand_name, obj_name))

    global filled
    if filled:
        while not obj_held.states[Filled].get_value(get_system(filled)):
            assert obj_held.states[Filled].set_value(get_system(filled), True)
            run_sim()
        # assert obj_held.states[Filled].get_value(filled)
        filled = None
    obj_held = None


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

# Load the environment
env = og.Environment(configs=config)
scene = env.scene
robot = env.robots[0]

robot_init_z = robot.get_position()[2]
# Allow user to move camera more easily
og.sim.enable_viewer_camera_teleoperation()
# run_sim_inf()
ap = StarterSemanticActionPrimitives(env)

is_oracle = False

num_success = 0
num_failed = 0

for _ in range(100):
    env.reset()
    watch_robot()
    obj_held = None
    filled = None
    inside_relationships = []
    sim_counter = 0
    # TODO: the episode initialization has some bug -- hard code something for now
    env.task.object_scope["mug.n.04_1"].set_position_orientation(
        position=[-7.62, -3.76, 0.66], orientation=[0, 0, 0, 1]
    )
    run_sim()

    # begin execution
    goto("cabinet.n.01_1", oracle=is_oracle)

    openit("cabinet.n.01_1", oracle=is_oracle)
    goto("mug.n.04_1", oracle=is_oracle)
    # fill('mug.n.04_1', 'sink.n.01_1')

    grasp("mug.n.04_1", oracle=is_oracle)
    goto("sink.n.01_1", oracle=is_oracle)
    # place_with_predicate('mug.n.04_1', 'sink.n.01_1', Inside)

    fill_sink("sink.n.01_1", oracle=is_oracle)
    fill("mug.n.04_1", "sink.n.01_1", oracle=is_oracle)
    # env.task.object_scope['water.n.06_1'].disable_gravity()
    # env.task.object_scope['sink.n.01_1'].states[ToggledOn].set_value(False)
    # run_sim()

    # grasp('mug.n.04_1')
    goto("microwave.n.02_1", oracle=is_oracle)

    openit("microwave.n.02_1", oracle=is_oracle)

    place_with_predicate("mug.n.04_1", "microwave.n.02_1", Inside, oracle=is_oracle)

    closeit("microwave.n.02_1", oracle=is_oracle)

    turnon("microwave.n.02_1", oracle=is_oracle)
    # env.task.object_scope['mug.n.04_1'].states[Filled].get_value(get_system('water'))
    # inspect()

    # check success condition
    if (
        env.task.object_scope["mug.n.04_1"].states[Filled].get_value(get_system("water"))
        and ("mug.n.04_1", "microwave.n.02_1") in inside_relationships
        and env.task.object_scope["microwave.n.02_1"].states[ToggledOn].get_value()
    ):
        print("success")
        num_success += 1
    else:
        print("failed")
        num_failed += 1

print ("="*30)
print ("SUMMARY:")
print (f"Number of success: {num_success}")
print (f"Number of failure: {num_failed}")
print ("="*30)
