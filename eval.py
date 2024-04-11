"""
Example script demo'ing robot primitive to solve a task
"""

import copy
import random
import os
import json
import time
import yaml
import numpy as np
import omnigibson as og
from omnigibson import object_states
from PIL import Image
from omnigibson.macros import gm
from omnigibson.utils.constants import CLASS_NAME_TO_CLASS_ID
from pddl_sim import pddlsim
from gpt4v import GPT4VAgent

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
PICK_OBJ_HEIGHT = 1.15
DOMAIN = "domains/boil_water_in_the_microwave/domain.pddl"
NUM_TRIALS = 30
CHECK_PRECONDITION = True
CHECK_EFFECT = True
MAX_NUM_ACTION = 50
MAX_TELEPORT_DIST = 1.0
MIN_TELEPORT_DIST = 2.5
OTHER_ACTION_SUCCESS_PROB = 1.0
PICK_SUCCESS_PROB = 0.5
PLACE_SUCCESS_PROB = 1.0
OPEN_SUCCESS_PROB = 0.8
CLOSE_SUCCESS_PROB = 0.8
LOG_DIR = "datadump/"

is_oracle = False

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

            distance = np.random.uniform(MIN_TELEPORT_DIST, MAX_TELEPORT_DIST)
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
    robot.tuck()
    run_sim()
    obj = env.task.object_scope[obj_name].wrapped_obj
    xyt = sample_teleport_pose_near_object(ap, obj)
    if xyt is None:
        print("there is no free space near the object -- action failed")
        return False
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
    return True


def turnon(obj, oracle=False):
    if not oracle:
        if not inview(obj):
            # breakpoint()
            print("turn on object failed -- unsatisfied hidden preconditions")
            return
        prob = random.random()
        if prob > OTHER_ACTION_SUCCESS_PROB:
            print("action failed randomly")
            return
    env.task.object_scope[obj].states[ToggledOn].set_value(True)
    run_sim()


def fill_sink(sink, oracle=False):
    if not oracle:
        if not inview(sink):
            # breakpoint()
            print("filling sink failed -- unsatisfied hidden preconditions")
            return
        prob = random.random()
        if prob > OTHER_ACTION_SUCCESS_PROB:
            print("action failed randomly")
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

        # there is some probability that the grasp action will fail
        prob = random.random()
        if prob > PICK_SUCCESS_PROB:
            print("action failed randomly")
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
        prob = random.random()
        if prob > OTHER_ACTION_SUCCESS_PROB:
            print("action failed randomly")
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
        prob = random.random()
        if prob > OPEN_SUCCESS_PROB:
            print("action failed randomly")
            return
    env.task.object_scope[obj].states[Open].set_value(True)
    run_sim()


def closeit(obj, oracle=False):
    if not oracle:
        if not inview(obj):
            # breakpoint()
            print("closing object failed -- unsatisfied hidden preconditions")
            return
        prob = random.random()
        if prob > CLOSE_SUCCESS_PROB:
            print("action failed randomly")
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
        prob = random.random()
        if prob > PLACE_SUCCESS_PROB:
            print("action failed randomly")
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


def format_action_params(action):
    params = action[1:]
    return [p.replace("-", ".") for p in params]


def translate_fact_to_question(fact):
    neg_suffix = " "
    if fact[0] == "not":
        # neg_suffix = " not "
        fact = fact[1:]

    predicate = fact[0]
    if predicate in ["inside", "inroom", "filled"]:
        return f"Is {fact[1].split('-')[0]}{neg_suffix}{predicate} {fact[2].split('-')[0]}?"
    elif predicate in ["insource", "inhand", "inview"]:
        return f"Is {fact[2].split('-')[0]}{neg_suffix}{predicate} {fact[1].split('-')[0]}?"
    elif predicate in ["handempty", "closed", "turnedon", "cooked"]:
        return f"Is {fact[1].split('-')[0]}{neg_suffix}{predicate}?"
    else:
        raise RuntimeError


def update_states_by_fact(states, fact):

    if fact[0] == "not":
        formatted_fact = f"(not ({fact[1]}"
        for param in fact[2:]:
            formatted_fact += f" {param}"
        formatted_fact += "))"
    else:
        formatted_fact = f"({fact[0]}"
        for param in fact[1:]:
            formatted_fact += f" {param}"
        formatted_fact += ")"
    if formatted_fact in states:
        states.remove(formatted_fact)
        print(f"removing {formatted_fact}")

    return states


def write_states_into_problem(states, previous_problem):
    prob = open(previous_problem).readlines()
    init_line = prob.index("    (:init\n")
    end_init_line = prob.index("    (:goal\n")
    before_init = prob[: init_line + 1]
    after_init = prob[end_init_line - 2 :]
    for state_id, state in enumerate(states):
        states[state_id] = f"        {state}\n"
    new_problem = before_init + states + after_init
    new_problem_name = "updated_problem.pddl"
    with open(new_problem_name, "w") as f:
        f.write("".join(new_problem))
    return new_problem_name


def check_states_and_update_problem(
    int_states, effs, pres, previous_problem, prev_states
):
    states = int_states.copy()
    unmatched_pres = []
    unmatched_effs = []

    is_state_updated_by_eff = False
    is_state_updated_by_pre = False

    facts_nl = []
    valid_facts = []
    facts = effs + pres
    pre_start_idx = 0

    for fact_id, fact in enumerate(facts):

        # TODO: need more discussions here
        # we assume these two predicates don't change over time or cannot be detected visually
        invalid_preds = [
            "inroom",
            "insource",
            "inhand",
            "handempty",
            "cooked",
            "filled",
            "inside",
            "turnedon",
        ]
        if fact[0] in invalid_preds or fact[1] in invalid_preds:
            continue

        fact_nl = translate_fact_to_question(fact)
        facts_nl.append(fact_nl)
        valid_facts.append(fact)

        if fact_id < len(effs):
            pre_start_idx += 1

    print(f"Questions to VLMs: ")
    print(f"current action effects: {facts_nl[:pre_start_idx]}")
    print(f"next action preconditions: {facts_nl[pre_start_idx:]}")

    if len(facts_nl) > 0:
        is_match_results = vlm_agent.ask(";".join(facts_nl), get_fpv_rgb())
    else:
        is_match_results = []
    for idx, is_match in enumerate(is_match_results):
        if idx < len(valid_facts) and (
            (("no" in is_match) and valid_facts[idx][0] != "not")
            or (("yes" in is_match) and valid_facts[idx][0] == "not")
        ):
            if idx < pre_start_idx:
                unmatched_effs.append(valid_facts[idx])
                is_state_updated_by_eff = True
            else:
                unmatched_pres.append(valid_facts[idx])
                is_state_updated_by_pre = True

    if (not is_state_updated_by_pre) and (not is_state_updated_by_eff):
        print("All facts match -- current states remain unchanged")
        updated_problem_file = write_states_into_problem(states, previous_problem)
    else:
        print(f"Here is a list of unmatched facts according to VLM's response --")
        print(f"unmatched effs: {unmatched_effs}")
        print(f"unmatched pres: {unmatched_pres}")

    if CHECK_EFFECT and is_state_updated_by_eff:
        print(
            "updating problem with previous states based on effects mismatch,"
            + " and will suggest to re-execute the current skill if needed"
        )
        updated_problem_file = write_states_into_problem(prev_states, previous_problem)
    elif CHECK_PRECONDITION and is_state_updated_by_pre:
        print(
            "updating states based on preconditions mismatch, and will suggest to replan if needed"
        )
        for unmatched_fact in unmatched_pres:
            print(f"Previous states: {states}")
            states = update_states_by_fact(states, unmatched_fact)
            print(f"Updated states: {states}")
        updated_problem_file = write_states_into_problem(states, previous_problem)

    # updated_problem_file = previous_problem
    breakpoint()
    return (is_state_updated_by_eff or is_state_updated_by_pre), updated_problem_file


def log_writer(message, log_file):
    with open(log_file, "a+") as f:
        f.write(message)


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
og.sim.set_lighting_mode("Default")
# run_sim_inf()
ap = StarterSemanticActionPrimitives(env)
planner = pddlsim(DOMAIN)

vlm_agent = GPT4VAgent()

# from claude3 import Claude3Agent
# vlm_agent = Claude3Agent()

init_problem_file = f"domains/{a_name}/problem.pddl"

exp_results = {"num_success": 0, "num_failed": 0}

trial_counter = 0
os.makedirs(LOG_DIR, exist_ok=True)
run_dir = os.path.join(LOG_DIR, str(time.time()))
os.makedirs(run_dir, exist_ok=False)

while trial_counter < NUM_TRIALS:
    trial_dir = os.path.join(run_dir, str(trial_counter))
    os.makedirs(trial_dir, exist_ok=False)
    env.reset()
    watch_robot()
    obj_held = None
    filled = None
    inside_relationships = []
    sim_counter = 0
    run_sim()
    action_counter = 0
    # TODO: the episode initialization has some bug -- hard code something for now
    env.task.object_scope["mug.n.04_1"].set_position_orientation(
        position=[-7.62, -3.76, 0.66], orientation=[0, 0, 0, 1]
    )
    run_sim()

    problem_file = init_problem_file
    terminate = False
    invalid_epi = False

    while not terminate:
        # planning
        plan = planner.plan(problem_file)
        print(f"Planning -- {plan}")
        if not plan:
            break
        # begin execution
        intermediate_states = planner.get_intermediate_states(
            problem_file, "pddl_output.txt"
        )
        for action_step, action in enumerate(plan):
            primitive = action[0]
            action_params = format_action_params(action)
            print(f"Executing action: {action}")
            # if CHECK_PRECONDITION:
            #     # 1. given an action, get all the preconditions formatted as NL
            #     # 2. ask each precondition using vlm
            #     # 3. if a precondition is not satisfied, modify that in the intermediate state
            #     # 4. breakout the loop, use intermediate state to generate new problem file
            #     current_states = intermediate_states[action_step]
            #     preconditions = planner.get_preconditions_by_action(action)
            #     unmatch, problem_file = check_states_and_update_problem(
            #         current_states, preconditions, problem_file, replan=True
            #     )
            #     if unmatch:
            #         break

            if primitive == "find":
                # if not goto(action_params[1], oracle=is_oracle):
                #     # there is something wrong with the skill, not the agent's fault
                #     invalid_epi = True
                goto(action_params[1], oracle=is_oracle)
            elif primitive == "openit":
                openit(action_params[1], oracle=is_oracle)
            elif primitive == "grasp":
                grasp(action_params[1], oracle=is_oracle)
            elif primitive == "fillsink":
                fill_sink(action_params[1], oracle=is_oracle)
            elif primitive == "fill":
                fill(action_params[1], action_params[2], oracle=is_oracle)
            elif primitive == "placein":
                place_with_predicate(
                    action_params[1], action_params[2], Inside, oracle=is_oracle
                )
            elif primitive == "closeit":
                closeit(action_params[1], oracle=is_oracle)
            elif primitive == "microwave_water":
                turnon(action_params[1], oracle=is_oracle)
            else:
                raise RuntimeError
            action_counter += 1

            if CHECK_EFFECT or CHECK_PRECONDITION:
                current_states = intermediate_states[action_step + 1]
                # current action effect
                effects = planner.get_effects_by_states(
                    intermediate_states[action_step], current_states
                )
                # next action precondition
                if len(plan) > action_step + 1:
                    preconditions = planner.get_preconditions_by_action(
                        plan[action_step + 1]
                    )
                else:
                    preconditions = []

                unmatch, problem_file = check_states_and_update_problem(
                    current_states,
                    effects,
                    preconditions,
                    problem_file,
                    intermediate_states[action_step],
                )
                if unmatch:
                    break

            if (
                action_step == len(plan) - 1
                or invalid_epi
                or action_counter >= MAX_NUM_ACTION
            ):
                terminate = True
                break

    # predefined
    # goto("cabinet.n.01_1", oracle=is_oracle)
    # openit("cabinet.n.01_1", oracle=is_oracle)
    # goto("mug.n.04_1", oracle=is_oracle)
    # grasp("mug.n.04_1", oracle=is_oracle)
    # goto("sink.n.01_1", oracle=is_oracle)
    # fill_sink("sink.n.01_1", oracle=is_oracle)
    # fill("mug.n.04_1", "sink.n.01_1", oracle=is_oracle)
    # goto("microwave.n.02_1", oracle=is_oracle)
    # openit("microwave.n.02_1", oracle=is_oracle)
    # place_with_predicate("mug.n.04_1", "microwave.n.02_1", Inside, oracle=is_oracle)
    # closeit("microwave.n.02_1", oracle=is_oracle)
    # turnon("microwave.n.02_1", oracle=is_oracle)

    # check success condition
    if not invalid_epi:
        if (
            env.task.object_scope["mug.n.04_1"]
            .states[Filled]
            .get_value(get_system("water"))
            and ("mug.n.04_1", "microwave.n.02_1") in inside_relationships
            and env.task.object_scope["microwave.n.02_1"].states[ToggledOn].get_value()
        ):
            print("success")
            exp_results["num_success"] += 1
        else:
            print("failed")
            exp_results["num_failed"] += 1
        with open("exp_results.json", "w") as f:
            json.dump(exp_results, f)
        trial_counter += 1

print("=" * 30)
print("SUMMARY:")
print(f"Number of success: {exp_results['num_success']}")
print(f"Number of failure: {exp_results['num_failed']}")
print("=" * 30)
