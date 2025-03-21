NUM_OBJS = 1
NUM_TARGS = 1

from datetime import datetime
import os
import os.path

import numpy as np

from collections import OrderedDict

import opentamp
import opentamp.policy_hooks.utils.policy_solver_utils as utils
from opentamp.core.util_classes.namo_grip_predicates import ATTRMAP
# from opentamp.pma.namo_grip_solver import NAMOSolverOSQP as NAMOSolver
from opentamp.pma.custom_solvers.toy_solver import ToySolver
from opentamp.policy_hooks.custom_agents.rnn_gym_floorplan_agent_no_obstacle import RnnGymFloorplanAgent
from opentamp.policy_hooks.custom_agents.gym_agent import GymAgent
# import opentamp.new_specs.floorplan_domain_belief.gym_prob as prob
from opentamp.policy_hooks.utils.file_utils import LOG_DIR
from opentamp.policy_hooks.observation_models import *

from opentamp.envs.will_floorplan_env import FloorplanEnvWrapper

import torch.nn.functional as F
from opentamp.policy_hooks.core_agents.tamp_agent import ACTION_SCALE
from opentamp.policy_hooks.utils.policy_solver_utils import *

BASE_DIR = opentamp.__path__._path[0] +  '/policy_hooks/'
EXP_DIR = BASE_DIR + 'experiments/'

# prob.NUM_OBJS = NUM_OBJS
# prob.NUM_TARGS = NUM_TARGS

NUM_CONDS = 1 # Per rollout server
NUM_PRETRAIN_STEPS = 20
NUM_PRETRAIN_TRAJ_OPT_STEPS = 1
NUM_TRAJ_OPT_STEPS = 1
N_SAMPLES = 10
N_TRAJ_CENTERS = 1
HL_TIMEOUT = 600
OPT_WT_MULT = 5e2
N_ROLLOUT_SERVERS = 34 # 58
N_ALG_SERVERS = 0
N_OPTIMIZERS = 0
N_DIRS = 16
N_GRASPS = 4
TIME_LIMIT = 14400
OBS_DIM = 7

## gets the task nums for the plan, repeating the last action if necessary
def get_task_nums(plan, agent):
    active_anums = []
    i=0
    tasks = agent.encode_plan(plan)
    for a_num in range(len(plan.actions)):
        active_anums.append(tasks[a_num])
        i+=1
        if i > 19:
            break
    while i < 20:
        active_anums.append(active_anums[-1])
        i+=1
    return active_anums


def build_hist_info(path):
    if len(path) > 19:
        path = path[:19]

    mjc_obs_array = []
    mjc_obs_array = torch.tensor([[0.]*OBS_DIM] + [list(s.get(MJC_SENSOR_ENUM)[-1,:]) for s in path])
    mjc_obs_array = torch.flatten(mjc_obs_array.T)
    mjc_obs_array = [x.item() for x in list(mjc_obs_array)]

    return [len(path),
        np.array([0.]),
        sum([1.0 if s.task[0] == 1 else 0.0 for s in path]),
        sum([1.0 if s.task[0] == 0 else 0.0 for s in path]),
        (path[-1].task)[0],
        [0.] + [s.task[0] for s in path],
        mjc_obs_array] if path \
    else [len(path), np.array([0.]), 0, 0, -1.0, [0.], mjc_obs_array]

def build_aux_info(plan, conditioned_targ, a_num_idx):
    if a_num_idx <= len(conditioned_targ)-1:
        targ_pred = conditioned_targ[a_num_idx]
    else:
        targ_pred = plan.params['targ'].value[:,0]
    return targ_pred

## take path object use it for each
# def sample_fill_method(path, plan, agent, x0, conditioned_targ):
#     if len(path) > 19:
#         path = path[:19]

#     # Remove observation actions for easier imitation
#     active_anums = []
#     for a_num in range(len(plan.actions)):
#         active_anums.append(a_num)

#     ## populate the sample with the entire plan
#     # a_num = 0
#     # plan.params['rob'].actions

#     for i in range(20):
#         ## computing featured used in the plan
#         if i > len(plan.actions)-1:
#             break ## NEED TO REPLAY LAST ACTION! CUZ NOT DOING THE STOP THING ANYMORE!
#         a_num_idx = min(i, len(plan.actions)-1)
#         if a_num_idx > 0:
#             # prior_st = plan.actions[active_anums[a_num_idx-1]].active_timesteps[0]
#             # past_targ = plan.params['target1'].pose[:, prior_st]
#             past_targ = np.array([3.0, 3.0])
#             past_ang = np.arctan(np.array([past_targ[1]])/np.array([past_targ[0]])) \
#                 if not np.any(np.isnan(np.arctan(np.array([past_targ[1]])/np.array([past_targ[0]])))) \
#                     else np.pi/2
#             past_ang *= ACTION_SCALE
#         else:
#             past_targ = np.array([0., 0.])
#             past_ang = np.array([0.])

#         if a_num_idx <= len(conditioned_targ)-1:
#             targ_pred = conditioned_targ[a_num_idx]
#         else:
#             targ_pred = plan.params['targ'].value[:,0]

#         mjc_obs_array = []
#         mjc_obs_array = torch.tensor([[0.]*OBS_DIM] + [list(s.get(MJC_SENSOR_ENUM)[-1,:]) for s in path])
#         mjc_obs_array = torch.flatten(mjc_obs_array.T)
#         mjc_obs_array = [x.item() for x in list(mjc_obs_array)]

#         print('Targ pred: ', targ_pred)

#         ## run the action
#         new_path, x0 = agent.run_action(plan,
#                     active_anums[a_num_idx],
#                     x0,
#                     [np.array([0.0, 0.0,])],
#                     tasks[active_anums[a_num_idx]],
#                     st,
#                     reset=True,
#                     save=True,
#                     record=True,
#                     hist_info=[len(path),
#                                 past_ang,
#                                 sum([1 if (s.task)[0] == 1 else 0 for s in path]),
#                                 sum([1 if (s.task)[0] == 0 else 0 for s in path]),
#                                 (path[-1].task)[0] if len(path) > 0 else -1.0,
#                                 [0.] + [s.task[0] for s in path],
#                                 mjc_obs_array],
#                     aux_info=targ_pred)

#         path.extend(new_path)

## take path object, record historical info for stuff
# def rollout_fill_method(path, agent):
#     if len(path) > 19:
#         path = path[:19]

#     mjc_obs_array = []
#     mjc_obs_array = torch.tensor([[0.]*OBS_DIM] + [list(s.get(MJC_SENSOR_ENUM)[-1,:]) for s in path])
#     mjc_obs_array = torch.flatten(mjc_obs_array.T)
#     mjc_obs_array = [x.item() for x in list(mjc_obs_array)]

#     agent.store_hist_info([len(path),
#                             np.array([0.]),
#                             sum([1.0 if s.task[0] == 1 else 0.0 for s in path]),
#                             sum([1.0 if s.task[0] == 0 else 0.0 for s in path]),
#                             (path[-1].task)[0],
#                             [0.] + [s.task[0] for s in path],
#                             mjc_obs_array]) if path \
#     else agent.store_hist_info([len(path), np.array([0.]), 0, 0, -1.0, [0.], mjc_obs_array])

## populate the planning with custom skolem values
def skolem_populate_fcn(plan, repln=False):
    if not repln:
        plan.params['vantage'].value = plan.params['rob'].pose[:,0].copy().reshape((2, 1))
    else:
        plan.params['vantage'].value = plan.params['targ'].value.copy()

def refresh_config(no=NUM_OBJS, nt=NUM_TARGS):

    config = {
        'robot_name': 'pr2',  ## used to identify parameter
        'attr_map': ATTRMAP, ## used to get object dims??? why is this necessary?
        'agent_type': RnnGymFloorplanAgent,
        'gym_env_type': FloorplanEnvWrapper,
        'mp_solver_type': ToySolver, #? todo document
        'll_solver_type': ToySolver, #? todo document
        'meta_file': opentamp.__path__._path[0] + '/new_specs/will_test_domain/floorplan_meta.json',
        'acts_file': opentamp.__path__._path[0] + '/new_specs/will_test_domain/floorplan_acts.json',
        'prob_file': opentamp.__path__._path[0] + '/new_specs/will_test_domain/floorplan_prob.json',
        'observation_model': ParticleFilterTargetObservationModel,

        'state_include': [utils.STATE_ENUM],

        'obs_include': [#utils.LIDAR_ENUM,
                        utils.MJC_SENSOR_ENUM,
                        # utils.MJC_SENSOR_ENUM,
                        # utils.PAST_ANG_ENUM,
                        utils.TASK_ENUM,
                        utils.DEST_PRED
                        # utils.IM_ENUM,
                        # utils.PAST_COUNT_ENUM,
                        # utils.PAST_TASK_ENUM,
                        # utils.ANG_ENUM,
                        # utils.PAST_TASK_ARR_ENUM,
                        # utils.ONEHOT_GOAL_ENUM
                        # utils.TASK_ENUM,
                        # utils.END_POSE_ENUM,
                        # #utils.EE_ENUM,
                        # #utils.VEL_ENUM,
                        # utils.THETA_VEC_ENUM,
                        ],

        # 'recur_obs_include': [
        #      utils.PAST_TASK_ARR_ENUM,
        #      utils.PAST_MJCOBS_ARR_ENUM
        # ],

        'cont_obs_include': [#utils.LIDAR_ENUM,
                        utils.MJC_SENSOR_ENUM,
                        # utils.PAST_ANG_ENUM,
                        utils.TASK_ENUM,
                        utils.PAST_RECUR_ENUM,
                        utils.PAST_TASK_ENUM
                        # utils.PAST_TASK_ARR_ENUM,
                        # utils.PAST_MJCOBS_ARR_ENUM
                        # utils.PAST_TASK_ENUM,
                        # utils.PAST_POINT_ENUM,
                        # utils.ONEHOT_GOAL_ENUM
                        # utils.TASK_ENUM,
                        # utils.END_POSE_ENUM,
                        # #utils.EE_ENUM,
                        # #utils.VEL_ENUM,
                        # utils.THETA_VEC_ENUM,
                        ],

        'cont_recur_obs_include': [
            utils.PAST_RECUR_ENUM,
        ],

        'prim_obs_include': [
                            #  utils.THETA_VEC_ENUM,
                            utils.MJC_SENSOR_ENUM,
                            utils.PAST_RECUR_ENUM,
                            utils.PAST_TASK_ENUM
                            # utils.PAST_TASK_ENUM,
                            # utils.PAST_TASK_ARR_ENUM,
                            # utils.PAST_MJCOBS_ARR_ENUM,
                            # utils.PAST_VAL_ENUM,
                            # utils.PAST_TARG_ENUM,
                            # utils.ONEHOT_GOAL_ENUM
                             ],

        'cont_out_include': [utils.DEST_PRED],

        'prim_recur_obs_include': [
            utils.PAST_RECUR_ENUM,
        ],

        'prim_out_include': list(FloorplanEnvWrapper().get_prim_choices().keys()),

        'sensor_dims': {
                # utils.OBJ_POSE_ENUM: 2,
                utils.DEST_PRED: 2,
                utils.PAST_ANG_ENUM: 1,
                utils.TARG_ENUM: 2,
                utils.PAST_TARG_ENUM: 2,
                utils.PAST_COUNT_ENUM: 1,
                utils.PAST_POINT_ENUM: 1,
                utils.PAST_VAL_ENUM: 1,
                utils.PAST_TASK_ENUM: 1,
                utils.PAST_TASK_ARR_ENUM: 20,
                utils.PAST_MJCOBS_ARR_ENUM: 20 * FloorplanEnvWrapper().observation_space.shape[0],
                utils.PAST_RECUR_ENUM: 20 + 20 * FloorplanEnvWrapper().observation_space.shape[0],
                # utils.LIDAR_ENUM: N_DIRS,
                utils.MJC_SENSOR_ENUM: FloorplanEnvWrapper().observation_space.shape[0],
                utils.IM_ENUM: 256 * 256 * 3,
                # utils.EE_ENUM: 2,
                # utils.END_POSE_ENUM: 2,
                # utils.GRIPPER_ENUM: 1,
                # utils.VEL_ENUM: 2,
                # utils.THETA_ENUM: 1,
                # utils.THETA_VEC_ENUM: 2,
                # utils.GRASP_ENUM: N_GRASPS,
                # utils.GOAL_ENUM: 2*no,
                # utils.ONEHOT_GOAL_ENUM: no*NUM_TARGS,
                # utils.INGRASP_ENUM: no,
                # utils.TRUETASK_ENUM: 2,
                # utils.TRUEOBJ_ENUM: no,
                # utils.TRUETARG_ENUM: len(prob.END_TARGETS),
                # utils.ATGOAL_ENUM: no,
                # utils.FACTOREDTASK_ENUM: len(list(prob.get_prim_choices().keys())),
                # utils.INIT_OBJ_POSE_ENUM: 2,
            },

        'hist_len': 2, ## see if this survives TAMPAGENT refactor
        ## document what these overriding methods need to do...
        'build_hist_info': build_hist_info,
        'build_aux_info': build_aux_info,
        'get_task_nums': get_task_nums,
        'skolem_populate_fcn': skolem_populate_fcn,
        'permute_hl': 1, ## see if this survives TAMPAGENT refactor
        # 'll_loss_fn': 'GaussianNLLLoss',
        # 'cont_loss_fn': F.l1_loss,
    }

    return config

config = refresh_config()
