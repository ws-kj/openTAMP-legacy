NUM_OBJS = 1
NUM_TARGS = 1

from datetime import datetime
import os
import os.path

import numpy as np

import opentamp
import opentamp.policy_hooks.utils.policy_solver_utils as utils
from opentamp.core.util_classes.namo_grip_predicates import ATTRMAP
# from opentamp.pma.namo_grip_solver import NAMOSolverOSQP as NAMOSolver
from opentamp.pma.toy_solver import ToySolver
from opentamp.policy_hooks.custom_agents.gym_agent import GymAgent
import opentamp.policy_hooks.gym_prob as prob
from opentamp.policy_hooks.utils.file_utils import LOG_DIR
from opentamp.policy_hooks.observation_models import *

from opentamp.envs.blank_gym_env import BlankEnvWrapper

import torch.nn.functional as F

BASE_DIR = opentamp.__path__._path[0] +  '/policy_hooks/'
EXP_DIR = BASE_DIR + 'experiments/'

prob.NUM_OBJS = NUM_OBJS
prob.NUM_TARGS = NUM_TARGS

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

def refresh_config(no=NUM_OBJS, nt=NUM_TARGS):
    # cost_wp_mult = np.ones((3 + 2 * NUM_OBJS))
    prob.NUM_OBJS = no
    prob.NUM_TARGS = nt
    prob.N_GRASPS = N_GRASPS
    prob.FIX_TARGETS = True

    config = {
        'num_conds': NUM_CONDS,
        'solver_type': 'adam', #'rmsprop',
        'base_weight_dir': 'namo_',
        'max_sample_queue': 5e2,
        'max_opt_sample_queue': 10,
        'task_map_file': "policy_hooks/namo/pointer_taskmapping",
        'prob': prob,
        'get_vector': prob.get_vector,
        'robot_name': 'pr2',
        'obj_type': 'can',
        'num_objs': no,
        'num_targs': nt,
        'attr_map': ATTRMAP,
        'agent_type': GymAgent,
        'gym_env_type': BlankEnvWrapper,
        'mp_solver_type': ToySolver,
        'll_solver_type': ToySolver,
        'meta_file': opentamp.__path__._path[0] + '/new_specs/pointer_domain/pointer_meta.json',
        'acts_file': opentamp.__path__._path[0] + '/new_specs/pointer_domain/pointer_acts.json',
        'prob_file': opentamp.__path__._path[0] + '/new_specs/pointer_domain/pointer_prob.json',
        'observation_model': NoVIPointerObservationModel,
        'n_dirs': N_DIRS,

        'state_include': [utils.STATE_ENUM],

        'obs_include': [#utils.LIDAR_ENUM,
                        # utils.MJC_SENSOR_ENUM,
                        utils.TARG_ENUM,
                        # utils.ONEHOT_GOAL_ENUM
                        # utils.TASK_ENUM,
                        # utils.END_POSE_ENUM,
                        # #utils.EE_ENUM,
                        # #utils.VEL_ENUM,
                        # utils.THETA_VEC_ENUM,
                        ],

        'cont_obs_include': [#utils.LIDAR_ENUM,
                        utils.MJC_SENSOR_ENUM,
                        utils.PAST_TARG_ENUM,
                        utils.TASK_ENUM,
                        utils.PAST_TASK_ENUM,
                        # utils.PAST_POINT_ENUM,
                        # utils.ONEHOT_GOAL_ENUM
                        # utils.TASK_ENUM,
                        # utils.END_POSE_ENUM,
                        # #utils.EE_ENUM,
                        # #utils.VEL_ENUM,
                        # utils.THETA_VEC_ENUM,
                        ],

        'prim_obs_include': [
                            #  utils.THETA_VEC_ENUM,
                            utils.MJC_SENSOR_ENUM,
                            utils.PAST_TASK_ENUM,
                            utils.PAST_POINT_ENUM,
                            utils.PAST_VAL_ENUM,
                            # utils.PAST_TARG_ENUM,
                            # utils.ONEHOT_GOAL_ENUM
                             ],

        'prim_out_include': list(prob.get_prim_choices().keys()),

        'sensor_dims': {
                # utils.OBJ_POSE_ENUM: 2,
                utils.TARG_ENUM: 2,
                utils.PAST_TARG_ENUM: 2,
                utils.PAST_COUNT_ENUM: 1,
                utils.PAST_POINT_ENUM: 1,
                utils.PAST_VAL_ENUM: 1,
                utils.PAST_TASK_ENUM: 1,
                # utils.LIDAR_ENUM: N_DIRS,
                utils.MJC_SENSOR_ENUM: BlankEnvWrapper().observation_space.shape[0],
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
            
        'visual': False,
        'time_limit': TIME_LIMIT,
        'success_to_replace': 1,
        'steps_to_replace': no * 50,
        'curric_thresh': -1,
        'n_thresh': -1,
        'expand_process': False,
        'descr': '',
        'her': False,
        'prim_decay': 0.95,
        'prim_first_wt': 1e1,
        # 'll_loss_fn': F.l1_loss,
        # 'cont_loss_fn': F.l1_loss,
    }

    #config['prim_obs_include'].append(utils.EE_ENUM)
    # for o in range(no):
    #     config['sensor_dims'][utils.OBJ_DELTA_ENUMS[o]] = 2
    #     config['sensor_dims'][utils.OBJ_ENUMS[o]] = 2
    #     config['sensor_dims'][utils.TARG_ENUMS[o]] = 2
    #     config['sensor_dims'][utils.TARG_DELTA_ENUMS[o]] = 2
    #     config['prim_obs_include'].append(utils.OBJ_DELTA_ENUMS[o])
    #     #config['prim_obs_include'].append(utils.OBJ_ENUMS[o])
    #     #config['prim_obs_include'].append(utils.TARG_ENUMS[o])
    #     config['prim_obs_include'].append(utils.TARG_DELTA_ENUMS[o])
    return config

config = refresh_config()
