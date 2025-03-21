import copy
from collections import OrderedDict
import itertools
import numpy as np
import random
import time

import opentamp
from opentamp.core.internal_repr.plan import Plan
from opentamp.core.util_classes.openrave_body import *
from opentamp.pma.hl_solver import FFSolver
from opentamp.policy_hooks.utils.load_task_definitions import get_tasks, plan_from_str
from opentamp.policy_hooks.utils.policy_solver_utils import *
import opentamp.policy_hooks.utils.policy_solver_utils as utils


domain_file = opentamp.__path__._path[0] + "/domains/spot_domain/move_robot.domain"
mapping_file = opentamp.__path__._path[0] + "/policy_hooks/spot/spot_tasks"

N_OBJ = 5
END_TARGETS =[(0., 5.8), (0., 5.), (0., 4.), (2., 1.5),
                   (-2., 1.5),
                   (0.8, 1.5),
                   (-0.8, 1.5),
                   (-2.8, 1.5)]

def prob_file(descr=None):
    # return opentamp.__path__._path[0] + "/domains/robot_manipulation_domain/spot_probs/spot_prob_{0}.prob".format(N_OBJ)
    return opentamp.__path__._path[0] + "/domains/spot_domain/spot_probs/spot_nav_simple.prob"


def get_prim_choices(task_list=None):
    out = OrderedDict({})
    if task_list is None:
        out[utils.TASK_ENUM] = sorted(list(get_tasks(mapping_file).keys()))
    else:
        out[utils.TASK_ENUM] = sorted(list(task_list))

    out[utils.TARG_ENUM] = ['ROBOT_INIT_POSE', 'ROBOT_END_POSE']
    return out


def get_vector(config):
    state_vector_include = {
        'spot': ['pose', 'rotation', 'position', 'theta']
    }

    for obj in range(N_OBJ):
        state_vector_include['obj{0}'.format(obj)] = ['pose']

    action_vector_include = {
        'spot': ['pose', 'rotation', 'position', 'theta']
    }

    target_vector_include = {
        'target1': ['value', 'rotation'],
        'target2': ['value', 'rotation'],
    }

    return state_vector_include, action_vector_include, target_vector_include


def get_plans(use_tf=False):
    tasks = {'move_to': ['0: MOVE_TO SPOT ROBOT_INIT_POSE {0}']}
    task_ids = sorted(list(get_tasks(mapping_file).keys()))
    prim_options = get_prim_choices()
    plans = {}
    openrave_bodies = {}
    env = None
    params = None
    sess = None

    for task_ind, task in enumerate(task_ids):
        params = None        
        for targ_ind, targ in enumerate(prim_options[TARG_ENUM]):
            next_task_str = copy.deepcopy(tasks[task])
            new_task_str = []
            for step in next_task_str:
                new_task_str.append(step.format(targ))
            plan = plan_from_str(new_task_str, prob_file(), domain_file, env, openrave_bodies, params=params, sess=sess, use_tf=use_tf)

            params = plan.params
            if env is None:
                env = plan.env
                for param in list(plan.params.values()):
                    if hasattr(param, 'geom'):
                        if not hasattr(param, 'openrave_body') or param.openrave_body is None:
                            param.openrave_body = OpenRAVEBody(env, param.name, param.geom)
                        openrave_bodies[param.name] = param.openrave_body
            
            for j in range(len(prim_options[utils.TARG_ENUM])):
                    plans[(task_ids.index(task), i, j)] = plan

    return plans, openrave_bodies, env


def get_random_initial_state_vec(config, plans, dX, state_inds, conditions):
    return [np.zeros(dX)], [{'bin_target': np.zeros(3)}]

