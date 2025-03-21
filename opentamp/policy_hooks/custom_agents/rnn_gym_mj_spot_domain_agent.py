import opentamp.policy_hooks.utils.policy_solver_utils as utils
from opentamp.policy_hooks.custom_agents.rnn_gym_agent import RnnGymAgent
from opentamp.core.util_classes.floorplan_predicates import *
from opentamp.core.util_classes.matrix import Vector2d
from opentamp.core.internal_repr.parameter import *

import numpy as np
import json
import random 

class RnnGymMJSpotDomainAgent(RnnGymAgent):
    def get_hl_info(self, state=None, targets=None, problem=None, domain=None, goal=None, cond=0, plan=None, act=0):
        new_obj_arr = []
        
        new_obj_arr.append({
            'name': 'rob',
            'pose': [0.0]*10,
            'type': 'Robot'
        })


        new_obj_arr.append({
            'name': 'loc',
            'value': [1.,1., 0.05],
            'type': 'Loc'
        })

        # new_obj_dict = {'r1': room1, 'r2': room2, 'r3': room3, 'door1': door1, 'door2': door2, 'l0': l0, 'l1': l1, 'l2': l2, 'l3': l3, 'l4': l4, 'l5': l5, 'l6': l6, 'o1': o1}

        # add initial objects programmatically
        # problem.init_state.params.update(new_obj_dict)

        # new_obj_arr = [room1, room2, room3, door1, door2, l0, l1, l2, l3, l4, l5, l6, w1, w2, w3, w4, domain_settings, o1]

        targ_nav = np.dot(np.random.rand(2,), np.array([0.3, 0.4])) + np.array([.8, -0.2])

        # return ['(RobotInRoom rob r0 )', '(RobotAtLocation rob l0 )'] + init_facts, '(RobotAtLocation rob targ )', new_obj_arr, np.concatenate([locations[0][0], np.array([0.0])])
        return [], '(GripperAtLocation rob loc ) (GripperClosed rob )', new_obj_arr, np.concatenate([np.array([.15, 0.]), np.random.rand(7,) * 0.05 + np.array([0., 0., -np.pi/3, np.pi/3, 0., np.pi/2, 0.]), np.array([-np.pi/4]), np.array([targ_nav[0],targ_nav[1],0.8])])


    def populate_sim_with_facts(self, problem):
        pass
        # self.gym_env.populate_sim_with_facts(problem)
        # self.curr_obs = self.gym_env.compute_observation()
        # self.curr_state = self.gym_env.curr_state

