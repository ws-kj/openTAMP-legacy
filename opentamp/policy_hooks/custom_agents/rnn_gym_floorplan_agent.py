import opentamp.policy_hooks.utils.policy_solver_utils as utils
from opentamp.policy_hooks.custom_agents.rnn_gym_agent import RnnGymAgent
from opentamp.core.util_classes.floorplan_predicates import *
from opentamp.core.util_classes.matrix import Vector2d
from opentamp.core.internal_repr.parameter import *

import numpy as np
import json
import random 

NUM_OBSTACLES = 1

class RnnGymFloorplanAgent(RnnGymAgent):
    def get_hl_info(self, state=None, targets=None, problem=None, domain=None, goal=None, cond=0, plan=None, act=0):
        # TODO: randomization logic for initial stuff...
        init_loc = np.random.rand(2, 1) * np.array([5., 5.]).reshape(2, 1) ## random init location in the grid
        
        new_rand_locs = []
        while len(new_rand_locs) < 1:
            prop_sample = np.random.rand(2) * 5
            if len(new_rand_locs) > 0:
                if np.linalg.norm(prop_sample-init_loc.reshape(-1,)) < 1.5:
                    continue
            else:
                if np.linalg.norm(prop_sample-init_loc.reshape(-1,)) < 3.0:
                    continue
            
            is_close = False
            for loc in new_rand_locs:
                if np.linalg.norm(prop_sample-loc) < 1.0:
                    is_close = True
            
            if is_close:
                continue

            new_rand_locs.append(prop_sample)
        
        # parameters for setting up a new floorplan domain

        rooms = [([0.0, 0.0], [5.0, 5.0])] ## room boundaries
        doors = [] ## locations
        self.door_loc = random.random() * 4 + 0.5
        self.gym_env.door_loc = self.door_loc
        walls = [] # list of endpoints
        # possible_obstacles = [[2.5, 1.5], [2.5, 2.5], [2.5, 3.5], [3.5, 1.5], [3.5, 2.5], [3.5, 3.5]]
        # random_num = random.randrange(10)
        obstacles = [np.random.rand(2,) * 5]*NUM_OBSTACLES # new_rand_locs[1:] # list of positions
        # if random_num < len(possible_obstacles):
        #     obstacles.append(possible_obstacles[random_num])
        locations = [(init_loc.reshape(-1,), 0)]
        target = ([4.,4.],0)
        connections = []
        bounds = (np.array([0.0,0.0]), np.array([5.0, 5.0]))

        new_obj_arr = []
        init_facts = []

        for room_idx, room in enumerate(rooms):
            new_obj_arr.append(
                {
                    'name': 'r'+str(room_idx),
                    'value': [0.0,0.0],
                    'low_bound': room[0],
                    'high_bound': room[1],
                    'type': 'Room'
                }
            )
        
        for door_idx, door in enumerate(doors):
            new_obj_arr.append(
                {
                    'name': 'd'+str(door_idx),
                    'value': door,
                    'type': 'Door'
                }
            )


        for wall_idx, wall in enumerate(walls):
            new_obj_arr.append(
                {
                    'name': 'w'+str(wall_idx),
                    'value': [0.0, 0.0],
                    'endpoint1': wall[0],
                    'endpoint2': wall[1],
                    'type': 'Wall'
                }
            )

        for obs_idx, obs in enumerate(obstacles):
            new_obj_arr.append(
                {
                    'name': 'o'+str(obs_idx),
                    'value': obs,
                    'belief': [1000,0,0,5.0,5.0],
                    'type': 'Obstacle'
                }
            )

        for loc_idx, loc in enumerate(locations):
            new_obj_arr.append(
                {
                    'name': 'l'+str(loc_idx),
                    'value': loc[0],
                    'type': 'Location'
                }
            )
            init_facts.append('(LocationInRoom l'+str(loc_idx)+' r'+str(loc[1])+' )')

        new_obj_arr.append(
            {
                'name': 'targ',
                'value': target[0],
                'type': 'Target',
                'belief': [1000,0,0,5.0,5.0]
            }
        )

        # init_facts.append('(LocationInRoom targ r'+str(target[1])+' )')


        for connect_idx, connection in enumerate(connections):
            init_facts.append('(DoorConnectsLocs d'+ str(connect_idx) +' l'+str(connection[0])+' l'+str(connection[1])+' )')
            init_facts.append('(DoorConnectsLocs d'+ str(connect_idx) +' l'+str(connection[1])+' l'+str(connection[0])+' )')


        # for obs_idx, _ in enumerate(obstacles):
        #     for loc_idx_1, _ in enumerate(locations):
        #         for loc_idx_2, _ in enumerate(locations):
        #             init_facts.append('(PathClear rob o'+str(obs_idx)+' l'+str(loc_idx_1)+' l'+str(loc_idx_2)+' )')

        # for obs_idx, _ in enumerate(obstacles):
        #     for loc_idx_1, _ in enumerate(locations):
        #             init_facts.append('(PathClear rob o'+str(obs_idx)+' l'+str(loc_idx_1)+' targ )')
        #             init_facts.append('(PathClear rob o'+str(obs_idx)+' targ l'+str(loc_idx_1)+' )')

        # initially true -- will be overridden depending on observations
        # if init_loc[0] < 2.5:
        init_facts.append('(RobotInRoom rob r0 )')
        init_facts.append('(TargetInRoom targ r0 )')
        # else:
        #     init_facts.append('(RobotInRoom rob r1 )')
        #     init_facts.append('(TargetInRoom targ r1 )')
        #     locations[0] = (init_loc.reshape(-1,), 1)

        new_obj_arr.append({
            'name': 'rob',
            'pose': locations[0][0],
            'theta': np.array([0.]),
            'type': 'Robot'
        })


        new_obj_arr.append({
            'name': 'domset',
            'value': [0.,0.],
            'low_bound': bounds[0],
            'high_bound': bounds[1],
            'type': 'DomainSettings'
        })

        new_obj_arr.append(
            {
                'name': 'vantage',
                'value': np.array([0., 0.]),
                'type': 'Vantage'
            }
        )


        # new_obj_dict = {'r1': room1, 'r2': room2, 'r3': room3, 'door1': door1, 'door2': door2, 'l0': l0, 'l1': l1, 'l2': l2, 'l3': l3, 'l4': l4, 'l5': l5, 'l6': l6, 'o1': o1}

        # add initial objects programmatically
        # problem.init_state.params.update(new_obj_dict)

        # new_obj_arr = [room1, room2, room3, door1, door2, l0, l1, l2, l3, l4, l5, l6, w1, w2, w3, w4, domain_settings, o1]

        # return ['(RobotInRoom rob r0 )', '(RobotAtLocation rob l0 )'] + init_facts, '(RobotAtLocation rob targ )', new_obj_arr, np.concatenate([locations[0][0], np.array([0.0])])
        return [] + init_facts, '(TaskComplete rob )', new_obj_arr, np.concatenate([locations[0][0], np.array([0.0])])


    def populate_sim_with_facts(self, problem):
        self.gym_env.populate_sim_with_facts(problem)
        self.curr_obs = self.gym_env.compute_observation()
        self.curr_state = self.gym_env.curr_state

        # rejection sample points nearby to init
        for i in range(problem.init_state.params['targ'].belief.samples.shape[0]):
            while np.linalg.norm(problem.init_state.params['targ'].belief.samples[i,:,0] - problem.init_state.params['rob'].pose[:,0]) < 1.0:
                problem.init_state.params['targ'].belief.samples[i,:,0] = torch.rand((2,)) * 5.
        
        for idx in range(NUM_OBSTACLES):
            for i in range(problem.init_state.params['o'+str(idx)].belief.samples.shape[0]):
                while np.linalg.norm(problem.init_state.params['o'+str(idx)].belief.samples[i,:,0] - problem.init_state.params['rob'].pose[:,0]) < 1.0:
                    problem.init_state.params['o'+str(idx)].belief.samples[i,:,0] = torch.rand((2,)) * 5.
