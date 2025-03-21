import opentamp
import subprocess
import random
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import io
import cv2
import matplotlib.pyplot as plt
import torch
from shapely.geometry import LineString

from matplotlib.patches import Wedge

import opentamp.policy_hooks.utils.policy_solver_utils as utils
from collections import OrderedDict

MOV_INCR = 10
NUM_OBSTACLES = 1
OBS_DIM = 7
class FloorplanEnv(gym.Env):
    def __init__(self):
        ## assemble the board, and initialize position of to the center of a random square
        # self.num_rows, self.num_columns = num_rows, num_columns
        # self.grid_graph = self.gen_grid_graph(num_rows, num_columns)
        # self.removed_edges = self.randomly_remove_edges(self.grid_graph, self.num_rows, self.num_columns)
        # self.add_doors_until_connected(self.grid_graph, self.removed_edges)
        # ## spawn to an initial position of a random square
        # init_square = random.randrange(self.num_rows * self.num_columns)
        # self.position = np.array([init_square // num_rows + 0.5, init_square % num_rows + 0.5])
        self.walls = []
        self.locations = []
        self.obstacles = []
        self.position = np.array([0.0, 0.0])
        self.target = np.array([0.0,0.0])
        #self.obstacle = np.array([0.0,0.0])
        self.action_space = spaces.Box(low=np.array([-0.25,-0.25, -1.5]), high=np.array([0.25, 0.25, 1.5]), shape=(3,), dtype='float32')
        self.observation_space = spaces.Box(low=np.array([-10.0]*OBS_DIM), high=np.array([10.0]*OBS_DIM), shape=(OBS_DIM,), dtype='float32')
        self.curr_state = np.concatenate([self.position, np.array([0.0])])
        self.low_bound = np.array([0.0, 0.0])
        self.high_bound = np.array([5.0, 5.0])
        self.constraint_viol = False
        self.render_mode = 'rgb_array'
        self.curr_angle = 0.0
        self.init_angle = 0.0
        self.door_loc = 0.0
        self.t = 0
        self.skolem_seq = []
        self.skolem_idx = None

    def compute_observation(self):
        # rotation_matrix = np.array([[np.cos(self.curr_angle), np.sin(self.curr_angle)], [-np.sin(self.curr_angle), np.cos(self.curr_angle)]])
        obs_arr = [self.position, np.array([self.curr_angle])]
        angles = np.arange(0, (33/16)*np.pi, np.pi/8)
        # for i in range(len(angles)-1):
            # min_dist = 4.0
            # for obs in self.obstacles:
        rel_pose = self.target - self.position

        angle_diff = (np.arctan2(rel_pose[1], rel_pose[0]) - self.curr_angle) % (2 * np.pi)

        if np.linalg.norm(rel_pose) < 2.0 and (angle_diff < np.pi/4 or angle_diff > 7/4 * np.pi):
            obs_arr.append(self.target)
        else:
            obs_arr.append(np.array([-10., -10.]))

        #rel_obs_pose = self.obstacle - self.position
        #angle_diff_obs = (np.arctan2(rel_obs_pose[1], rel_obs_pose[0]) - self.curr_angle) % (2 * np.pi)

        #if np.linalg.norm(rel_obs_pose) < 2.0 and (angle_diff_obs < np.pi/4 or angle_diff_obs > 7/4 * np.pi):
        #    obs_arr.append(self.obstacle)
        #else:
        obs_arr.append(np.array([-10., -10.]))

            # rel_angle = (np.arctan2(rel_pose[1], rel_pose[0])) % (2*np.pi)
            # if angles[i] <= rel_angle and rel_angle < angles[i+1]:
            #     if np.linalg.norm(rel_pose) < min_dist:
            #         min_dist = np.linalg.norm(rel_pose)
            # if min_dist < 4.0:
            #     obs_arr.append(np.array([min_dist]))
            # else:
            #     obs_arr.append(np.array([10.0]))  ## clearly distinct no-read

        return np.concatenate(obs_arr)

    def step(self, action):
        ## break movement up into 10, and check if you cross
        # rotation_matrix = np.array([[np.cos(self.curr_angle), -np.sin(self.curr_angle)], [np.sin(self.curr_angle), np.cos(self.curr_angle)]])
        # xy_action = rotation_matrix @ (action[:2])
        xy_action = action[:2]

        small_incr = xy_action / MOV_INCR
        for _ in range(10):
            if self.wall_collides_after_action(small_incr):
                break
            else:
                self.position += small_incr
                self.postion = np.minimum(self.position, np.array([5.0,5.0]))
                self.position = np.maximum(self.postion, np.array([0.0, 0.0]))

        self.curr_state[:2] = self.position
        self.curr_state[2] = (self.curr_angle + action[2]) % (2*np.pi)
        self.curr_angle = self.curr_state[2]

        reward = 0.0

        # for obstacle in self.obstacles:
        #if np.linalg.norm(self.position - self.obstacle) < 0.3:
        #    constraint_viol = 0.3 - np.linalg.norm(self.position - self.obstacle)
        #    if constraint_viol > self.max_violation:
        #        self.max_violation = constraint_viol
        #    self.constraint_viol = True

        #if np.linalg.norm(self.position - self.obstacle) < 0.7:
        #    reward -= 50 # huge penalty for obstacle avoidance, clearance around the obstacle

        observation = self.compute_observation()

        self.t += 1

        rel_pose = self.target - self.position

        angle_diff = (np.arctan2(rel_pose[1], rel_pose[0]) - self.curr_angle) % (2 * np.pi)

        if np.linalg.norm(rel_pose) < 2.0 and (angle_diff < np.pi/4 or angle_diff > 7/4 * np.pi):
            reward += -np.linalg.norm(self.target - self.position)
        else:
            reward += -2.0

        reward += self.compute_skolem_cost()

        # reward how far up you point the pointer
        # reward = np.dot(np.array([0.,1.]), np.array([np.cos(self.curr_state[2]), np.sin(self.curr_state[2])]))

        # reward -- penalty for distance, with constant away from target
        return observation, reward, self.t == 400, False, {'is_success': self.assess_goal(None, None) == 0.0}

    def compute_skolem_cost(self):
        # hit all subgoals
        if self.skolem_idx == len(self.skolem_seq):
            return 0.0

        curr_skolem = self.skolem_seq[self.skolem_idx]
        rel_skolem_pose = curr_skolem - self.position

        angle_diff = (np.arctan2(rel_skolem_pose[1], rel_skolem_pose[0]) - self.curr_angle) % (2 * np.pi)

        # advance to next skolem, if pointing at the curent one
        if np.linalg.norm(rel_skolem_pose) < 2.0 and (angle_diff < np.pi/8 or angle_diff > 15/8 * np.pi):
            self.skolem_idx += 1

        return max(-np.linalg.norm(curr_skolem - (self.position + np.array([np.cos(self.curr_angle), np.sin(self.curr_angle)]))), -2.0) - 2.0 * (len(self.skolem_seq) - self.skolem_idx)

    def compute_skolem_seq(self):
        box_lls = [np.array([1.25, 3.75]), np.array([3.75, 3.75]), np.array([3.75, 1.25]), np.array([1.25,1.25])]
        start_skolem = 0
        for i in range(4):
            if np.linalg.norm(box_lls[i] - self.position, ord=np.inf) < 1.25:
                start_skolem = i

        targ_skolem = 0
        for i in range(4):
            if np.linalg.norm(box_lls[i] - self.target, ord=np.inf) < 1.25:
                targ_skolem = i

        self.skolem_seq = []
        idx = start_skolem
        while idx != targ_skolem:
            self.skolem_seq.append(box_lls[idx])
            idx = (idx + 1) % 4

        self.skolem_seq.append(box_lls[targ_skolem])
        self.skolem_idx = 0


    def reset(self, seed=None):
        self.t = 0
        ## reset position + target
        # init_square = random.randrange(self.width * self.height)
        rand1 = np.random.rand(2,) * 5.0
        rand2 = np.random.rand(2,) * 5.0
        self.max_violation = 0.0
        while np.linalg.norm(rand1-rand2) < 3.:
            rand1 = np.random.rand(2,) * 5.0
            rand2 = np.random.rand(2,) * 5.0

        self.position = rand1

        self.curr_angle = random.random() * 2*np.pi
        self.init_angle = 0.0
        self.curr_state[:2] = self.position
        self.curr_state[2] = self.curr_angle
        self.constraint_viol = False

        self.compute_skolem_seq()

        return self.compute_observation(), {}

    def render(self, mode='rgb'):
        ## simply do the environment with an added dot for the current location of the objcet

        # plot sin wave
        plt.axis('off')
        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.axis('off')

        ##  draw all walls, locations, and obstacles
        for wall in self.walls:
            ax.plot(np.array([wall[0][0], wall[1][0]]), np.array([wall[0][1], wall[1][1]]), 'k')

        #for loc in self.locations:
        #    ax.scatter([loc[0]], [loc[1]], color='g')

        # for obs in self.obstacles:
        #ax.scatter([self.obstacle[0]], [self.obstacle[1]], color='r')
        #ax.add_patch(plt.Circle((self.obstacle[0], self.obstacle[1]), 0.3,  color='r',  alpha=0.3))

        ## indicate current location
        ax.scatter([self.position[0]], [self.position[1]], color='b')
        ax.scatter([self.target[0]], [self.target[1]], color='g')

        patches = []

        ang1 = (self.curr_angle * 360 / (2 * np.pi) - 45.) % 360.
        ang2 = (self.curr_angle * 360 / (2 * np.pi) + 45.) % 360.

        we = Wedge(self.position,2.,ang1,ang2,color='y', alpha=0.3)

        ax.add_patch(we)

        # ax.set(xlim=[2740, 2800], ylim=[5339740, 5339780])


        # define a function which returns an image as numpy array from figure
        def get_img_from_fig(fig, dpi=300):
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=dpi)
            buf.seek(0)
            img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
            buf.close()
            img = cv2.imdecode(img_arr, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (256, 256))  ## make image 64x64
            return img

        # you can get a high-resolution image as numpy array!!
        plot_img_np = get_img_from_fig(fig)
        fig.clf() # ensure no carry-over for plt
        plt.close()

        return plot_img_np


    def wall_collides_after_action(self, incr):
        return False
        # for wall in self.walls:
        #     line1 = LineString([self.position, self.position+incr])
        #     line2 = LineString(wall)
        #     if line1.intersects(line2):
        #         return True
        # return False ## TODO: implement collisoin logic, for now never collides
        # new_pos = self.position+incr
        # if (new_pos < 0.0).any():
        #     return True
        # elif (new_pos > np.array([self.width, self.height])).any():
        #     return True
        # else:
        #     return False


class FloorplanEnvWrapper(FloorplanEnv):
    def reset_to_state(self, state):
        self.position = state[:2]
        self.curr_state[:2] = self.position
        self.constraint_viol = False
        self.t = 0
        self.curr_angle = state[2]
        self.max_violation = 0.0
        observation = self.compute_observation()
        return observation

    def get_vector(self):
        state_vector_include = {
            'rob': ['pose', 'theta']
        }

        action_vector_include = {
            'rob': ['pose', 'theta']
        }

        target_vector_include = {
        }

        return state_vector_include, action_vector_include, target_vector_include

    def get_prim_choices(self, task_list=None):
        out = OrderedDict({})
        out[utils.TASK_ENUM] = sorted(['move_to_loc_same_room', 'sleep'])
        out[utils.DEST_PRED] = 2
        return out

    # reset without affecting the simulator
    def get_random_init_state(self):
        # self.doors = []
        # self.walls = []
        # # init_pose = random.random() * np.pi/2  # give random initial state between 0 and 90 degrees
        # init_pose = np.array([0.0,0.0,0.0])
        # # init_theta = np.array([random.random() * 2 * np.pi - np.pi]) ## random on -np.pi to np.pi
        # # init_vel = np.array([0.0])

        # is_valid = False
        # while not is_valid:
        #     proposal_targ = self.dist.sample().detach().numpy()

        #     if np.linalg.norm(proposal_targ) <= 4.0:
        #         continue

        #     rand = random.random()

        #     avg_val = torch.tensor(proposal_targ * rand)

        #     obstacle_dist = distros.Uniform(avg_val - torch.tensor([1.0, 1.0]),
        #                                     avg_val + torch.tensor([1.0, 1.0]))
        #     proposal_obs = obstacle_dist.sample().detach().numpy()

        #     if np.linalg.norm(proposal_targ-proposal_obs) < 1.5 or np.linalg.norm(proposal_obs) < 1.5 :
        #         continue

        #     is_valid = True

        # # by default, point at the obstacle at spawn
        # obstacle_abs_angle = np.arctan(proposal_obs[1]/proposal_obs[0]) if np.abs(proposal_obs[0]) > 0.001 else (np.pi/2 if proposal_obs[1]*proposal_obs[0]>0 else -np.pi/2)
        # obstacle_angle = obstacle_abs_angle if proposal_obs[0] >= 0  else (obstacle_abs_angle + np.pi if -np.pi/2 <= obstacle_abs_angle < 0 else obstacle_abs_angle - np.pi)
        # init_pose[2] = obstacle_angle

        ## initalize to center
        arr = self.reset()[0]

        return np.concatenate([arr[:2], np.array([0.0])])

    def postproc_im(self, base_im, s, t, cam_id):
        return base_im

    def sample_belief_true(self):
        #self.target = np.random.rand(2,) * 5.0
        self.obstacles = []

        #for i in range(NUM_OBSTACLES):
        #    obstacle = np.random.rand(2,) * 5.0
        #    while np.linalg.norm(self.target - self.obstacle) < 1.0 or np.linalg.norm(self.position - self.obstacle) < 1.0 :
        #        self.obstacle = np.random.rand(2,) * 5.0
        #    self.obstacles.append(obstacle)


        # while np.linalg.norm(self.target - self.obstacle) < 1.0 or np.linalg.norm(self.position - self.obstacle) < 1.0 :
        #     self.obstacle = np.random.rand(2,) * 5.0

        belief_true = {'targ': torch.tensor(self.target)}
        belief_true.update({'o0': torch.tensor(self.obstacle)})
        return belief_true

    def set_belief_true(self, belief_dict):
        self.belief_true = belief_dict
        #self.target = belief_dict['targ']

    def get_im(self):
        return self.render()

    def populate_sim_with_facts(self, problem):
        ## clear all prior info
        self.walls = []
        self.locations = []
        self.obstacles = []

        for param_name, param in problem.init_state.params.items():
            if param._type == 'Wall':
                self.walls.append((param.endpoint1.reshape(-1), param.endpoint2.reshape(-1)))
            elif param._type == 'DomainSettings':
                self.low_bound, self.high_bound = param.low_bound.reshape(-1), param.high_bound.reshape(-1)
                # populate with the four boundary walls
                self.walls.append((self.low_bound, np.array([self.high_bound[0], self.low_bound[1]])))
                self.walls.append((self.low_bound, np.array([self.low_bound[0], self.high_bound[1]])))
                self.walls.append((np.array([self.low_bound[0], self.high_bound[1]]), self.high_bound))
                self.walls.append((np.array([self.high_bound[0], self.low_bound[1]]), self.high_bound))
            elif param._type == 'Target':
                if param.name == 'targ':
                    self.target = param.value.reshape(-1)

                else:
                    self.locations.append(param.value.reshape(-1))
            elif param._type == 'Robot':
                self.position = param.pose[:,0].reshape(-1)

        rel_targ = self.target - self.position
        self.init_angle = np.arctan2(rel_targ[1], rel_targ[0])
        self.curr_angle = self.init_angle

    def return_dict_with_facts(self):
        return {
            'walls': [[list(w) for w in wall] for wall in self.walls],
            'locations': [list(w) for w in self.locations],
            'position': list(self.position),
            'target': list(self.target)
        }

    # determine whether or not a given state satisfies a goal condition
    def assess_goal(self, condition, state, targets=None, cont=None):

        # angle = self.curr_state[2]
        goal_rel_pose = self.target - self.position
        # # if pointing directly at the object

        ## add same room logic here
        if np.linalg.norm(goal_rel_pose, ord=np.inf) <= 0.7 and not self.constraint_viol:
            # print(self.target)
            # print(self.position)
            return 0.0
        else:
            return 1.0
        # return 0.0 ## always succeeds for now

    # determine whether constraints have been violated since last reset
    def assess_constraint_viol(self):
        if self.constraint_viol:
            return 1.0
        else:
            return 0.0

    def compute_stats_dict(self):
        return {'final_dist_to_targ': np.linalg.norm(self.target - self.position),
                    'max_constraint_violation': self.max_violation}
