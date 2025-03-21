from gym import Env
from gym import spaces
import random
import numpy as np
import torch
import torch.distributions as distros

from opentamp.policy_hooks.core_agents.tamp_agent import *

from opentamp.policy_hooks.utils.policy_solver_utils import *

from opentamp.envs.gym_env_nav_belief import GymEnvNav

OBS_DIM = 21
NUM_OBSTACLES = 1

X_BOUNDS = np.array([-1., 15.])
Y_BOUNDS = np.array([-4., 4.])

class GymEnvNavTheta(GymEnvNav):
    def __init__(self, deterministic=False):
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(3,), dtype='float32')
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(OBS_DIM,), dtype='float32')
        self.curr_state = np.array([0.0]*(7+2*NUM_OBSTACLES))
        self.curr_obs = np.array([0.0]*OBS_DIM)
        self.obs_dist, self.target_dist = self.assemble_dist()
        self.curr_angle = 0.0
        self.belief_true = {}
        self.constraint_viol = False

    def assemble_dist(self):
        # weights = torch.tensor([0.5,0.5])
        # locs = torch.tensor([[6., 0],
        #                      [8., 0]])
        # scales = torch.tensor([1.0, 1.0])
        # cat_dist = distros.Categorical(probs=weights)
        # stack_eye = torch.tile(torch.eye(2).unsqueeze(dim=0), dims=(2, 1, 1))
        # stack_scale = torch.tile(scales.unsqueeze(dim=1).unsqueeze(dim=2), dims=(1, 2, 2))
        # cov_tensor = stack_eye * stack_scale
        # batched_multivar = distros.MultivariateNormal(loc=locs, covariance_matrix=cov_tensor)
        # obs_dist = distros.MixtureSameFamily(cat_dist, batched_multivar)

        obs_dist = distros.Uniform(torch.tensor([5., -4.]), torch.tensor([9., 4.]))

        # weights = torch.tensor([0.5,0.5])
        # locs = torch.tensor([[12., 1.],
        #                      [12., -1.]])
        # scales = torch.tensor([1.0, 1.0])
        # cat_dist = distros.Categorical(probs=weights)
        # stack_eye = torch.tile(torch.eye(2).unsqueeze(dim=0), dims=(2, 1, 1))
        # stack_scale = torch.tile(scales.unsqueeze(dim=1).unsqueeze(dim=2), dims=(1, 2, 2))
        # cov_tensor = stack_eye * stack_scale
        # batched_multivar = distros.MultivariateNormal(loc=locs, covariance_matrix=cov_tensor)
        # target_dist = distros.MixtureSameFamily(cat_dist, batched_multivar)

        target_dist = distros.Uniform(torch.tensor([10., -4.]), torch.tensor([12., 4.]))

        # dist = distros.Normal(torch.tensor([0.0, 0.0]), 1.0)
        return obs_dist, target_dist

    def compute_observation(self):
        obstacle_rel_distances = []
        obstacle_mod_angles = []
        curr_idx = 7

        for i in range(NUM_OBSTACLES):
            obstacle_rel_pose = self.curr_state[curr_idx:curr_idx+2] - self.curr_state[:2]
            obstacle_rel_distance = np.linalg.norm(obstacle_rel_pose, ord=2)
            obstacle_angle = self.compute_angle(obstacle_rel_pose)
            obstacle_rel_angle = obstacle_angle - self.curr_state[2] ## relative angle calculated with respect to CAMERA ANGLE
            obstacle_mod_angle = obstacle_rel_angle % (2*np.pi)
            obstacle_rel_distances.append(obstacle_rel_distance)
            obstacle_mod_angles.append(obstacle_mod_angle)
            curr_idx += 2


        # obstacle_rel_pos = (self.curr_state[7:9] - self.curr_state[:2]) * 1 
        # # obstacle_abs_angle = np.arctan(obstacle_rel_pos[1]/obstacle_rel_pos[0]) if np.abs(obstacle_rel_pos[0]) > 0.001 else (np.pi/2 if obstacle_rel_pos[1]*obstacle_rel_pos[0]>0 else -np.pi/2)
        # obstacle_rel_distance = np.linalg.norm(obstacle_rel_pos, ord=2)
        # # spot_abs_angle = np.arctan(action[1]/action[0]) if action∂[0] > 0.001 else (np.pi/2 if action[1]>0 else -np.pi/2)
        
        # # making formula globally true at all theta (correcting for angle readings behind)
        # obstacle_angle = self.compute_angle(obstacle_rel_pos)
        # # spot_angle = spot_abs_angle if action[0] >= 0 else (spot_abs_angle + np.pi if -np.pi/2 <= spot_abs_angle < 0 else spot_abs_angle - np.pi)
        
        # # relative angle of obstacle with respect to spot camera
        # obstacle_rel_angle = obstacle_angle - self.curr_state[2]

        ## rotate the relative pose to be in the frame of the SPOT
        # rot_matrix = np.array([[np.cos(spot_angle),np.sin(spot_angle)],[-np.sin(spot_angle),np.cos(spot_angle)]])
        # obstacle_rel_pos_spot_frame = np.dot(rot_matrix, obstacle_rel_pos)

        # lidar_obs = np.array([8.0] * 8)
        # lidar_list = [(np.arange(-np.pi/4, np.pi/4, np.pi/16)[i], np.arange(-np.pi/4, np.pi/2, np.pi/16)[i+1]) for i in range(8)]
        
        # # formulas only valid on -pi/2 to pi/2
        # for detect_idx, theta_thresh in enumerate(lidar_list):
        #     if theta_thresh[0] <= obstacle_rel_angle < theta_thresh[1] or \
        #         theta_thresh[0] <= obstacle_rel_angle + 2*np.pi < theta_thresh[1] or \
        #         theta_thresh[0] <= obstacle_rel_angle - 2*np.pi < theta_thresh[1]:
        #         lidar_obs[detect_idx] = obstacle_rel_distance

        cam_angle = (self.curr_state[2]+np.pi)%(2*np.pi) - np.pi
        # if np.abs(cam_angle - obstacle_angle) <= np.pi/4 and np.linalg.norm(obstacle_rel_pos) <= 6.0:
        # obs_view =  obstacle_rel_pos
        # else:
        #     obs_view = np.array([-10.0, -10.0])

        target_rel_pos = (self.curr_state[3:5] - self.curr_state[:2]) * 1 
        # target_abs_angle = np.arctan(target_rel_pos[1]/target_rel_pos[0]) if np.abs(target_rel_pos[0]) > 0.001 else (np.pi/2 if target_rel_pos[1]*target_rel_pos[0]>0 else -np.pi/2)
        target_angle = self.compute_angle(target_rel_pos)
        # if np.abs(cam_angle - target_angle) <= np.pi/4 and np.linalg.norm(target_rel_pos) <= 6.0:
        targ_view = target_rel_pos
        # else:
        #     targ_view = np.array([-10.0, -10.0])
        target_rel_distance = np.linalg.norm(target_rel_pos, ord=2)

        rel_obstacle_angle = obstacle_angle - self.curr_state[2]

        modulo_rel_obs_angle = rel_obstacle_angle % (2*np.pi)

        rel_target_angle = target_angle - self.curr_angle
        
        modulo_rel_targ_angle = rel_target_angle % (2 * np.pi)

        obstacle_lidar = np.ones((16,)) * 20

        target_lidar = np.ones((8,)) * 20

        # thresholds = np.arange(0, 17/8 * np.pi, 1/8 * np.pi)
        thresholds = np.arange(-(1/4) * np.pi, (5/16)*np.pi, 1/16 * np.pi)

        for i in range(len(thresholds)-1):
            min_dist = None
            for j in range(len(obstacle_rel_distances)):
                upper_thresh = thresholds[i+1] % (2*np.pi)
                ## covers corner case of interval directly to the left of the camera angle
                if np.linalg.norm(upper_thresh) < 0.01:
                    upper_thresh = 2 * np.pi
                if thresholds[i] % (2 * np.pi) < obstacle_mod_angles[j] < upper_thresh:
                    if (not min_dist or obstacle_rel_distances[j] < min_dist) and obstacle_rel_distances[j] < 6.0:
                        min_dist = obstacle_rel_distances[j]
            if min_dist is not None:
                obstacle_lidar[i] = min_dist

        # return np.concatenate([np.array([rel_target_angle, (target_rel_distance)%(2*np.pi)]), obstacle_lidar, np.array([(self.curr_state[2] - self.curr_angle) % (2*np.pi)])])
        return np.concatenate([self.curr_state[:2], np.array([rel_target_angle%(2*np.pi), target_rel_distance]), obstacle_lidar, np.array([(self.curr_state[2] - self.curr_angle) % (2*np.pi)])])

    # determines whether inside the outer bounds of the sim
    def is_in_bounds(self):
        return (self.curr_state[0] > X_BOUNDS[0] and self.curr_state[0] < X_BOUNDS[1]) and (self.curr_state[1] > Y_BOUNDS[0] and self.curr_state[1] < Y_BOUNDS[1])

    # determines whether a wall was crossed -- walls for now defined by major axis, and segment length
    def crosses_wall(self, incr):
        # here, a single line wall halfway down, with a gap in the middle of width 2
        is_above_line_before = self.curr_state[0] < 7.5
        is_above_line_after = self.curr_state[0] + incr[0] < 7.5
        crosses_line = is_above_line_before ^ is_above_line_after
        is_in_hall_before = -1.0 < self.curr_state[1] < 1.0
        is_in_hall_after = -1.0 < self.curr_state[1] + incr[1] < 1.0
        is_in_hall_both = is_in_hall_before and is_in_hall_after
        return crosses_line and not is_in_hall_both

    def step(self, action):
        # make single step in direction of target
        # self.curr_state[:2] += action[:2]  # move by action
        reg_val = action[:2] / ACTION_SCALE
        self.curr_angle += reg_val[1]
        xy_vals = np.array([reg_val[0] * np.cos(self.curr_angle), reg_val[0] * np.sin(self.curr_angle)])

        mvt_interval = 10

        xy_vals_incr = xy_vals / mvt_interval

        for _ in range(mvt_interval):
            if self.is_in_bounds():
                self.curr_state[:2] += xy_vals_incr
            else:
                break

        self.curr_state[2] += action[2] # increment camera angle gradually
        
        self.curr_obs = self.compute_observation()

        # for i in range(len(thresholds)-1):
        #     if thresholds[i] < modulo_rel_targ_angle < thresholds[i+1]:
        #         target_lidar[i] = target_rel_distance

        
        # if 3/2 * np.pi < modulo_rel_obs_angle < 13/8 * np.pi:
        #     obstacle_lidar[0] = obstacle_rel_distance
        # if 13/8 * np.pi < modulo_rel_obs_angle < 7/4 * np.pi:
        #     obstacle_lidar[1] = obstacle_rel_distance
        # if 7/4 * np.pi < modulo_rel_obs_angle < 15/8 * np.pi:
        #     obstacle_lidar[2] = obstacle_rel_distance
        # if 15/8 * np.pi < modulo_rel_obs_angle:
        #     obstacle_lidar[3] = obstacle_rel_distance
        # if modulo_rel_obs_angle < 1/8 * np.pi:
        #     obstacle_lidar[4] = obstacle_rel_distance
        # if 1/8 * np.pi < modulo_rel_obs_angle < 1/4 * np.pi:
        #     obstacle_lidar[5] = obstacle_rel_distance
        # if 1/4 * np.pi < modulo_rel_obs_angle < 3/8 * np.pi:
        #     obstacle_lidar[6] = obstacle_rel_distance
        # if 3/8 * np.pi < modulo_rel_obs_angle < 1/2 * np.pi:
        #     obstacle_lidar[7] = obstacle_rel_distance

        obstacle_rel_distances = []
        obstacle_mod_angles = []
        curr_idx = 7

        for i in range(NUM_OBSTACLES):
            obstacle_rel_pose = self.curr_state[curr_idx:curr_idx+2] - self.curr_state[:2]
            obstacle_rel_distance = np.linalg.norm(obstacle_rel_pose, ord=2)
            obstacle_angle = self.compute_angle(obstacle_rel_pose)
            obstacle_rel_angle = obstacle_angle - self.curr_state[2] ## relative angle calculated with respect to CAMERA ANGLE
            obstacle_mod_angle = obstacle_rel_angle % (2*np.pi)
            obstacle_rel_distances.append(obstacle_rel_distance)
            obstacle_mod_angles.append(obstacle_mod_angle)
            curr_idx += 2


        # if too close to object, indicate that the current trajectory violated a safety constraint
        if (np.array(obstacle_rel_distances) <= 1.0).any(): ## TODO: DEBUG
            self.constraint_viol = True

        # self.curr_obs = np.concatenate((self.curr_obs)) ## add norm of destination as proxy for speed

        # ## TODO: integrate noise in the flag
        # if self.is_in_ray(action, self.belief_true['target1'].detach().numpy()):
        #     ## sample around the true belief, with extremely low variation / error
        #     # noisy_obs = distros.MultivariateNormal(self.belief_true['target1'], 0.01 * torch.eye(2)).sample().numpy()
        #     no_noisy_obs = self.belief_true['target1'].detach().numpy()

        #     if no_noisy_obs[0] < 0.001:
        #         nan_ang = np.pi/2 if no_noisy_obs[1] >= 0.0 else -np.pi/2
        #         self.curr_obs = np.array([nan_ang, 1.0])
        #     else:
        #         self.curr_obs = np.array([np.arctan(no_noisy_obs[1]/no_noisy_obs[0]), 1.0])
        # else:
        #     ## reject this observation, give zero reading
        #     # noisy_obs = distros.MultivariateNormal(torch.zeros((2,)), 0.01 * torch.eye(2)).sample().numpy()
        #     no_noisy_obs = np.zeros((2,))
        #     self.curr_obs = np.array([0.0, 0.0])

        return self.curr_obs, 1.0, False, {}
    
    def sample_belief_true(self):
        belief_true =  {'obs'+str(i+1): torch.from_numpy(self.curr_state[7 + 2*i:9+2*i]) for i in range(NUM_OBSTACLES)}
        belief_true.update({'target1': torch.from_numpy(self.curr_state[3:5])})
        return belief_true

    def reset(self):
        self.curr_state = np.array([0.0]*(7+2*NUM_OBSTACLES))
        self.curr_angle = 0.0
        self.curr_obs = self.compute_observation()
        self.constraint_viol = False
        return self.curr_obs

    def render(self, mode='rgb_array'):
        def is_in_ray_vectorized(ang, curr_loc, x_coord, y_coord, ray_ang):
            adjust_ang = (ang + np.pi/2)%(2 * np.pi) - np.pi/2

            return np.where(x_coord - curr_loc[0] > 0, 
                            np.all([np.abs(np.arctan((y_coord - curr_loc[1])/(x_coord - curr_loc[0])) - adjust_ang) <= ray_ang, np.linalg.norm(np.tile(curr_loc.reshape(-1, 1,1,1), (1, 256, 256, 3)) - np.array([x_coord, y_coord]), axis=0)<= 6.0], axis=0),
                            np.all([np.abs(np.arctan((y_coord - curr_loc[1])/(x_coord - curr_loc[0])) - (adjust_ang - np.pi)) <= ray_ang, np.linalg.norm(np.tile(curr_loc.reshape(-1, 1,1,1), (1, 256, 256, 3)) - np.array([x_coord, y_coord]), axis=0)<= 6.0], axis=0))
            
        def is_close_to_obj_vectorized(true_loc, x_coord, y_coord, r=0.2):
            return np.linalg.norm(np.stack((x_coord, y_coord)) - np.tile(true_loc.reshape(-1, 1, 1, 1), (1, 256, 256, 3)), axis=0) <= r

        color_arr = np.ones((256, 256, 3), dtype=np.uint8) * 255
        
        ## initializing vectorized x_coord and y_coord arrays
        x_coords = np.stack([np.tile(np.arange(X_BOUNDS[0], X_BOUNDS[1], (X_BOUNDS[1]-X_BOUNDS[0])/256.).reshape(-1, 1), (1, 256))]*3, axis=2)
        y_coords = np.stack([np.tile(np.arange(Y_BOUNDS[0], Y_BOUNDS[1], (Y_BOUNDS[1] - Y_BOUNDS[0])/256.).reshape(1, -1), (256, 1))]*3, axis=2)

        red = np.stack((np.ones((256, 256), dtype=np.uint8)*255, np.zeros((256, 256), dtype=np.uint8), np.zeros((256, 256), dtype=np.uint8)), axis=2)
        green = np.stack((np.zeros((256, 256), dtype=np.uint8), np.ones((256, 256), dtype=np.uint8)*255, np.zeros((256, 256), np.uint8)), axis=2)
        blue = np.stack((np.zeros((256, 256), np.uint8), np.zeros((256, 256), np.uint8), np.ones((256, 256), dtype=np.uint8)*255), axis=2)
        white = np.ones((256, 256, 3), dtype=np.uint8) * 255
        orange = np.stack((np.ones((256, 256), np.uint8)*255, np.ones((256, 256), np.uint8)*172, np.ones((256, 256), dtype=np.uint8)*28), axis=2)

        ## coloring in the robot location
        color_arr = np.where(is_close_to_obj_vectorized(self.curr_state[:2], x_coords, y_coords),
                                            red, 
                                            white)

        ## coloring in pointer
        color_arr = np.where(is_in_ray_vectorized(self.curr_state[2], self.curr_state[:2], x_coords, y_coords, np.pi/4),
                                            green+red, 
                                            color_arr)
        
        ## coloring in the safety constraint
        curr_idx = 7

        for _ in range(NUM_OBSTACLES):
            color_arr = np.where(is_close_to_obj_vectorized(self.curr_state[curr_idx:curr_idx+2], x_coords, y_coords, r=1.0),
                                        orange, 
                                        color_arr)

            curr_idx += 2

        curr_idx = 7
        ## coloring in the obstacle
        for _ in range(NUM_OBSTACLES):
            color_arr = np.where(is_close_to_obj_vectorized(self.curr_state[curr_idx:curr_idx+2], x_coords, y_coords),
                                        blue, 
                                        color_arr)
            
            curr_idx += 2

                
        ## coloring in the target
        color_arr = np.where(is_close_to_obj_vectorized(self.curr_state[3:5], x_coords, y_coords),
                                    green, 
                                    color_arr)

        return color_arr
    

class GymEnvNavWrapper(GymEnvNavTheta):
    def reset_to_state(self, state):
        # self.curr_state[:3] = state[:3]
        # self.curr_state[5:7] = state[5:7]
        self.curr_state = state
        self.curr_angle = 0.0
        self.curr_obs = self.compute_observation()
        self.constraint_viol = False
        return self.curr_obs

    def get_vector(self):
        state_vector_include = {
            'pr2': ['pose', 'theta'],
            'target1': ['value'],
            'softtarget1': ['value'],
        }
        state_vector_include.update({'obs'+str(i): ['value'] for i in range(1, NUM_OBSTACLES+1)})
        
        action_vector_include = {
            'pr2': ['pose', 'theta']
        }

        target_vector_include = {
            'target': ['pose']
        }
        
        return state_vector_include, action_vector_include, target_vector_include


    # reset without affecting the simulator
    def get_random_init_state(self):
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

        rand_side_init = random.randrange(4)
        # rand_side_goal = random.randrange(4)
        # while rand_side_goal == rand_side_init:
        #     rand_side_goal = random.randrange(4)

        init_coords = random.random() * 6 - 3
        # goal_coords = random.random() * 6 - 3

        init_pos = np.array([-1, -1])
        # goal_pos = np.array([-1, -1])

        if rand_side_init == 0:
            init_pos = np.array([-3., init_coords])
        if rand_side_init == 1:
            init_pos = np.array([init_coords, -3.])
        if rand_side_init == 2:
            init_pos = np.array([3., init_coords])
        if rand_side_init == 3:
            init_pos = np.array([init_coords, 3.])

        # if rand_side_goal == 0:
        #     goal_pos = np.array([-3., goal_coords])
        # if rand_side_goal == 1:
        #     goal_pos = np.array([goal_coords, -3.])
        # if rand_side_goal == 2:
        #     goal_pos = np.array([3., goal_coords])
        # if rand_side_goal == 3:
        #     goal_pos = np.array([goal_coords, 3.])

        # goal_pos = -init_pos
        # soft_goal_pos = init_pos * 2/3
        # proposal_obs = goal_pos
        # obstacle_abs_angle = np.arctan(proposal_obs[1]/proposal_obs[0]) if np.abs(proposal_obs[0]) > 0.001 else (np.pi/2 if proposal_obs[1]*proposal_obs[0]>0 else -np.pi/2)
        # obstacle_angle = obstacle_abs_angle if proposal_obs[0] >= 0  else (obstacle_abs_angle + np.pi if -np.pi/2 <= obstacle_abs_angle < 0 else obstacle_abs_angle - np.pi)

        obs_arr = []

        for i in range(NUM_OBSTACLES):
            # obs_arr.append(self.obs_dist.sample().detach().numpy())
            obs_arr.append(np.array([7.0, 0.0]))

        # obs = self.obs_dist.sample().detach().numpy()
        # obs_2 = self.obs_dist.sample().detach().numpy()
        # obs_3 = self.obs_dist.sample().detach().numpy()
        # obs_4 = self.obs_dist.sample().detach().numpy()
        # obs_5 = self.obs_dist.sample().detach().numpy()
        # goal_pos = self.target_dist.sample().detach().numpy()
        goal_pos = np.array([12.0, 0.0])

        curr_idx = 7

        for i in range(NUM_OBSTACLES):
            self.curr_state[curr_idx:curr_idx+2] = obs_arr[i]
            curr_idx += 2

        # self.curr_state[7:9] = obs
        # self.curr_state[9:11] = obs_2
        # self.curr_state[11:13] = obs_3
        # self.curr_state[13:15] = obs_4
        # self.curr_state[15:17] = obs_5
        self.curr_state[3:5] = goal_pos

        self.curr_angle = self.compute_angle(self.curr_state[3:5])

        ## initalize to center
        return np.concatenate([np.array([0.0, 0.0]), np.array([self.curr_angle]), goal_pos, np.array([8.0,0.])] + obs_arr)

    # determine whether or not a given state satisfies a goal condition
    def assess_goal(self, condition, state, targets=None, cont=None):
        angle = self.curr_state[2]
        goal_rel_pose = self.curr_state[3:5] - self.curr_state[:2]
        # if pointing directly at the object

        if np.linalg.norm(goal_rel_pose) <= 2.0:
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
