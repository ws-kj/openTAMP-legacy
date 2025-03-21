import mujoco
import roboticstoolbox as rtb
from roboticstoolbox import ET
import gym
from gym import spaces
import os
import numpy as np
import opentamp
import random

ACTION_SCALE = 40
OBS_DIM = 13  

class SpotEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Box(low=np.array([-0.25]*10), high=np.array([0.25]*10), shape=(10,), dtype='float32')
        self.observation_space = spaces.Box(low=np.array([-10.0]*OBS_DIM), high=np.array([10.0]*OBS_DIM), shape=(OBS_DIM,), dtype='float32')

        self.set_ungrasped_kin_tree()
        
        urdf_path = opentamp.__path__._path[0]+ "/assets/mujoco/scene.xml"

        with open(urdf_path) as f:
            data = f.read()

        past_wd = os.getcwd()
        
        os.chdir(opentamp.__path__._path[0]+ "/assets/mujoco/")
        
        self.model = mujoco.MjModel.from_xml_string(data)
        self.data = mujoco.MjData(self.model)

        os.chdir(past_wd)

        self.curr_state = np.array([0.0]*13) ## robot config + ball pose
        self.is_grasping_ball = False

        self.joint_bounds = np.array([
            [-5/6*np.pi, np.pi],
            [-np.pi, np.pi/6],
            [0., np.pi],
            [-8/9*np.pi, 8/9*np.pi],
            [-7/12*np.pi, 7/12*np.pi],
            [-8/9*np.pi, 8/9*np.pi],
            [-np.pi/2, 0]
        ])  # bounds for all arm joints

        self.set_mjc_pose()
        

    def set_ungrasped_kin_tree(self):
        self.transforms = ET.tx() * ET.ty() * ET.tz(0.75) * ET.Rz() * ET.tx(0.292) * ET.tz(0.188) *ET.Rz() *ET.Ry()* ET.tx(0.3385) *  ET.Ry() * ET.tx(0.1033) * ET.tz(0.075)* ET.Rx() * ET.tx(0.3) * ET.Ry() * ET.tx(0.11745) * ET.Rx() * ET.tx(0.075) 
        self.kin_tree = rtb.Robot(self.transforms)
    
    def set_grasped_kin_tree(self):
        # get relative pose of an object in end effector frame 
        self.kin_tree.q = self.curr_state[:9]
        eff_pose = self.kin_tree.fkine(self.kin_tree.q)
        rotation_mat = eff_pose.R
        rel_vec = self.curr_state[10:13] - np.array([eff_pose.x, eff_pose.y, eff_pose.z])
        rot_rel_vec = np.linalg.pinv(rotation_mat) @ rel_vec 
        
        # add an additional transformation, end effector is now the grasped ball
        self.transforms = ET.tx() * ET.ty() * ET.tz(0.75) * ET.Rz() * ET.tx(0.292) * ET.tz(0.188) *ET.Rz() *ET.Ry()* ET.tx(0.3385) *  ET.Ry() * ET.tx(0.1033) * ET.tz(0.075)* ET.Rx() * ET.tx(0.3) * ET.Ry() * ET.tx(0.11745) * ET.Rx() * ET.tx(0.075) * ET.tx(rot_rel_vec[0]) * ET.ty(rot_rel_vec[1]) * ET.tz(rot_rel_vec[2]) 
        self.kin_tree = rtb.Robot(self.transforms)
    
    def set_mjc_pose(self):
        self.data.qpos[0] = self.curr_state[0]
        self.data.qpos[1] = self.curr_state[1]
        self.data.qpos[2] = 0.75
        spot_angle = self.curr_state[2]
        self.data.qpos[3] = np.cos(spot_angle/2) # quaternion shit
        self.data.qpos[6] = np.sin(spot_angle/2) # quaternion shit
        self.data.qpos[19:26] = self.curr_state[3:10] # remainder is a straightforward assignment

        # if ball is being grasped, do forward kin to determine the position of the ball
        if self.is_grasping_ball:
            self.kin_tree.q = self.curr_state[:9]
            ball_pose = self.kin_tree.fkine(self.kin_tree.q)
            ball_coords = np.array([ball_pose.x, ball_pose.y, ball_pose.z])
            # if np.linalg.norm(ball_coords - np.array([1.,1.,1.])) > 0.1:
            #     breakpoint()

            self.data.qpos[26:29] = ball_coords
    
    def round_in_grasp_bounds(self):
        self.curr_state[3:10] = np.maximum(self.curr_state[3:10], self.joint_bounds[:,0])
        self.curr_state[3:10] = np.minimum(self.curr_state[3:10], self.joint_bounds[:,1])
    
    def ball_is_near_gripper(self):
        self.kin_tree.q = self.curr_state[:9]
        joint_pose = self.kin_tree.fkine(self.kin_tree.q)
        grip_x, grip_y, grip_z = joint_pose.x, joint_pose.y, joint_pose.z
        if np.linalg.norm(np.array([grip_x, grip_y, grip_z]) - self.curr_state[10:13]) < 0.1:
            return True
        else:
            return False
    
    def reset(self):
        self.curr_state = np.array([0.0]*13) ## robot config + ball pose
        self.is_grasping_ball = False
        self.set_ungrasped_kin_tree()
        self.set_mjc_pose()
        return self.curr_state[:13], {}

    # assume incremental actions
    def step(self, action):
        action /= ACTION_SCALE ## command velocity

        # rotate em back!
        forward_rot_mat = np.array([[np.cos(self.curr_state[2]), -np.sin(self.curr_state[2])],[np.sin(self.curr_state[2]), np.cos(self.curr_state[2])]])
        action[:2] = forward_rot_mat @ action[:2]
        
        prior_grip = self.curr_state[9]
        self.curr_state[:10] += action
        self.round_in_grasp_bounds()
        
        # if grasped
        if prior_grip < -0.5 and self.curr_state[9] > -0.5:
            if self.ball_is_near_gripper():
                self.is_grasping_ball = True
                self.set_grasped_kin_tree()

        # if ungrasped
        if prior_grip > -0.5 and self.curr_state[9] < -0.5:
            self.is_grasping_ball = False
            self.set_ungrasped_kin_tree()
        
        # self.set_mjc_pose()

        # for _ in range(25):
        #     # with mujoco.Renderer(model) as renderer:
        #     mujoco.mj_forward(self.model, self.data)
        #     # mujoco.mj_step(self.model, self.data)
        #     self.set_mjc_pose()

        self.curr_state[10:13] = self.data.qpos[26:29] 

        self.kin_tree.q = self.curr_state[:9]
        eff_pose = self.kin_tree.fkine(self.kin_tree.q)
        rotation_mat = eff_pose.R
        # rel_vec = self.curr_state[10:13] - np.array([eff_pose.x, eff_pose.y, eff_pose.z])
        # rot_rel_vec = np.linalg.pinv(rotation_mat) @ rel_vec

        spot_rot_mat = np.array([[np.cos(self.curr_state[2]), np.sin(self.curr_state[2]), 0],[-np.sin(self.curr_state[2]), np.cos(self.curr_state[2]), 0.], [0.,0.,0.]])

        spot_to_grip = spot_rot_mat @ np.array([eff_pose.x, eff_pose.y, eff_pose.z]) - np.array([self.curr_state[0], self.curr_state[1], 0.75])
        grip_to_ball = spot_rot_mat @ (self.curr_state[10:13] - np.array([eff_pose.x, eff_pose.y, eff_pose.z]))
        spot_to_ball = spot_rot_mat @ (self.curr_state[10:13] - np.array([self.curr_state[0], self.curr_state[1], 0.75]))
        ball_to_origin = spot_rot_mat @ (np.array([0.,0.,1.]) - self.curr_state[10:13])

        obs =  np.concatenate([self.curr_state[:13]])

        # return current robot state, plus the relative distance to the gripper of the object
        return obs, 0., False, False, {}
        
    # gives 240 x 320 RGB image
    def render(self):
        self.set_mjc_pose()
        with mujoco.Renderer(self.model) as renderer:
            mujoco.mj_forward(self.model, self.data)
            renderer.update_scene(self.data, camera="track_spot")

            img = renderer.render()
        return img

class SpotEnvWrapper(SpotEnv):
    def reset_to_state(self, state):
        self.curr_state = state ## robot config + ball pose
        self.set_ungrasped_kin_tree()
        self.set_mjc_pose()
        # also add in the initial position of the ball
        self.data.qpos[26:29] = state[10:13]

        self.is_grasping_ball = False

        self.kin_tree.q = self.curr_state[:9]
        eff_pose = self.kin_tree.fkine(self.kin_tree.q)
        rotation_mat = eff_pose.R
        # rel_vec = self.curr_state[10:13] - np.array([eff_pose.x, eff_pose.y, eff_pose.z])
        # rot_rel_vec = np.linalg.pinv(rotation_mat) @ rel_vec

        spot_rot_mat = np.array([[np.cos(self.curr_state[2]), np.sin(self.curr_state[2]), 0],[-np.sin(self.curr_state[2]), np.cos(self.curr_state[2]), 0.], [0.,0.,0.]])

        spot_to_grip = spot_rot_mat @ (np.array([eff_pose.x, eff_pose.y, eff_pose.z]) - np.array([self.curr_state[0], self.curr_state[1], 0.75]))
        grip_to_ball = spot_rot_mat @ (self.curr_state[10:13] - np.array([eff_pose.x, eff_pose.y, eff_pose.z]))
        spot_to_ball = spot_rot_mat @ (self.curr_state[10:13] - np.array([self.curr_state[0], self.curr_state[1], 0.75]))
        ball_to_origin = spot_rot_mat @ (np.array([0.,0.,1.]) - self.curr_state[10:13])

        return self.curr_state[:13]

    def get_vector(self):
        state_vector_include = {
            'rob': ['pose'],
            'loc': ['value']
        }
        
        action_vector_include = {
            'rob': ['pose']
        }

        target_vector_include = {
        }
        
        return state_vector_include, action_vector_include, target_vector_include


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
        targ_nav = np.dot(np.random.rand(2,), np.array([0.4, 0.6])) + np.array([.7, -0.3])

        return np.concatenate([np.array([0., 0.]), np.random.rand(7,) * 0.05, np.array([-np.pi/4]), np.array([targ_nav[0],targ_nav[1],0.6])])

    def postproc_im(self, base_im, s, t, cam_id):
        return base_im

    # def sample_belief_true(self):
    #     belief_true = {'targ': torch.rand((2,))*5}
    #     return belief_true

    # def set_belief_true(self, belief_dict):
    #     self.belief_true = belief_dict
    #     self.target = belief_dict['targ']

    def get_im(self):
        return self.render()

    def populate_sim_with_facts(self, problem):
        pass
    #     ## clear all prior info
    #     self.walls = []
    #     self.locations = []
    #     self.obstacles = []

    #     for param_name, param in problem.init_state.params.items():
    #         if param._type == 'Wall':
    #             self.walls.append((param.endpoint1.reshape(-1), param.endpoint2.reshape(-1)))
    #         elif param._type == 'DomainSettings':
    #             self.low_bound, self.high_bound = param.low_bound.reshape(-1), param.high_bound.reshape(-1)
    #             # populate with the four boundary walls
    #             self.walls.append((self.low_bound, np.array([self.high_bound[0], self.low_bound[1]])))
    #             self.walls.append((self.low_bound, np.array([self.low_bound[0], self.high_bound[1]])))
    #             self.walls.append((np.array([self.low_bound[0], self.high_bound[1]]), self.high_bound))
    #             self.walls.append((np.array([self.high_bound[0], self.low_bound[1]]), self.high_bound))
    #         elif param._type == 'Location':
    #             if param.name == 'targ':
    #                 self.target = param.value.reshape(-1)
    #             else:
    #                 self.locations.append(param.value.reshape(-1))
    #         elif param._type == 'Obstacle':
    #             self.obstacles.append(param.value.reshape(-1))
    #         elif param._type == 'Robot':
    #             self.position = param.pose[:,0].reshape(-1)

    #     rel_targ = self.target - self.position
    #     self.init_angle = np.arctan2(rel_targ[1], rel_targ[0])
    #     self.curr_angle = self.init_angle

    def return_dict_with_facts(self):
        pass
        # return {
        #     'walls': [[list(w) for w in wall] for wall in self.walls],
        #     'locations': [list(w) for w in self.locations],
        #     'obstacles': [list(w) for w in self.obstacles],
        #     'position': list(self.position),
        #     'target': list(self.target)
        # }

    # determine whether or not a given state satisfies a goal condition
    def assess_goal(self, condition, state, targets=None, cont=None):
        # angle = self.curr_state[2]
        # goal_rel_pose = self.target - self.position
        # # # if pointing directly at the object

        # if np.linalg.norm(goal_rel_pose) <= 0.5:
        #     return 0.0
        # else:
        #     return 1.0
        self.kin_tree.q = self.curr_state[:9]
        eff_pose = self.kin_tree.fkine(self.kin_tree.q)
        actuator_pose = np.array([eff_pose.x, eff_pose.y])

        # if brought within 0.5 of origin (in x-y plane)
        if self.is_grasping_ball:
            return 0.0
        else:
            return 1.0


        # return 0.0 ## always succeeds for now

    # determine whether constraints have been violated since last reset
    def assess_constraint_viol(self):
        # if self.constraint_viol:
        #     return 1.0
        # else:
        #     return 0.0
        return 0.0