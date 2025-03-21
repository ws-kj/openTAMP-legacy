from opentamp.policy_hooks.core_agents.tamp_agent import TAMPAgent
import opentamp.policy_hooks.utils.policy_solver_utils as utils
from opentamp.policy_hooks.custom_agents.gym_agent import GymAgent

import numpy as np
# from opentamp.policy_hooks.gym_prob import *

class RnnGymAgent(GymAgent):
    def __init__(self, hyperparams):
        super().__init__(hyperparams)
        self.gym_env = hyperparams['gym_env_type']()
        self.done = False
        self.curr_obs = self.gym_env.reset()
        self.curr_state = self.gym_env.curr_state
        self.num_tasks = 0
        self.curr_targ = np.array([0., 0.])
        self.past_targ = np.array([0.])
        self.past_point = 0
        self.past_val = 0
        self.past_task = -1.0
        self.horizon = 20
        self.past_task_arr = np.array([np.inf] * self.horizon)
        self.past_obs_arr = np.array([np.inf] * self.gym_env.observation_space.shape[0] * self.horizon)

    def reset_to_state(self, x):
        self.curr_obs = self.gym_env.reset_to_state(x)
        self.curr_state = self.gym_env.curr_state
        self.past_task_arr = np.array([np.inf] * self.horizon)
        self.past_obs_arr = np.array([np.inf] * self.gym_env.observation_space.shape[0] * self.horizon)

    def reset(self, m):
        self.curr_obs = self.gym_env.reset()
        self.curr_state = self.gym_env.curr_state
        self.past_task_arr = np.array([np.inf] * self.horizon)
        self.past_obs_arr = np.array([np.inf] * self.gym_env.observation_space.shape[0] * self.horizon)

    ## subclass as needed for different kinds of sample populations (e.g. RNN stuff)
    def fill_sample(self, cond, sample, mp_state, t, task, fill_obs=False, targets=None):
        sample.set(utils.MJC_SENSOR_ENUM, self.curr_obs, t)
        sample.set(utils.STATE_ENUM, self.curr_state, t)
        sample.set(utils.PAST_COUNT_ENUM, np.array([self.num_tasks]), t)
        if fill_obs:
            sample.set(utils.DEST_PRED, self.curr_targ, t=t) ## only write if cont policy isn't
        sample.set(utils.PAST_ANG_ENUM, self.past_targ, t=t)
        sample.set(utils.PAST_POINT_ENUM, np.array([self.past_point]), t=t)
        sample.set(utils.PAST_VAL_ENUM, np.array([self.past_val]), t=t)
        sample.set(utils.PAST_TASK_ENUM, np.array([self.past_task]), t=t)
        sample.set(utils.PAST_TASK_ARR_ENUM, self.past_task_arr, t)
        sample.set(utils.PAST_MJCOBS_ARR_ENUM, self.past_obs_arr, t)
        sample.set(utils.PAST_RECUR_ENUM, np.concatenate([self.past_task_arr.reshape((-1,1)), self.past_obs_arr.reshape(-1, self.gym_env.observation_space.shape[0])], axis=1).flatten(), t)
        # sample.set(utils.PAST_MJCOBS_ARR_CURR_ENUM, sample., t)
        # past_mjc_obs_curr_arr = np.array([-1.] * self.gym_env.observation_space.shape[0] * self.horizon)
        # past_mjc_obs_curr_arr[:t*self.gym_env.observation_space.shape[0]] = sample.get(utils.MJC_SENSOR_ENUM)[:t,:].flatten()
        # sample.set(utils.PAST_MJCOBS_CURR_ARR_ENUM, past_mjc_obs_curr_arr, t)
        sample.set(utils.TIME_ENUM, np.array([t]), t)
        # sample.set(utils.IM_ENUM, self.gym_env.get_im().flatten(), t)

    def store_hist_info(self, hist_info):
        self.num_tasks = hist_info[0] 
        self.past_targ = hist_info[1]
        self.past_point = hist_info[2]
        self.past_val = hist_info[3]
        self.past_task = hist_info[4]
        self.past_task_arr[:self.num_tasks+1] = hist_info[5][:self.num_tasks+1]
        obs_dim = self.gym_env.observation_space.shape[0]
        # try:
        self.past_obs_arr[:(self.num_tasks+1)*obs_dim] = hist_info[6][:(self.num_tasks+1)*obs_dim]
        # except:
        #     pass

    def store_aux_info(self, aux_info):
        self.curr_targ = aux_info

    def get_inv_cov(self):
        return np.eye(self.dU)