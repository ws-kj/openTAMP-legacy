import numpy as np

import gymnasium as gym

import random
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from opentamp.envs.floorplan_env import FloorplanEnv, FloorplanEnvWrapper

from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

from stable_baselines3.common.monitor import Monitor 

from stable_baselines3.common.logger import configure

import mediapy as media
import imageio

import pickle

import matplotlib.pyplot as plt

# import multiprocessing
# multiprocessing.set_start_method('spawn')

# env = BlankEnvWrapperBimodal()

class tmp_env(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Box(low=np.array([0.]*1), high=np.array([1.0]*1), shape=(1,), dtype='float32')
        self.observation_space = gym.spaces.Box(low=np.array([0.]*1), high=np.array([1.0]*1), shape=(1,), dtype='float32')

    def reset(self, seed=None):
        self.num = random.random()
        self.t = 0
        return np.array([self.num]), {}

    def step(self, action):
        self.t += 1
        if self.t == 2:
            done = True
            reward = 1.0 if np.linalg.norm(self.num - action) < 0.1 else 0.0
        else:
            done = False
            reward = 0.0

        return np.array([0.0]), reward, done, False, {}

    def render(self):
        pass

class TmpCallback(BaseCallback):
    def _on_step(self):
        print(self.locals['rewards'])

        return True


# del model # remove to demonstrate saving and loading

# model = RecurrentPPO.load("ppo_recurrent")

# obs = vec_env.reset()
# # cell and hidden state of the LSTM
# lstm_states = None
# num_envs = 1
# # Episode start signals are used to reset the lstm states
# episode_starts = np.ones((num_envs,), dtype=bool)
# while True:
#     action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
#     obs, rewards, dones, info = vec_env.step(action)
#     episode_starts = dones
    # print(rewards)
    # vec_env.render("human") ## TODO fill in with render logic here

# have to add walls and stuff manually here...
def make_env():
    env = FloorplanEnvWrapper()
    env.walls = [(np.array([0.,0.]), np.array([0.,5.])), (np.array([0.,0.]), np.array([5.,0.])), (np.array([5.0, 0.0]), np.array([5.,5.])), (np.array([0.,5.]), np.array([5.,5.]))]
    env.reset()
    return Monitor(env)


def main_load(load_string):
    stats_dicts = []

    for i in range(100):
        num_envs = 128

        model = RecurrentPPO.load(load_string)

        vec_env = DummyVecEnv([make_env for _ in range(num_envs)])

        vec_env.render_mode = 'rgb_array'

        obs = vec_env.reset()
        while np.linalg.norm(vec_env.envs[0].position - vec_env.envs[0].target) < 3.0:
            obs = vec_env.reset()

        # cell and hidden state of the LSTM
        lstm_states = None
        # Episode start signals are used to reset the lstm states
        episode_starts = np.ones((num_envs,), dtype=bool)

        should_stop = False
        gif_images = []

        dummy_env = make_env()
        dummy_env.reset()

        dummy_env.env.position = vec_env.envs[0].position
        dummy_env.env.target = vec_env.envs[0].target
        dummy_env.env.walls = vec_env.envs[0].walls
        dummy_env.env.curr_angle = vec_env.envs[0].curr_angle

        curr_stats_dict = None

        while not should_stop:
            action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts)
            obs, rewards, dones, info = vec_env.step(action)

            if dones[0]:
                stats_dicts.append(curr_stats_dict)

            episode_starts = dones
            dummy_env.step(action[0])

            # gif_images.append(dummy_env.render())        
            # media.write_image('tmp_image', imgs[0])
            should_stop = dones[0]

            curr_stats_dict = dummy_env.compute_stats_dict()

        print('Finished episode!')

        # imageio.mimwrite('tmp/obstacle_gifs/baseline_rollout'+str(i)+'.gif', gif_images, duration=1)
    
    # plt.axis('on')

    # # plt.hist(dists)
    # plt.xlabel('Distance from Target')
    # plt.ylabel('Frequency')

    # plt.savefig('tmp/distance_hist.pdf')

    with open('pickle_samp_baselines_redo.pkl', 'wb') as f:    
        pickle.dump(stats_dicts, f)


    
    
def main_train(save_str):
    tmp_path = "./tmp/server_logs/"+save_str
    # set up logger
    new_logger = configure(tmp_path, ["stdout", "csv"])
    num_envs = 2
    num_ts = 1e7

    env = SubprocVecEnv([make_env for _ in range(num_envs)])

    # env = tmp_env()

    model = RecurrentPPO("MlpLstmPolicy", env, verbose=2, clip_range=0.05, ent_coef=0.05, batch_size=4096, deterministic=True)
    model.set_logger(new_logger)
    model.learn(num_ts, callback=EvalCallback(DummyVecEnv([make_env for _ in range(num_envs)]), eval_freq=1000, n_eval_episodes=20))

    vec_env = model.get_env()
    mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=20, warn=False)
    print(mean_reward)

    model.save(save_str)

    # num_gif = 20

    # for i in range(num_gif):
    #     obs = vec_env.reset()
    #     while np.linalg.norm(vec_env.envs[0].position - vec_env.envs[0].target) < 3.0:
    #         obs = vec_env.reset()

    #     # cell and hidden state of the LSTM
    #     lstm_states = None
    #     # Episode start signals are used to reset the lstm states
    #     episode_starts = np.ones((num_envs,), dtype=bool)

    #     should_stop = False
    #     gif_images = []

    #     while not should_stop:
    #         action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
    #         obs, rewards, dones, info = vec_env.step(action)
    #         episode_starts = dones
    #         gif_images.append(vec_env.envs[0].render())
    #         # media.write_image('tmp_image', imgs[0])
    #         should_stop = dones[0]

    #         print('Finished episode!')

    #         imageio.mimwrite("./tmp/server_saves/" + save_str + '_' + str(i), gif_images, duration=0.0001)

if __name__ == '__main__':
    # main_train('tmp/server_runs/server_saves/ppo_search_singleroomobsspotlight_testrotate')
    main_load('tmp/server_runs/server_runs/server_saves/ppo_search_singleroomobsspotlight_avoid0.7_50pen_debugobs_skolemrewardprog')