import numpy as np
import gym

from opentamp.policy_hooks.utils.agent_env_wrapper import *
from opentamp.policy_hooks.utils.load_agent import *

def run(config, mode='train'):
    args = config['args']
    agent_config = load_agent(config)
    agent = build_agent(agent_config)
    register_env(config, 'exampleEnv-v0')
    env = gym.make('exampleEnv-v0')
    import ipdb; ipdb.set_trace()

