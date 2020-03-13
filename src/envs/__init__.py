from functools import partial
from smac.env import MultiAgentEnv, StarCraft2Env
import sys
import os
import gym
from gym.envs import registry as gym_registry
from gym.spaces import flatdim
import numpy as np
from gymma.wrappers import TimeLimit, FlattenObservation

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH",
                          os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))

class _GymmaWrapper(MultiAgentEnv):
    def __init__(self, key, time_limit, **kwargs):
        self.episode_limit = time_limit
        self._env = TimeLimit(gym.make(f"gymma:{key}"), max_episode_steps=time_limit)
        self._env = FlattenObservation(self._env)
        self.n_agents = self._env.n_agents
        self._obs = None

        self.longest_action_space = max(self._env.action_space, key=lambda x: x.n)
        self.longest_observation_space = max(self._env.observation_space, key=lambda x: x.shape)

        self._seed = kwargs["seed"]
        self._env.seed(self._seed)
  
    def step(self, actions):
        """ Returns reward, terminated, info """
        actions = [int(a) for a in actions]
        self._obs, reward, done, info = self._env.step(actions)        
        self._obs = [np.pad(o, (0,self.longest_observation_space.shape[0] - len(o)), 'constant', constant_values=0) for o in self._obs]

        return float(sum(reward)), all(done), {}

    def get_obs(self):
        """ Returns all agent observations in a list """
        return self._obs

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        raise self._obs[agent_id]

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return flatdim(self.longest_observation_space)

    def get_state(self):
        return np.concatenate(self._obs, axis=0).astype(np.float32)
        
    def get_state_size(self):
        """ Returns the shape of the state"""
        return self.n_agents*flatdim(self.longest_observation_space)

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        raise self._obs[agent_id]

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return flatdim(self.longest_observation_space)

    def get_state(self):
        return np.concatenate(self._obs, axis=0).astype(np.float32)
        
    def get_state_size(self):
        """ Returns the shape of the state"""
        return self.n_agents*flatdim(self.longest_observation_space)

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        valid = flatdim(self._env.action_space[agent_id]) * [1]
        invalid = [0] * (self.longest_action_space.n - len(valid))
        return valid+invalid

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        return flatdim(self.longest_action_space)

    def reset(self):
        """ Returns initial observations and states"""
        self._obs = self._env.reset()
        self._obs = [np.pad(o, (0,self.longest_observation_space.shape[0] - len(o)), 'constant', constant_values=0) for o in self._obs]
        return self.get_obs(), self.get_state()


    def render(self):
        self._env.render()

    def close(self):
        self._env.close()

    def seed(self):
        return self._env.seed

    def save_replay(self):
        pass

    def get_stats(self):
        return {}

REGISTRY["gymma"] = partial(env_fn, env=_GymmaWrapper)
