"""
Script for testing BridgeSimultaneousActionsEnv environment.

Three AgentRandom and one AgentDummy agents are initialized and used to interact with the environment.
"""

import gym
from time import sleep
from agents import AgentRandom, AgentDummy
import numpy as np


env = gym.make('gym_bridge:bridge_simultaneous-v0')
env.seed(0)
agents = {'N': None, 'E': None, 'S': None, 'W': None}

episode_count = 1

for i in range(episode_count):
    rewards = {agent: 0 for agent in agents}
    dones = {agent: False for agent in agents}
    observations = env.reset()
    for agent in agents:
        agents[agent] = AgentRandom()
    agents[env.players_roles['dummy']] = AgentDummy(agents[env._get_next_player(env._get_next_player(env.players_roles['dummy']))])
    print(env.render())
    while True:
        actions = {agent: None for agent in agents}
        for agent in agents:
            actions[agent] = agents[agent].act(env.get_available_actions(agent), observations[agent], rewards[agent],
                                               dones[agent])
        print("ACTIONS", actions)
        observations, rewards, dones, _ = env.step(actions)
        sleep(0)
        print(env.render())
        if np.all([done for done in dones.values()]):
            break
env.close()
