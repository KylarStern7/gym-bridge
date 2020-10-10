"""
Script for testing BridgeEnv environment.

Three AgentRandom and one AgentDummy agents are initialized and used to interact with the environment.
"""

import gym
from time import sleep
from agents import AgentRandom, AgentDummy


env = gym.make('gym_bridge:bridge-v0')
env.seed(0)
agents = {'N': None, 'E': None, 'S': None, 'W': None}

episode_count = 100

for i in range(episode_count):
    reward = 0
    done = False
    ob = env.reset()
    for agent in agents:
        agents[agent] = AgentRandom()
    agents[env.players_roles['dummy']] = AgentDummy(agents[env._get_next_player(env._get_next_player(env.players_roles['dummy']))])
    print(env.render())
    while True:
        agent = agents[env.state['active_player']]
        action = agent.act(env.get_available_actions(env.state['active_player']), ob, reward, done)
        ob_old = ob
        ob, reward, done, _ = env.step(action)
        sleep(0)
        print(env.render())
        if done:
            break
env.close()
