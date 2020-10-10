"""
Script for training AgentDQN with BridgeEnv environment.

During training AgentDQN has permanent position ("E") and role ("defender_1").
Other players are AgentRandom instances.
AgentDQN is initialized with parameters:
    gamma = 0.75
    max_experiences = 1000
    min_experiences = 500
    batch_size = 64
    epsilon = 0.99

Training is performed during 20000 environment episodes with epsilon_decay set to 0.9997
and minimal epsilon set to 0.01. At the end of training agent's neural network model is saved to "model.h5" file.
"""

import gym
from agents import AgentDQN, AgentRandom
import numpy as np
from time import time


env = gym.make('gym_bridge:bridge-v0', reward_mode='play_cards')
env.seed(0)
player = AgentRandom()
defender_1 = AgentDQN(gamma=0.75, max_experiences=1000, min_experiences=500, batch_size=64, epsilon=0.99)
defender_2 = AgentRandom()
dummy = AgentRandom()
agents = {'N': player, 'E': defender_1, 'S': dummy, 'W': defender_2}

episode_count = 20000
reward = 0
done = False
eps_decay = 0.9997
eps_min = 0.01

total_rewards = []

for i in range(episode_count):
    ob = env.reset({'player': 'N'})
    ob_list = []
    reward_list = []
    action_list = []
    done_list = [0] * 12 + [1]
    total_rewards_per_episode = 0
    reward = None
    done = False
    while True:

        agent = agents[env.state.get('active_player')]
        available_actions = env.get_available_actions(env.state.get('active_player'))
        if env.state.get('active_player') == 'E':
            ob = ob.get('player_hand') + ob.get('current_suit')
            ob_list.append(ob)
            if reward is not None:
                reward_list.append(reward)
            #done_list.append(done)
            action = agent.act(ob)
            last_card = env.state.get('hands').get('E')[0]
            action_learning_agent = action
            action_list.append(action)
            agent.epsilon = max(agent.epsilon * eps_decay, eps_min)

        else:
            action = agent.act(available_actions, ob, reward, done)

        ob, reward, done, _ = env.step(action)
        if done:
            if last_card == action_learning_agent:
                reward_list.append(1)
            else:
                reward_list.append(-10)

            for j, state in enumerate(ob_list):
                if j == len(ob_list)-1:
                    exp = (ob_list[j], action_list[j], reward_list[j], 0, True)
                else:
                    exp = (ob_list[j], action_list[j], reward_list[j], ob_list[j+1], done_list[j])
                total_rewards_per_episode += reward_list[j]
                agents['E'].add_experience(exp)

            agents['E'].train()

            if i % 25 == 0:
                agents['E'].copy_weights()
            break
    total_rewards.append(total_rewards_per_episode)
    if i % 100 == 0:
        print(i, np.mean(total_rewards[-100:]))
env.close()
print([(i, r) for i, r in enumerate(total_rewards)])
agents['E'].save_model()
