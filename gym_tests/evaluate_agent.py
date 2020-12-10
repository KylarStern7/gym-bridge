"""
Script for evaluating trained AgentDQN in BridgeEnv environment.

During evaluation AgentDQN has permanent position ("E") and role ("defender_1").
Other players are AgentRandom instances.

Evaluation is performed on 1000 environment episodes.
"""
import gym
from agents import AgentDQN, AgentRandom
import numpy as np
from tensorflow.keras.models import load_model

env = gym.make('gym_bridge:bridge-v0', reward_mode='play_cards')
env.seed(0)
player = AgentRandom()
model = load_model('model.h5')
defender_1 = AgentDQN(gamma=0.75, max_experiences=1000, min_experiences=500, batch_size=64, epsilon=0, model=model)
defender_2 = AgentRandom()
dummy = AgentRandom()
agents = {'N': player, 'E': defender_1, 'S': dummy, 'W': defender_2}

episode_count = 1000
reward = 0
done = False
eps_decay = 0.9997
eps_min = 0.01

total_rewards = []
total_rewards_per_trick = {"13": [], "1": [], "2": [], "3": [], "4": [], "5": [], "6": [], "7": [], "8": [], "9": [],
                           "10": [], "11": [], "12": []}

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
                total_rewards_per_trick[str(env.tricks_played)].append(reward)
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
                total_rewards_per_trick[str(13)].append(1)
            else:
                reward_list.append(-2)
                total_rewards_per_trick[str(13)].append(-2)

            break
    total_rewards.append(np.sum(reward_list))
    print(f'Episode {i} ended with total reward: {np.sum(reward_list)}/13.')
env.close()
#print([(i, r) for i, r in enumerate(total_rewards)])
print(f'Mean reward per episode: {np.mean(total_rewards)}')
for i, v in total_rewards_per_trick.items():
    print(f'Mean reward in trick number {i}: {np.mean(v)}')
