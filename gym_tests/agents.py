import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization
from _collections import deque
import random


# noinspection SpellCheckingInspection
class AgentDQN:
    def __init__(self, gamma, max_experiences, min_experiences, batch_size, epsilon, model=None):
        """
        Initialization of the agent.

        Args:
            gamma (float): discount rate
            max_experiences (int): maximum number of stored training samples
            min_experiences (int): minumum number of collected samples to perform training
            batch_size (int): batch size used in training process
            epsilon (float): probability of pikcing random action
            model (object): agent's neural network model
        """

        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.online_network = self.neural_model() if model is None else model
        self.target_network = self.neural_model() if model is None else model
        self.copy_weights()
        self.experience = deque(maxlen=max_experiences)
        self.min_experiences = min_experiences

    def neural_model(self):
        """
        Creates and compiles agent's neural network.

        Returns:
            Neural network model.
        """
        input_x = Input(shape=(56,))
        x = Dense(128, activation='tanh')(input_x)
        x = Dense(256, activation='tanh')(x)
        x = Dense(512, activation='tanh')(x)
        output_x = Dense(52, activation='linear')(x)

        model = Model(input_x, output_x)
        model.compile(optimizer='Adamax', loss='mse')

        return model

    def train(self):
        """Select random batch from agent's memory and fit neural network on it."""
        if len(self.experience) < self.min_experiences:
            return 0

        minibatch = random.sample(self.experience, self.batch_size)
        states = []
        targets = []

        for state, action, reward, next_state, done in minibatch:

            target = self.online_network.predict(np.atleast_2d(state))

            if done:
                target[0][action] = reward
            else:
                t = self.target_network.predict(np.atleast_2d(next_state))
                target[0][action] = reward + self.gamma * np.amax(t)

            states.append(state)
            targets.append(target[0])

        states = np.asarray(states)
        targets = np.asarray(targets)
        self.online_network.fit(states, targets, epochs=1, verbose=0)

    def act(self, observation):
        """
        Agent chooses action to perform based on it's logic.

        Args:
            observation (object): agent's observation of current environment's state.

        Returns:
            Action to perform.
        """
        if np.random.random() < self.epsilon:
            return np.random.choice(52)
        else:
            preds = self.online_network.predict(np.atleast_2d(observation))[0]
            return int(np.random.choice(np.argwhere(preds == np.max(preds)).flatten()))

    def copy_weights(self):
        """Copies weights from online_network to target_network."""
        self.target_network.set_weights(self.online_network.get_weights())

    def add_experience(self, exp):
        """
        Adds experience tuple to agent's memory.

        Args:
            exp (tuple): sample of agent's experience. Tuple contains previous_observation, performed_action,
            received_reward, new_observation and done_information.
        """
        self.experience.append(exp)

    def save_model(self):
        """Saves agent's neural network to "model.h5" file"""
        self.online_network.save('model.h5')


class AgentRandom(object):
    """Basic agent performing random actions."""
    def __init__(self):
        """Initialization of the agent."""
        self.available_actions = None

    def act(self, available_actions, observation, reward, done):
        """
        Agent chooses action to perform based on it's logic.

        Args:
            available_actions (list): valid actions in current environment's state.
            observation (object): agent's observation of current environment's state.
            reward (float): reward value received after last performed action.
            done (bool): indicates if episode has ended.

        Returns:
            Action to perform.
        """
        self.available_actions = available_actions
        return random.choice(self.available_actions)


class AgentDummy(object):
    """Basic agent letting its partner perform actions."""
    def __init__(self, partner):
        """
        Initialization of the agent.

        Args:
            partner: Other agent that will choose actions to perform.
        """
        self.available_actions = None
        self.partner = partner

    def act(self, available_actions, observation, reward, done):
        """
        Agent chooses action to perform based on it's logic.

        Args:
            available_actions (list): valid actions in current environment's state.
            observation (object): agent's observation of current environment's state.
            reward (float): reward value received after last performed action.
            done (bool): indicates if episode has ended.

        Returns:
            Action to perform.
        """
        self.available_actions = available_actions
        return self.partner.act(available_actions, observation, reward, done)