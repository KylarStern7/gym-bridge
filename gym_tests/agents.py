import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization
from _collections import deque
import random


# noinspection SpellCheckingInspection
class AgentDQN:
    def __init__(self, gamma, max_experiences, min_experiences, batch_size, epsilon):
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = 0.99
        self.online_network = self.neural_model()
        self.target_network = self.neural_model()
        self.copy_weights()
        self.experience = deque(maxlen=max_experiences)
        self.min_experiences = min_experiences

    def neural_model(self):
        #input_x = Input(shape=(2852,))
        #x = Dense(1024, activation='tanh')(input_x)
        #x = Dense(2048, activation='tanh')(x)
        #x = Dense(2048, activation='tanh')(x)
        #output_x = Dense(52, activation='linear')(x)

        #input_x = Input(shape=(56,))
        #x = Dense(128, activation='tanh')(input_x)
        #x = Dense(256, activation='tanh')(x)
        #x = Dense(256, activation='tanh')(x)
        #output_x = Dense(52, activation='linear')(x)

        input_x = Input(shape=(56,))
        x = Dense(128, activation='tanh')(input_x)
        x = Dense(256, activation='tanh')(x)
        output_x = Dense(52, activation='linear')(x)

        model = Model(input_x, output_x)
        model.compile(optimizer='Adamax', loss='mse')

        return model

    def train(self):
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

    def act(self, states):
        if np.random.random() < self.epsilon:
            return np.random.choice(52)
        else:
            preds = self.online_network.predict(np.atleast_2d(states))[0]
            return np.random.choice(np.argwhere(preds == np.max(preds)).flatten())

    def copy_weights(self):
        self.target_network.set_weights(self.online_network.get_weights())

    def add_experience(self, exp):
        self.experience.append(exp)

    def save_model(self):
        self.online_network.save('model.h5')


class AgentRandom(object):
    """The world's simplest agent!"""
    def __init__(self):
        self.available_actions = None

    def act(self, available_actions, observation, reward, done):
        self.available_actions = available_actions
        return random.choice(self.available_actions)


class AgentDummy(object):
    def __init__(self, partner):
        self.available_actions = None
        self.partner = partner

    def act(self, available_actions, observation, reward, done):
        self.available_actions = available_actions
        return self.partner.act(available_actions, observation, reward, done)