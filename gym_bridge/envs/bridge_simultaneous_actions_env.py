import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from random import shuffle, choice
from copy import deepcopy
from ..spaces.multi_binary_limited import MultiBinaryLimited
from ..spaces.multi_integer_limited import MultiIntegerLimited
from ..utils.card_list import CardList
from ..rendering.rendering import Viewer


# noinspection SpellCheckingInspection,PyTypeChecker
class BridgeSimultaneousActionsEnv(gym.Env):
    metadata = {'render.modes': ['human', 'ansi'],
                'action_space.modes': ['integer', 'multi_binary'],
                'observation_space.modes': ['integer', 'multi_binary', 'mixed'],
                'reward.modes': ['win_contract', 'win_tricks', 'win_points', 'play_cards']}

    def __init__(self, action_space_mode='integer', observation_space_mode='integer', reward_mode='play_cards',
                 render_mode='ansi'):
        self.action_space_mode = action_space_mode
        self.observation_space_mode = observation_space_mode
        self.reward_mode = reward_mode
        self.render_mode = render_mode

        self.players = ['N', 'E', 'S', 'W']

        self.action_space = self._init_action_space()
        self.observation_space = self._init_observation_space()

        self.state = {'active_player': None,
                      'hands': {'N': CardList(),
                                'E': CardList(),
                                'S': CardList(),
                                'W': CardList()},
                      'table': {'N': CardList(),
                                'E': CardList(),
                                'S': CardList(),
                                'W': CardList()},
                      'played_tricks': {i: {plr: CardList() for plr in self.players} for i in range(13)},
                      'won_tricks': {plr: 0 for plr in self.players},
                      'current_suit': None,
                      }
        self.trump = None
        self.contract_value = None
        self.players_roles = None

        self.n_cards_on_table = 0
        self.rewards = {'N': 0, 'E': 0, 'S': 0, 'W': 0}

        self.render_state = deepcopy(self.state)
        self.viewer = None
        self.tricks_played = 0

    def step(self, actions):
        self._game_controller(actions)
        observations = {player: self.get_player_observation(player) for player in self.players}
        rewards = self.rewards
        dones = {player: self.state.get('hands').get(player).is_empty()
                 for player in self.players}
        info = {}

        return observations, rewards, dones, info

    def reset(self, initial_state=None):
        if initial_state is None:
            self.tricks_played = 0
            self.n_cards_on_table = 0
            self._set_players_roles(choice(self.players))
            self.trump = choice([0, 1, 2, 3, None])
            self.contract_value = choice([1, 2, 3, 4, 5, 6, 7])
            self.state = {'active_player': self.players_roles.get('defender_1', 'E'),
                          'hands': {'N': CardList(),
                                    'E': CardList(),
                                    'S': CardList(),
                                    'W': CardList()},
                          'table': {'N': CardList(),
                                    'E': CardList(),
                                    'S': CardList(),
                                    'W': CardList()},
                          'played_tricks': {i: {plr: CardList() for plr in self.players} for i in range(13)},
                          'won_tricks': {plr: 0 for plr in self.players},
                          'current_suit': None,
                          }
            self._deal_random_cards()
            self.render_state = deepcopy(self.state)
            if self.viewer:
                self.viewer.reset()
        else:
            self.tricks_played = 0
            self.n_cards_on_table = 0
            self._set_players_roles(initial_state.get('player', 'N'))
            self.trump = initial_state.get('trump', 0)
            self.contract_value = initial_state.get('contract_value', 0)
            self.state = {'active_player': self.players_roles.get('defender_1', 'E'),
                          'hands': {'N': CardList().add_cards(initial_state.get('hands').get('N')),
                                    'E': CardList().add_cards(initial_state.get('hands').get('E')),
                                    'S': CardList().add_cards(initial_state.get('hands').get('S')),
                                    'W': CardList().add_cards(initial_state.get('hands').get('W'))},
                          'table': {'N': CardList(),
                                    'E': CardList(),
                                    'S': CardList(),
                                    'W': CardList()},
                          'played_tricks': {i: {plr: CardList() for plr in self.players} for i in range(13)},
                          'won_tricks': {plr: 0 for plr in self.players},
                          'current_suit': None,
                          }
            self.render_state = deepcopy(self.state)
            if self.viewer:
                self.viewer.reset()

        return {player: self.get_player_observation(player) for player in self.players}

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode=None):
        mode = self.render_mode if mode is None else mode
        if mode == 'ansi':
            suits_dict = {0: '\u2663',
                          1: '\u2666',
                          2: '\u2665',
                          3: '\u2660',
                          None: 'NT'}
            render_info = f'Players roles: {self.players_roles}\n' \
                          f'Players hands: {self.render_state.get("hands")}\n' \
                          f'Table: {self.render_state.get("table")}\n' \
                          f'Contract: {self.contract_value}{suits_dict.get(self.trump)}\n' \
                          f'Won tricks: {self.render_state.get("won_tricks")}'
            return render_info
        elif mode == 'human':
            if self.viewer is None:
                self.viewer = Viewer()
            if not self.viewer.window_running:
                self.viewer.init_view(self.render_state['hands'], self.contract_value, self.trump)
            self.viewer.render_state(self.render_state)

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def get_available_actions(self, player=None):
        assert player is not None, "No selected player"
        if player == self.state['active_player']:
            if self.state['current_suit'] is None:
                available_actions = self.state['hands'].get(player, CardList())
            else:
                available_actions = self.state['hands'].get(player, CardList()).get_suit_cards(
                    self.state.get('current_suit'))
                if len(available_actions) < 1:
                    available_actions = self.state['hands'].get(player, CardList())

            if self.action_space_mode == 'multi_binary':
                available_actions_multi_binary = []
                for card in available_actions:
                    valid_action = [0] * 52
                    valid_action[card] = 1
                    available_actions_multi_binary.append(valid_action)
                available_actions = available_actions_multi_binary
        else:
            available_actions = CardList().add_cards(-1)
            if self.action_space_mode == 'multi_binary':
                available_actions = [].append([0] * 52)

        return available_actions

    def _set_players_roles(self, declarer='N'):
        if declarer in self.players:
            self.players_roles = {'declarer': declarer,
                                  'defender_1': self._get_next_player(declarer),
                                  'dummy': self._get_next_player(self._get_next_player(declarer)),
                                  'defender_2': self._get_next_player(
                                      self._get_next_player(self._get_next_player(declarer)))}
        else:
            raise Exception(f'Setting players roles failed. "{declarer}" is not a valid player.')

    def _get_next_player(self, player='N'):
        try:
            next_player = self.players[(self.players.index(player) + 1) % 4]
        except:
            raise (Exception(f'"{player}" is not a valid player.'))
        return next_player

    def _deal_random_cards(self):
        random_cards = list(range(52))
        shuffle(random_cards)
        self.state['hands']['N'].add_cards(random_cards[0:13]).sort_cards()
        self.state['hands']['E'].add_cards(random_cards[13:26]).sort_cards()
        self.state['hands']['S'].add_cards(random_cards[26:39]).sort_cards()
        self.state['hands']['W'].add_cards(random_cards[39:52]).sort_cards()

    def get_player_observation(self, player):
        observation = dict()
        if self.observation_space_mode == 'integer':
            observation['player_position'] = self.players.index(player)
            observation['dummy_position'] = self.players.index(self.players_roles['dummy'])
            observation['active_player_position'] = self.players.index(self.state['active_player'])
            observation['player_hand'] = self.state['hands'][player]
            observation['dummy_hand'] = [] if (self.tricks_played == 0) & (self.n_cards_on_table == 0) \
                else self.state['hands'][self.players_roles['dummy']]
            observation['table'] = {item[0]: item[1] for item in self.state['table'].items()}
            observation['played_tricks'] = {i: {item[0]: item[1] for item in self.state['played_tricks'][i].items()}
                                            for i in range(13)}
            observation['current_suit'] = self.state['current_suit']
            observation['trump'] = self.trump
            observation['contract_value'] = self.contract_value
            observation['won_tricks'] = self.state['won_tricks'][player]

        elif self.observation_space_mode == 'multi_binary':
            observation['player_position'] = [1 if plr == player else 0 for plr in self.players]
            observation['dummy_position'] = [1 if self.players_roles['dummy'] == plr else 0 for plr in self.players]
            observation['active_player_position'] = [1 if plr == self.state['active_player']
                                                     else 0 for plr in self.players]
            observation['player_hand'] = self.state['hands'][player].get_cards_multi_binary()
            observation['dummy_hand'] = [0 for _ in range(52)] if ((self.tricks_played == 0) &
                                                                   (self.n_cards_on_table == 0))\
                else self.state['hands'][self.players_roles['dummy']].get_cards_multi_binary()
            observation['table'] = {item[0]: item[1].get_cards_multi_binary() for item in self.state['table'].items()}
            observation['played_tricks'] = {i: {item[0]: item[1].get_cards_multi_binary()
                                                for item in self.state['played_tricks'][i].items()} for i in range(13)}
            observation['current_suit'] = [1 if i == self.state['current_suit'] else 0 for i in range(4)]
            observation['trump'] = [1 if i == self.trump else 0 for i in range(4)]
            observation['contract_value'] = [1 if i == self.contract_value else 0 for i in range(7)]
            observation['won_tricks'] = [1 if i == self.state['won_tricks'][player] else 0 for i in range(13)]

        elif self.observation_space_mode == 'mixed':
            observation['player_position'] = self.players.index(player)
            observation['dummy_position'] = self.players.index(self.players_roles['dummy'])
            observation['active_player_position'] = self.players.index(self.state['active_player'])
            observation['player_hand'] = self.state['hands'][player].get_cards_multi_binary()
            observation['dummy_hand'] = [0 for _ in range(52)] if ((self.tricks_played == 0) &
                                                                   (self.n_cards_on_table == 0))\
                else self.state['hands'][self.players_roles['dummy']].get_cards_multi_binary()
            observation['table'] = {item[0]: item[1].get_cards_multi_binary() for item in self.state['table'].items()}
            observation['played_tricks'] = {i: {item[0]: item[1].get_cards_multi_binary()
                                                for item in self.state['played_tricks'][i].items()} for i in range(13)}
            observation['current_suit'] = self.state['current_suit']
            observation['trump'] = self.trump
            observation['contract_value'] = self.contract_value
            observation['won_tricks'] = self.state['won_tricks'][player]

        return observation

    def _game_controller(self, actions):
        trick_winner = None
        actions_are_valid = {player: actions.get(player) in self.get_available_actions(player)
                             for player in self.players}

        if actions_are_valid.get(self.state.get('active_player')):
            card = self.state['hands'][self.state['active_player']].remove_card(
                actions.get(self.state['active_player']))
        else:
            card = self.state['hands'][self.state['active_player']].remove_card(
                choice(self.get_available_actions(self.state['active_player'])))

        self.state['table'][self.state['active_player']].add_cards(card)
        self.n_cards_on_table += 1

        self.render_state = deepcopy(self.state)

        if self.n_cards_on_table < 4:
            next_player = self._get_next_player(self.state['active_player'])
            if self.state['current_suit'] is None:
                self.state['current_suit'] = card % 4
        else:
            trick_winner = self._get_trick_winner()
            next_player = trick_winner
            self._clear_table()
            self.tricks_played += 1
            self.state['won_tricks'][trick_winner] += 1
            self.state['won_tricks'][self._get_next_player(self._get_next_player(trick_winner))] += 1

        self.rewards = self._get_rewards(trick_winner, actions_are_valid)
        self.state['active_player'] = next_player

        return next_player

    def _get_trick_winner(self):
        assert self.n_cards_on_table == 4, "Every player has to play a card."
        trick_winner = self.players[np.argmax([card.one_card_power(self.state['current_suit'], self.trump)
                                               for card in self.state['table'].values()])]

        return trick_winner

    def _get_rewards(self, trick_winner=None, chosen_cards_are_valid=None):
        if chosen_cards_are_valid is None:
            chosen_cards_are_valid = {'N': True, 'E': True, 'S': True, 'W': True}
        rewards = deepcopy(self.rewards)

        if self.reward_mode == 'win':
            if self.tricks_played == 13:
                if self.state['won_tricks'][self.players_roles['declarer']] >= self.contract_value + 6:
                    rewards[self.players_roles['declarer']] = 1
                    rewards[self.players_roles['dummy']] = 1
                    rewards[self.players_roles['defender_1']] = 0
                    rewards[self.players_roles['defender_2']] = 0
                else:
                    rewards[self.players_roles['declarer']] = 0
                    rewards[self.players_roles['dummy']] = 0
                    rewards[self.players_roles['defender_1']] = 1
                    rewards[self.players_roles['defender_2']] = 1
            else:
                pass

        elif self.reward_mode == 'win_tricks':
            if trick_winner is not None:
                rewards[trick_winner] = 1
                rewards[self._get_next_player(trick_winner)] = 0
                rewards[self._get_next_player(self._get_next_player(trick_winner))] = 1
                rewards[self._get_next_player(self._get_next_player(self._get_next_player(trick_winner)))] = 0
            else:
                pass

        elif self.reward_mode == 'win_points':
            if self.tricks_played == 13:
                if self.state['won_tricks'][self.players_roles['declarer']] >= self.contract_value + 6:
                    rewards[self.players_roles['declarer']] = 1
                    rewards[self.players_roles['dummy']] = 1
                    rewards[self.players_roles['defender_1']] = 0
                    rewards[self.players_roles['defender_2']] = 0
                else:
                    rewards[self.players_roles['declarer']] = 0
                    rewards[self.players_roles['dummy']] = 0
                    rewards[self.players_roles['defender_1']] = 1
                    rewards[self.players_roles['defender_2']] = 1
            else:
                pass

        elif self.reward_mode == 'play_cards':
            for player, valid_card in chosen_cards_are_valid.items():
                if valid_card:
                    rewards[player] = 0

        else:
            raise Exception(f'Reward mode "{self.reward_mode}" is not supported. Available reward'
                            f' modes are: {self.metadata["reward.modes"]}')
        for player, valid_card in chosen_cards_are_valid.items():
            if not valid_card:
                rewards[player] = -1000

        return rewards

    def _clear_table(self):
        # Move cards form "table" to "played"
        for player in self.players:
            self.state['played_tricks'][self.tricks_played][player].add_cards(self.state['table'][player].remove_card())
        self.n_cards_on_table = 0
        self.state['current_suit'] = None

    def _init_action_space(self):
        if self.action_space_mode == 'integer':
            action_space = MultiIntegerLimited(0, 1, -1, 51)
        elif self.action_space_mode == 'multi_binary':
            action_space = MultiBinaryLimited(52, 0, 1)
        else:
            raise Exception(f'Action space mode "{self.action_space_mode}" is not supported. Available action space'
                            f'modes are: {self.metadata["action_space.modes"]}')
        return action_space

    def _init_observation_space(self):
        if self.observation_space_mode == 'integer':
            observation_space = spaces.Dict({
                'player_position': spaces.Discrete(4),
                'dummy_position': spaces.Discrete(4),
                'active_player_position': spaces.Discrete(4),
                'player_hand': MultiIntegerLimited(0, 13, 0, 51),
                'dummy_hand': MultiIntegerLimited(0, 13, 0, 51),
                'table': spaces.Dict({
                    'N': MultiIntegerLimited(0, 1, 0, 51),
                    'E': MultiIntegerLimited(0, 1, 0, 51),
                    'S': MultiIntegerLimited(0, 1, 0, 51),
                    'W': MultiIntegerLimited(0, 1, 0, 51)
                }),
                'played_tricks': spaces.Dict(
                    {i: spaces.Dict({plr: MultiIntegerLimited(0, 13, 0, 51) for plr in self.players})
                     for i in range(13)}
                ),
                'current_suit': MultiIntegerLimited(0, 1, 0, 3),
                "trump": MultiIntegerLimited(0, 1, 0, 3),
                "contract_value": MultiIntegerLimited(1, 1, 1, 7),
                "won_tricks": MultiIntegerLimited(1, 1, 1, 13)
            })
        elif self.observation_space_mode == 'multi_binary':
            observation_space = spaces.Dict({
                'player_position': MultiBinaryLimited(4, 1, 1),
                'dummy_position': MultiBinaryLimited(4, 1, 1),
                'active_player_position': MultiBinaryLimited(4, 1, 1),
                'player_hand': MultiBinaryLimited(52, 0, 13),
                'dummy_hand': MultiBinaryLimited(52, 0, 13),
                'table': spaces.Dict({
                    'N': MultiBinaryLimited(52, 0, 1),
                    'E': MultiBinaryLimited(52, 0, 1),
                    'S': MultiBinaryLimited(52, 0, 1),
                    'W': MultiBinaryLimited(52, 0, 1)
                }),
                'played_tricks': spaces.Dict(
                    {i: spaces.Dict({plr: MultiBinaryLimited(52, 0, 1) for plr in self.players}) for i in range(13)}
                ),
                'current_suit': MultiBinaryLimited(4, 0, 1),
                "trump": MultiBinaryLimited(4, 0, 1),
                "contract_value": MultiBinaryLimited(7, 1, 1),
                "won_tricks": MultiBinaryLimited(13, 0, 1)
            })
        elif self.observation_space_mode == 'mixed':
            observation_space = spaces.Dict({
                'player_position': spaces.Discrete(4),
                'dummy_position': spaces.Discrete(4),
                'active_player_position': spaces.Discrete(4),
                'player_hand': MultiBinaryLimited(52, 0, 13),
                'dummy_hand': MultiBinaryLimited(52, 0, 13),
                'table': spaces.Dict({
                    'N': MultiBinaryLimited(52, 0, 1),
                    'E': MultiBinaryLimited(52, 0, 1),
                    'S': MultiBinaryLimited(52, 0, 1),
                    'W': MultiBinaryLimited(52, 0, 1)
                }),
                'played_tricks': spaces.Dict(
                    {i: {plr: MultiBinaryLimited(52, 0, 1) for plr in self.players} for i in range(13)}
                ),
                'current_suit': MultiIntegerLimited(0, 1, 0, 3),
                "trump": MultiIntegerLimited(0, 1, 0, 3),
                "contract_value": MultiIntegerLimited(1, 1, 1, 7),
                "won_tricks": MultiIntegerLimited(1, 1, 1, 13)
            })
        else:
            raise Exception(f'Observation space mode "{self.observation_space_mode}" is not supported. Available'
                            f' observation space modes are: {self.metadata["observation_space.modes"]}')

        return observation_space
