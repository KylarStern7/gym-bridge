import pyglet

import numpy as np
import sys


class CardImage(pyglet.sprite.Sprite, pyglet.event.EventDispatcher):
    def __init__(self, card_number=0, x=0, y=0, rotation=0, front=True, back_color='green', batch=None, group=None):
        self.front = front
        self.back_color = back_color
        self.front_image = pyglet.resource.image(str(card_number) + '.png')
        self.front_image.anchor_x = self.front_image.width / 2
        self.front_image.anchor_y = self.front_image.height / 2
        self.back_image = pyglet.resource.image(back_color + '_back.png')
        self.back_image.anchor_x = self.back_image.width / 2
        self.back_image.anchor_y = self.back_image.height / 2
        if self.front:
            super(CardImage, self).__init__(self.front_image, x=x, y=y, batch=batch, group=group)
        else:
            super(CardImage, self).__init__(self.back_image, x=x, y=y, batch=batch, group=group)
        self.scale = 0.15
        self.rotation = rotation

        self.new_x = self.x
        self.new_y = self.y
        self.new_rotation = self.rotation
        self.d_x = 0
        self.d_y = 0
        self.d_rotation = 0
        self.card_number = card_number

    def flip_card(self):
        if self.front:
            self.image = self.back_image
            self.front = False
        else:
            self.image = self.front_image
            self.front = True


class GameWindowStatic(pyglet.window.Window):
    def __init__(self):
        super(GameWindowStatic, self).__init__(1280, 800, resizable=True)
        # self.set_fullscreen()
        self.set_minimum_size(640, 480)
        self.set_caption("Bridge")

        self.label_batch = pyglet.graphics.Batch()
        self.card_batch = pyglet.graphics.Batch()
        self.batch_groups = [pyglet.graphics.OrderedGroup(x) for x in range(13)]

        self.border_dst = 120
        self.label_N = pyglet.text.Label('N', x=self.width / 2, y=self.height - 25, font_size=20,
                                         batch=self.label_batch)
        self.label_E = pyglet.text.Label('E', x=self.width - 20, y=self.height / 2, font_size=20,
                                         batch=self.label_batch)
        self.label_S = pyglet.text.Label('S', x=self.width / 2, y=15, font_size=20, batch=self.label_batch)
        self.label_W = pyglet.text.Label('W', x=20, y=self.height / 2, font_size=20, batch=self.label_batch)
        self.label_contract = pyglet.text.Label(x=self.width - 200, y=self.height - 50, font_size=20,
                                                batch=self.label_batch)
        self.label_tricks = pyglet.text.Label('N-S: 0 E-W: 0', x=self.width - 200, y=self.height - 100, multiline=True,
                                              width=100, font_size=20, batch=self.label_batch)

        self.players_position = {'N': (self.width / 2, self.height - self.border_dst, 180),
                                 'E': (self.width - self.border_dst, self.height / 2, 270),
                                 'S': (self.width / 2, self.border_dst, 0),
                                 'W': (self.border_dst, self.height / 2, 90)}
        self.table_position = {'N': (self.width / 2, self.height / 2 + 100, 180),
                               'E': (self.width / 2 + 150, self.height / 2, 270),
                               'S': (self.width / 2, self.height / 2 - 100, 0),
                               'W': (self.width / 2 - 150, self.height / 2, 90)}
        self.cards = [None] * 52

        self.is_open = True

        self.dummy = None

    def on_draw(self):
        self.clear()
        self.label_N.draw()
        self.label_E.draw()
        self.label_S.draw()
        self.label_W.draw()
        self.label_contract.draw()
        self.label_tricks.draw()
        self.card_batch.draw()

        # self.flip()

    def on_resize(self, width, height):
        super().on_resize(width, height)
        self.players_position = {'N': (width / 2, height - self.border_dst, 180),
                                 'E': (width - self.border_dst, height / 2, 270),
                                 'S': (width / 2, self.border_dst, 0),
                                 'W': (self.border_dst, height / 2, 90)}
        self.table_position = {'N': (width / 2, height / 2 + 100, 180),
                               'E': (width / 2 + 150, height / 2, 270),
                               'S': (width / 2, height / 2 - 100, 0),
                               'W': (width / 2 - 150, height / 2, 90)}
        self.label_N.x = width / 2
        self.label_N.y = height - 25
        self.label_E.x = width - 20
        self.label_E.y = height / 2
        self.label_S.x = width / 2
        self.label_S.y = 15
        self.label_W.x = 15
        self.label_W.y = height / 2
        self.label_contract.x = width - 200
        self.label_contract.y = height - 50
        self.label_tricks.x = width - 200
        self.label_tricks.y = height - 100
        print(width, height)

    def on_close(self):
        sys.exit(0)

    def on_mouse_press(self, x, y, button, modifiers):
        print('aaaaa')

    def _set_contract(self, contract, trump):
        trump_dict = {None: 'NT',
                      0: '\u2663',
                      1: '\u2666',
                      2: '\u2665',
                      3: '\u2660'}
        self.label_contract.text = 'Contract: ' + str(contract) + trump_dict[trump]

    def _set_dummy(self, player=None):
        assert player is not None, "Player None"
        self.dummy = player

    def _init_cards(self, players_hands=None):
        assert players_hands is not None, "Player_hands None"
        for player, card_list in players_hands.items():
            for i, card in enumerate(card_list):
                self.cards[card] = CardImage(card, x=self.width / 2, y=self.height / 2, batch=self.card_batch,
                                             group=self.batch_groups[i])

    def _draw_hands(self, players_hands=None):
        assert players_hands is not None, "Player_hands None"
        shift = 30
        for player, card_list in players_hands.items():
            n_cards = len(card_list)
            if player in ('N', 'S'):
                for i, card in enumerate(card_list):
                    self.cards[card].update(self.players_position[player][0] - int(n_cards / 2) * shift + shift * i,
                                            self.players_position[player][1],
                                            self.players_position[player][2])
            else:
                for i, card in enumerate(card_list):
                    self.cards[card].update(self.players_position[player][0],
                                            self.players_position[player][1] + int(n_cards / 2) * shift - shift * i,
                                            self.players_position[player][2])

    def _draw_dummy_hand(self, card_list):
        shift_dummy_cover = 20
        shift_dummy_between = 100
        if self.dummy in ('N', 'S'):
            for suit in range(4):
                for x in np.argwhere(np.array(card_list) % 4 == suit):
                    self.cards[card_list[x[0]]].update(
                        self.players_position[self.dummy][0] - int(n_cards / 2) * shift + shift * i,
                        self.players_position[self.dummy][1],
                        self.players_position[self.dummy][2])
        else:
            pass

    def _draw_table(self, table=None):
        assert table is not None, "Player_hands None"
        for player, card in table.items():
            if card:
                self.cards[card[0]].update(*self.table_position[player])
            else:
                pass

    def _draw_played(self, played_tricks=None):
        assert played_tricks is not None, "Player_hands None"
        for trick in played_tricks.values():
            for player, card_list in trick.items():
                for i, card in enumerate(card_list):
                    self.cards[card].update(self.width - 50, 50, 0)
                    if self.cards[card].front:
                        self.cards[card].flip_card()

    def _draw_tricks(self, trick_N, trick_E):
        self.label_tricks.text = 'N-S: {} E-W: {}'.format(trick_N, trick_E)

    def draw_state(self, game_state):
        ...

    def update(self):
        self.clear()
        self.dispatch_events()
        self.on_draw()
        self.flip()
