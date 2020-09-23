from ..rendering.game_window import GameWindowStatic
import pyglet
import threading
import os

class Viewer():
    def __init__(self):
        self.window = GameWindowStatic()
        self.window_running = False
        pyglet.resource.path = [os.path.join(os.path.dirname(__file__), '..', 'resources', 'card_back'),
                                os.path.join(os.path.dirname(__file__), '..', 'resources', 'card_front')]
        pyglet.resource.reindex()

    def init_view(self, player_hands=None, contract=1, trump=1, dummy='S'):
        self.window._init_cards(player_hands)
        self.window._set_contract(contract, trump)
        self.window._set_dummy(dummy)
        self.window_running = True

    def render_state(self, state):
        self.window._draw_table(state['table'])
        self.window._draw_hands(state['hands'])
        self.window._draw_played(state['played_tricks'])
        self.window._draw_tricks(state['won_tricks']['N'], state['won_tricks']['E'])
        self.window.update()

    def close(self):
        self.window.clear()
        self.window.close()

    def reset(self):
        self.window.close()
        self.window = GameWindowStatic()
        self.window_running = False
