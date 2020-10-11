from ..rendering.game_window import GameWindowStatic
import pyglet
import threading
import os

class Viewer():
    """Class creates and controls windows for rendering environment state (with render_mode='human')."""
    def __init__(self):
        """Initialize Viewer."""
        self.window = GameWindowStatic()
        self.window_running = False
        pyglet.resource.path = [os.path.join(os.path.dirname(__file__), '..', 'resources', 'card_back'),
                                os.path.join(os.path.dirname(__file__), '..', 'resources', 'card_front')]
        pyglet.resource.reindex()

    def init_view(self, players_hands=None, contract=1, trump=1, dummy='S'):
        """
        Creates initial view.

        Args:
            players_hands (dict): each player's cards
            contract (int): contract value
            trump (int or None): trump coded as integer: None - no trump, 0 - clubs, 1 - diamonds, 2 - hearts, 3 - clubs
            dummy (str): dummy's position
        """
        self.window._init_cards(players_hands)
        self.window._set_contract(contract, trump)
        self.window._set_dummy(dummy)
        self.window_running = True

    def render_state(self, state):
        """
        Renders view according to given state.

        Args:
            state (dict): environment state containing all necessary information
        """
        self.window._draw_table(state['table'])
        self.window._draw_hands(state['hands'])
        self.window._draw_played(state['played_tricks'])
        self.window._draw_tricks(state['won_tricks']['N'], state['won_tricks']['E'])
        self.window.update()

    def close(self):
        """Close viewer's window."""
        self.window.clear()
        self.window.close()

    def reset(self):
        """Close viewer's window and create new empty."""
        self.window.close()
        self.window = GameWindowStatic()
        self.window_running = False
