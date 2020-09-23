import numpy as np

class CardList(list):

    def __init__(self):
        super(CardList, self).__init__()

    def add_cards(self, cards_list):
        assert isinstance(cards_list, (int, list)), 'TypeError: Given argument "cards_list" must be one of (int, list).'
        if isinstance(cards_list, int):
            self.append(cards_list)
        elif isinstance(cards_list, list):
            self.extend(cards_list)
        return self

    def remove_card(self, card=None):
        if card is None:
            card = self.pop()
        elif isinstance(card, int):
            self.remove(card)
        elif isinstance(card, list):
            card = self.convert_multi_binary_to_integer(card)
            self.remove(card)
        return card

    def is_empty(self):
        empty = False
        if not self:
            empty = True
        return empty

    def get_random_card(self):
        if not self.is_empty():
            card = np.random.choice(self)
        else:
            card = None
        return card

    def get_suit_cards(self, suit):
        assert suit in (0, 1, 2, 3), "Given suit doesn't exist. Possible suits are: {0,1,2,3}."
        return [x for x in self if x % 4 == suit]

    def get_cards_multi_binary(self):
        return [1 if card in self else 0 for card in range(52)]

    def convert_multi_binary_to_integer(self, card_multi_binary):
        return int(np.argmax(card_multi_binary))

    def one_card_power(self, current_suit, trump):
        assert len(self) == 1, "Can't count power of multiple cards"
        card = self[0]
        if trump is not None and card % 4 == trump:
            card += 200
        elif card % 4 == current_suit:
            card += 100
        return card

    def sort_cards(self):
        self.sort()
        self.sort(key=lambda x: x % 4)

    def human_friendly_print(self):
        cards_meaning = {}
        for i in range(52):
            suit = i % 4
            value = int(i/4)
            if value < 9:
                value_name = str(value+2)
            elif value == 9:
                value_name = 'J'
            elif value == 10:
                value_name = 'Q'
            elif value == 11:
                value_name = 'K'
            elif value == 12:
                value_name = 'A'
            else:
                raise Exception("Card doesn't exist")

            if suit == 0:
                suit_name = '\u2663'
            elif suit == 1:
                suit_name = '\u2666'
            elif suit == 2:
                suit_name = '\u2665'
            elif suit == 3:
                suit_name = '\u2660'
            else:
                raise Exception("Card doesn't exist")

            cards_meaning[i] = value_name+suit_name
        cards_meaning[-1] = "WAIT"

        return "["+", ".join([cards_meaning[x] for x in self])+"]"

    def __repr__(self):
        return self.human_friendly_print()
