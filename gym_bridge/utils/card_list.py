import numpy as np

class CardList(list):
    """
    Cards representation as list of integers.
    """
    def __init__(self):
        super(CardList, self).__init__()

    def add_cards(self, cards_list):
        """
        Add one or more cards.
        Args:
            cards_list (int or list): int or list of integers denoting cards to add.

        Returns:
            CardList: CardList object with added cards.
        """
        assert isinstance(cards_list, (int, list)), 'TypeError: Given argument "cards_list" must be one of (int, list).'
        if isinstance(cards_list, int):
            self.append(cards_list)
        elif isinstance(cards_list, list):
            self.extend(cards_list)
        return self

    def remove_card(self, card=None):
        """
        Remove card.
        Args:
            card (int or one hot encoding of int): int or one hot encoding of int denoting card to remove. If no card is
                given the last card in CardList is removed.

        Returns:
            int: Removed card.
        """
        if card is None:
            card = self.pop()
        elif isinstance(card, int):
            self.remove(card)
        elif isinstance(card, list):
            card = self.convert_multi_binary_to_integer(card)
            self.remove(card)
        return card

    def is_empty(self):
        """
        Check if CardList is empty.

        Returns:
            bool: indicates if CardList is empty
        """
        empty = False
        if not self:
            empty = True
        return empty

    def get_random_card(self):
        """
        Returns random card.

        Returns:
            int: one of cards from CardList.
        """
        if not self.is_empty():
            card = np.random.choice(self)
        else:
            card = None
        return card

    def get_suit_cards(self, suit):
        """
        Get all cards whit given suit.

        Args:
            suit (int): One of: 0 - clubs, 1 - diamonds, 2 - hearts, 3 - clubs.

        Returns:
            list: Cards with given suit.
        """
        assert suit in (0, 1, 2, 3), "Given suit doesn't exist. Possible suits are: {0,1,2,3}."
        return [x for x in self if x % 4 == suit]

    def get_cards_multi_binary(self):
        """
        Change CardList into "multi_binary" format.

        Returns:
            list: "multi_binary" format of CardList
        """
        return [1 if card in self else 0 for card in range(52)]

    def convert_multi_binary_to_integer(self, card_multi_binary):
        """
        Convert one card coded in "multi_binary" format into "integer" format.
        Args:
            card_multi_binary (list): card coded in "multi_binary" format

        Returns:
            int: card coded as integer
        """
        return int(np.argmax(card_multi_binary))

    def one_card_power(self, current_suit, trump):
        """
        Calculates "power" of single card.

        Power of card depends on card itself, current_suit and trump.

        Args:
            current_suit (int): One of: 0 - clubs, 1 - diamonds, 2 - hearts, 3 - clubs.
            trump (int): One of: None - "no trump", 0 - clubs, 1 - diamonds, 2 - hearts, 3 - clubs.

        Returns:
            int: card's "power"
        """
        assert len(self) == 1, "Can't count power of multiple cards"
        card = self[0]
        if trump is not None and card % 4 == trump:
            card += 200
        elif card % 4 == current_suit:
            card += 100
        return card

    def sort_cards(self):
        """Sort CardList by suit."""
        self.sort()
        self.sort(key=lambda x: x % 4)

    def human_friendly_print(self):
        """
        Present CardList as easily readable string.

        Returns:
            str: string representation of CardList
        """
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
