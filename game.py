from blackjack_deck import *


PLAYERCOUNT = 2

class Player:

    def __init__(self):
        self.hand = Hand()

    def draw(self, deck):
        drawn_card = deck.deal()
        self.hand.add_card(drawn_card)

class BlackJack:
    def __init__(self):
        self.players = []

        for i in range(PLAYERCOUNT):
            self.players.append(Player())

        self.deck = Deck()
        self.deck.shuffle