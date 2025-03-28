import random
from constants import *
from collections import namedtuple

class Deck:
    def __init__(self):
        self.cards = []
        self.dealt_cards = []
        self.build()

    def build(self):
        for value in RANKS:
            for suit in SUITS:
                self.cards.append((value, suit))
  
    def shuffle(self):
        random.shuffle(self.cards)
        
    def deal(self):
        if len(self.cards) > 1:
            drawn_card = self.cards.pop()
            self.dealt_cards.append(drawn_card)
            return drawn_card

        else:
            print("no more cards in deck")
            
class Hand(Deck):
    def __init__(self):
        self.cards = []
        self.card_img = []
        self.value = 0
        
        self.nbAces = 0

    def add_card(self, card):
        self.cards.append(card)

    def calc_hand(self):
        self.value = 0
        first_card_index = [a_card[0] for a_card in self.cards]
        non_aces = [c for c in first_card_index if c != 'A']
        aces = [c for c in first_card_index if c == 'A']

        for card in non_aces:
            if card in 'JQK':
                self.value += 10
            else:
                self.value += int(card)

        for card in aces:
            if self.value <= 10:
                self.value += 11
            else:
                self.value += 1

        self.nbAces = len(aces)

    def display_cards(self):
        for card in self.cards:
            cards = "".join((card[0], card[1]))
            if cards not in self.card_img:
                self.card_img.append(cards)
                
    def score(self):
        self.value = 0
        self.calc_hand()
        return self.value
                
    def __str__(self):
        return ', '.join([f"{card[0]}{card[1]}" for card in self.cards])
    
    def __len__(self):
        return len(self.cards)
    
    def __iter__(self):
        return iter(self.cards)






GameState = namedtuple('State', ['dealerScore', 'humanScore','AIScore', 'hasAce'])
    

class Player:

    def __init__(self):
        self.finishedTurn = False
        self.busted = False
        self.hand = Hand()
        self.name = "Player"

    def draw(self, deck):
        self.finishedTurn = True
        drawn_card = deck.deal()
        self.hand.add_card(drawn_card)
        
        if self.get_score() > 21:
            self.busted = True
        
    def stand(self):
        self.finishedTurn = True
        
    def play(self,deck, state):
        self.draw(deck)
        
    def get_score(self):
        return self.hand.score()
    
class Dealer(Player):
    
    def __init__(self):
        super().__init__()
        self.name = "Dealer"
        
    def draw(self,deck, state):
        super().draw(deck,[])
        
    def play(self,deck, state):
        self.hand.calc_hand()
        
        if self.hand.value >= 17:
            self.stand()
        else:
            self.draw(deck)
    
class AIPlayer(Player):
    def __init__(self):
        super().__init__()
        self.name = "AI"
        
    def play(self,deck, state):
        self.draw(deck)
        if len(self.hand) > 3:
            self.stand()

class BlackJack:
    def __init__(self):
        self.players = []
        
        self.playerTurn = 0

        """ for i in range(PLAYERCOUNT):
            self.players.append(Player()) """
            
        self.players.append(Player())
        self.players.append(AIPlayer())

        self.deck = Deck()
        self.deck.shuffle()
        
        self.dealer = Dealer()
        
        self.reset_players()
                
        
    def print_hands(self):
        print("Dealer's hand:")
        print(self.dealer.hand)
        for i, player in enumerate(self.players):
            print(f"Player {i + 1}'s hand:")
            print(player.hand)
        
        
    def playTurn(self):
        self.dealer.hand.calc_hand()
        for player in self.players:
            player.hand.calc_hand()
        
        self.dealer.play(self.deck, [])

        self.players[0].play(self.deck, [])
        
        GameState(self.dealer.hand.value, self.players[0].hand.value, self.players[1].hand.value, self.players[1].hand.hasAce )
        
        self.players[1].play(self.deck, [])
        
        self.print_hands()
        
    def reset_players(self):
        self.playerTurn = 0
        
        self.deck = Deck()
        self.deck.shuffle()
        
        self.dealer.hand = Hand()        
        
        for player in self.players:
            player.hand = Hand()
            
        self.dealer.draw(self.deck)
        self.players[0].draw(self.deck)
        self.players[0].draw(self.deck)
        self.players[1].draw(self.deck)
        self.players[1].draw(self.deck)
        
        for player in self.players:
            player.finishedTurn = False
            player.busted = False
            
    def check_winner(self):
        dealer_score = self.dealer.get_score()
        player_scores = [player.get_score() for player in self.players]
        
        if dealer_score > 21:
            winners = [player.name for player in self.players if player.get_score() <= 21]
            if winners:
            print(f"Winner(s): {', '.join(winners)}")
            else:
            print("No winners, everyone busted.")
            return
        
        # Check for players who didn't bust and have a higher score than the dealer
        winners = []
        for player in self.players:
            if not player.busted and player.get_score() > dealer_score:
            winners.append(player.name)
        
        # If no players beat the dealer, the dealer wins
        if not winners:
            print("Dealer wins!")
        else:
            print(f"Winner(s): {', '.join(winners)}")
        
    def update(self):
        if self.players[self.playerTurn].finishedTurn or self.players[self.playerTurn].busted:
            print(f"Condition: finishedTurn={self.players[self.playerTurn].finishedTurn}, busted={self.players[self.playerTurn].busted}")
            print(f"player turn is {self.playerTurn}")
            self.playerTurn += 1
            
            if self.playerTurn >= len(self.players):
                while not self.dealer.finishedTurn:
                    self.dealer.play(self.deck, [])
                    
                    self.playerTurn = 0
                self.check_winner()
            
        self.check_winner()
            
        return False
        
        
        
        