


GameState = namedtuple('State', ['dealerScore', 'humanScore','AIScore', 'hasAce'])
    

class Player:

    def __init__(self):
        self.hand = Hand()

    def draw(self, deck):
        drawn_card = deck.deal()
        self.hand.add_card(drawn_card)
        
    def stand(self):
        pass
        
    def play(self,deck, state):
        self.draw(deck)
    
class Dealer(Player):
    
    def __init__(self):
        self.hand = Hand()

    def draw(self, deck):
        drawn_card = deck.deal()
        self.hand.add_card(drawn_card)
        
    def stand(self):
        pass
        
    def play(self,deck, state):
        self.hand.calc_hand()
        
        if self.hand.value >= 17:
            self.stand()
        else:
            self.draw(deck)
    
class AIPlayer(Player):
    def __init__(self):
        self.hand = Hand()

    def draw(self, deck):
        drawn_card = deck.deal()
        self.hand.add_card(drawn_card)
        
    def stand(self):
        pass
        
    def play(self,deck, state):
        self.draw(deck)

class BlackJack:
    def __init__(self):
        self.players = []

        """ for i in range(PLAYERCOUNT):
            self.players.append(Player()) """

        self.deck = Deck()
        self.deck.shuffle()
        
        self.dealer = Dealer()
        
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
        
        
        
        
