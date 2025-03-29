import random

import numpy as np
from constants import *
from collections import defaultdict, namedtuple

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

        drawn_card = deck.deal()
        self.hand.add_card(drawn_card)
        
        if self.get_score() > 21:
            self.busted = True
            self.finishedTurn = True
        
    def stand(self):
        self.busted = False
        self.finishedTurn = True
        
    def play(self,deck, state):
        self.draw(deck)
        
    def get_score(self):
        return self.hand.score()
    
class Dealer(Player):
    
    def __init__(self):
        super().__init__()
        self.name = "Dealer"
        
    def draw(self, deck):
        # Changed from draw(self, deck, state) to match parent class
        super().draw(deck)
        
    def play(self,deck, state):
        self.hand.calc_hand()
        
        if self.hand.value >= 17:
            self.stand()
        else:
            self.draw(deck)
    
class AIPlayer(Player):
    
    def __init__(self, e=0.1, gamma=1, alpha=0.02):
        super().__init__()
        self.name = "AI"
        self.e = e
        self.gamma = gamma
        self.alpha = alpha
        self.Q = defaultdict(lambda: np.zeros(2))  
        self.currentEpisode = [] 

    def play(self, otherPlayerHand, dealerHand):
        """       
        print(f"Cartes du joueur 0: {self.hand}")
        print(f"Score: {self.hand.score()}")
        print(f"Cartes du joueur 1: {otherPlayerHand}")
        print(f"Score: {otherPlayerHand.score()}")
        print(f"Cartes du Dealer: {dealerHand}")
        print(f"Score: {dealerHand.score()}") 
        """
        currentState = self.createStateValues(otherPlayerHand, dealerHand)
        print(f"Current state: {currentState}")
        action = self.genAction(currentState, self.e, self.Q)
        print(f"Action: {action}")
        self.currentEpisode.append((currentState, action, None))

        return action


    # créer un etat en fonction du score de la main de l'ia de l'autre joueur et du dealer 
    def createStateValues(self,otherPlayerHand, dealerHand ):
        return  self.hand.score(), otherPlayerHand.score(), dealerHand.score() 

    def genAction(self, state, e, Q):
        probHit = Q[state][1]
        probStick = Q[state][0]
        
        if probHit>probStick:
            probs = [e, 1-e]
        elif probStick>probHit:
            probs = [1-e, e]
        else:
            probs = [0.5, 0.5]
            
        action = np.random.choice(np.arange(2), p=probs)   
        return action 


    def setQ(self): 
        if not self.currentEpisode:
            return True
        
        for t in range(len(self.currentEpisode)):
            state, action, _ = self.currentEpisode[t]
            
            # Calculer la récompense future avec discount
            Gt = 0
            for i in range(t, len(self.currentEpisode)):
                _, _, reward = self.currentEpisode[i]
                Gt += reward * (self.gamma ** (i - t))
            
            # Mise à jour de Q
            self.Q[state][action] += self.alpha * (Gt - self.Q[state][action])
        
        self.currentEpisode = []
        return True


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
        self.gameOver = False
                
        
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
        self.players[1].setQ()
        self.players[1].currentEpisode = []

        
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
            player.currentEpisode = [] 
            
    def check_winner(self):
        dealer_score = self.dealer.get_score()
        
        if dealer_score > 21:
            winners = [player.name for player in self.players if player.get_score() <= 21]
            if winners:
                print(f"Winner(s): {', '.join(winners)}")
            else:
                print("No winners, everyone busted.")
            return
        
        winners = []
        ties = []
        
        for player in self.players:
            if not player.busted:
                player_score = player.get_score()
                if player_score > dealer_score:
                    winners.append(player.name)
                elif player_score == dealer_score:
                    ties.append(player.name)
        
        if winners:
            print(f"Winner(s): {', '.join(winners)}")
        if ties:
            print(f"Tie(s): {', '.join(ties)}")
        if not winners and not ties:
            print(f"Dealer wins with score {dealer_score}")

        ai_player = self.players[1]
        ai_score = ai_player.get_score()
        human_score = self.players[0].get_score()

        reward = -1

        if not ai_player.busted:
            if (ai_score > human_score or self.players[0].busted) and (ai_score > dealer_score or dealer_score > 21):
                reward = 1
            elif (ai_score == human_score and (ai_score > dealer_score or dealer_score > 21)) or (ai_score == dealer_score and (ai_score > human_score or self.players[0].busted)):
                reward = 0

        for i in range(len(ai_player.currentEpisode)):
            state, action, _ = ai_player.currentEpisode[i]
            ai_player.currentEpisode[i] = (state, action, reward)


        self.players[1].setQ()
        

    def update(self):
        all_players_finished = all(player.finishedTurn or player.busted for player in self.players)
        
        if all_players_finished:
            return True 
       
        if self.players[self.playerTurn].finishedTurn or self.players[self.playerTurn].busted:
            currrenPlayer = self.playerTurn
            self.playerTurn = (self.playerTurn + 1) % len(self.players)
            
            while (self.players[self.playerTurn].finishedTurn or self.players[self.playerTurn].busted):
                if (self.playerTurn + 1) % len(self.players) == currrenPlayer:
                    return True 
        
        return False


