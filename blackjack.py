import random
import os 
import pickle
import numpy as np
from constants import *
from collections import defaultdict, namedtuple

from collections import deque

# Keep track of the last 10,000 outcomes
recent_results = deque(maxlen=10000)


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
    
    def __init__(self, e=1.0, gamma=1.00, alpha=0.02):
        super().__init__()
        self.name = "AI"
        self.e = e
        self.gamma = gamma
        self.alpha = alpha
        self.Q = defaultdict(lambda: np.zeros(2))  
        self.currentEpisode = [] 

    def play(self, otherPlayerHand, dealerHand):
        # Get normalized state
        currentState = self.createStateValues(otherPlayerHand, dealerHand)
        
        # Select action based on state (using your existing genAction method)
        action = self.genAction(currentState, self.e, self.Q)
        
        # Store state-action pair with initial None reward (will be updated later)
        self.currentEpisode.append((currentState, action, None))
        
        return action


    # créer un etat en fonction du score de la main de l'ia de l'autre joueur et du dealer 
    def createStateValues(self, otherPlayerHand, dealerHand):
        """
        Normalize the state values for better generalization.
        - Cap player score at 21
        - Group dealer scores
        """
        # Cap player score at 21 to reduce state space
        player_score = min(21, self.hand.score())
        
        # Get other scores
        other_score = otherPlayerHand.score()
        dealer_score = dealerHand.score()
        
        # Track if we have usable ace (value 11) for better state representation
        has_usable_ace = self.hand.nbAces > 0 and player_score <= 21
        
        # Normalize dealer score - group high values to reduce state space
        if dealer_score > 19:
            dealer_score = 19
        
        return (player_score, other_score, dealer_score, has_usable_ace)

    def genAction(self, state, e, Q):
        # Always hit if score <= 11
        #if self.get_score() <= 11:
        #    return 1
        
        # With probability e, choose a random action (exploration)
        if random.random() < e:
            return random.choice([0, 1])
        
        # With probability 1-e, choose the best action (exploitation)
        if Q[state][0] > Q[state][1]:
            return 0  # Stand
        elif Q[state][1] > Q[state][0]:
            return 1  # Hit
        else:
            # If both actions have equal value, choose randomly
            return random.choice([0, 1])



    def setQ(self):
        if not self.currentEpisode:
            return True

        visits_count = defaultdict(int)
        Gt = 0  
        # Parcours en ordre inverse, calculer Gt = r + gamma * Gt AVANT la mise à jour
        for t in reversed(range(len(self.currentEpisode))):
            state, action, reward = self.currentEpisode[t]
            Gt = reward + self.gamma * Gt
            visits_count[(state, action)] += 1  
            self.Q[state][action] += self.alpha * (Gt - self.Q[state][action])
            
        self.currentEpisode = []
        return True



    def save_q(self, filename="q_table.pkl"):
        """Sauvegarde la table Q dans un fichier"""
        try:
            # Convertir defaultdict en dict standard pour la sauvegarde
            q_dict = dict(self.Q)
            with open(filename, 'wb') as f:
                pickle.dump(q_dict, f)
            print(f"Table Q sauvegardée dans {filename}")
        except Exception as e:
            print(f"Erreur lors de la sauvegarde de la table Q: {e}")

    def load_q(self, filename="q_table.pkl"):
        """Charge la table Q depuis un fichier"""
        try:
            if os.path.exists(filename):
                with open(filename, 'rb') as f:
                    q_dict = pickle.load(f)
                # Transfert des valeurs dans le defaultdict
                for k, v in q_dict.items():
                    self.Q[k] = v
                print(f"Table Q chargée depuis {filename}")
                return True
            else:
                print("Aucun fichier de table Q trouvé, utilisation d'une nouvelle table")
                return False
        except Exception as e:
            print(f"Erreur lors du chargement de la table Q: {e}")
            return False 

    def draw(self, deck):
        # Store old score before drawing
        old_score = self.get_score()
        
        # Draw card (use parent method)
        drawn_card = deck.deal()
        self.hand.add_card(drawn_card)
        
        # Calculate new score
        new_score = self.get_score()
        
        # Provide intermediate reward based on score improvement
        if len(self.currentEpisode) > 0 and self.currentEpisode[-1][1] == 1:  # If last action was Hit
            last_state, last_action, _ = self.currentEpisode[-1]
            
            # Calculate intermediate reward
            intermediate_reward = 0
            
            # Reward for improving hand without busting
            if new_score > old_score and new_score <= 21:
                intermediate_reward = 0.1
            # Small penalty for drawing but not improving (like drawing a low card when already high)
            elif new_score == old_score:
                intermediate_reward = -0.05
            # Larger penalty for busting
            elif new_score > 21:
                intermediate_reward = -0.2
                
            # Update the last action in the episode with the intermediate reward
            self.currentEpisode[-1] = (last_state, last_action, intermediate_reward)
        
        # Check if busted
        if self.get_score() > 21:
            self.busted = True
            self.finishedTurn = True

class BlackJack:
    def __init__(self):
        self.players = []
        
        self.playerTurn = 0

        """ for i in range(PLAYERCOUNT):
            self.players.append(Player()) """
            
        self.players.append(Player())
        self.players.append(AIPlayer())

        
        if isinstance(self.players[1], AIPlayer):
            self.players[1].load_q()
            pass

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

        else :
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

        # Calcul de la récompense selon la nouvelle règle :
        # L'IA gagne (reward = 1) si ni le joueur ni le dealer ne bustent et si son score est strictement supérieur aux deux.
        # En cas d'égalité, la récompense est nulle (reward = 0) sinon c'est une défaite (reward = -1)
        reward = -1

        ai_busted = ai_player.busted
        human_busted = self.players[0].busted
        dealer_busted = dealer_score > 21

        # Vérification prioritaire du score de l'IA
        if ai_score > 21:
            reward = -1
        else:
            ai_busted = ai_player.busted
            human_busted = self.players[0].busted
            dealer_busted = dealer_score > 21

            if not human_busted and not dealer_busted:
                # Ni le joueur ni le dealer n'ont busté
                if ai_score > human_score and ai_score > dealer_score:
                    reward = 1
                elif ai_score == human_score or ai_score == dealer_score:
                    reward = 0
                else:
                    reward = -1
            elif dealer_busted and not human_busted:
                # Seul le dealer est busté
                if ai_score > human_score:
                    reward = 1
                elif ai_score == human_score:
                    reward = 0
                else:
                    reward = -1
            elif human_busted and not dealer_busted:
                # Seul le joueur est busté
                if ai_score > dealer_score:
                    reward = 1
                elif ai_score == dealer_score:
                    reward = 0
                else:
                    reward = -1
            elif not ai_busted and dealer_busted and human_busted:
                reward = 1
            else:
                reward = -1


        if reward == -1:
            print("AI busted or lost, reward = -1")
        elif reward == 1:
            print("AI wins, reward = 1")
            print(f"AI score: {ai_score}, Human score: {human_score}, Dealer score: {dealer_score}")
        elif reward == 0:
            print("AI ties with someone, reward = 0")

        # Propager la récompense à toutes les actions prises par l'IA
        for idx in range(len(ai_player.currentEpisode)):
            state, action, current_reward = ai_player.currentEpisode[idx]
            
            # If this is the last action of the episode or there's no intermediate reward yet
            if idx == len(ai_player.currentEpisode) - 1 or current_reward is None:
                ai_player.currentEpisode[idx] = (state, action, reward)
            else:
                # Add final reward to existing intermediate reward
                ai_player.currentEpisode[idx] = (state, action, current_reward + reward * 0.5)

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



    def train_ai(self, episodes=10000):
        """Entraîne l'IA en simulant de nombreuses parties automatiquement"""
        print(f"Démarrage de l'entraînement sur {episodes} parties...")
        wins = losses = ties = 0
        
        # Sauvegarder les états actuels
        save_game_over = self.gameOver
        
        for i in range(episodes):
            # Réinitialiser le jeu
            self.reset_players()
            game_finished = False
            
            # Simuler la partie
            while not game_finished:
                # Tour du joueur humain (simulé avec une stratégie basique)
                if self.playerTurn == 0 and not self.players[0].finishedTurn:
                    # Stratégie simple: tirer si score < 17, sinon rester
                    if self.players[0].get_score() < 17:
                        self.players[0].draw(self.deck)
                    else:
                        self.players[0].stand()
                
                # Tour de l'IA
                elif self.playerTurn == 1 and not self.players[1].finishedTurn:
                    action = self.players[1].play(self.players[0].hand, self.dealer.hand)
                    if action == 1:  # Hit
                        self.players[1].draw(self.deck)
                    else:  # Stand
                        self.players[1].stand()
                
                # Vérifier si tous les joueurs ont terminé
                game_finished = self.update()
            
            # Tour du dealer
            while self.dealer.get_score() < 17:
                self.dealer.draw(self.deck)
            
            # Déterminer le gagnant et calculer les récompenses
            ai_player = self.players[1]
            ai_score = ai_player.get_score()
            human_score = self.players[0].get_score()
            dealer_score = self.dealer.get_score()
            
            
            reward_win = 1
            reward_tie = 0
            reward_loss = -1
            
            # Calculer la récompense
            reward = reward_loss  # Par défaut (défaite)
            

            # Vérification prioritaire du score de l'IA
            if ai_score > 21:
                reward = reward_loss
            else:
                ai_busted = ai_player.busted
                human_busted = self.players[0].busted
                dealer_busted = dealer_score > 21

                if not human_busted and not dealer_busted:
                    # Ni le joueur ni le dealer n'ont busté
                    if ai_score > human_score and ai_score > dealer_score:
                        reward = reward_win
                    elif ai_score == human_score or ai_score == dealer_score:
                        reward = reward_tie
                    else:
                        reward = reward_loss
                elif dealer_busted and not human_busted:
                    # Seul le dealer est busté
                    if ai_score > human_score:
                        reward = reward_win
                    elif ai_score == human_score:
                        reward = reward_tie
                    else:
                        reward = reward_loss
                elif human_busted and not dealer_busted:
                    # Seul le joueur est busté
                    if ai_score > dealer_score:
                        reward = reward_win
                    elif ai_score == dealer_score:
                        reward = reward_tie
                    else:
                        reward = reward_loss
                elif not ai_busted and dealer_busted and human_busted:
                    reward = reward_win
                else:
                    reward = reward_loss
                    
            if reward == reward_loss:
                losses += 1
                recent_results.append(0)
            elif reward == reward_win:
                wins += 1
                recent_results.append(1)
            elif reward == reward_tie:
                ties += 1
                recent_results.append(0)

            #Mettre à jour currentEpisode avec la récompense calculée
            for idx in range(len(ai_player.currentEpisode)):
                state, action, _ = ai_player.currentEpisode[idx]
                ai_player.currentEpisode[idx] = (state, action, reward)
            
            #Mise à jour de la table Q
            self.players[1].setQ()
            
            # Afficher la progression régulièrement
            if (i + 1) % 10000 == 0:
                win_rate = (wins / (i + 1)) * 100
                recent_win_rate = (sum(recent_results) / len(recent_results)) * 100 if recent_results else 0
                progress = ((i + 1) / episodes) * 100
                print(f"Progress: {progress:.2f}% - Partie {i + 1}/{episodes} - "
                        f"global wr: {win_rate:.2f}%, "
                        f"10k last: {recent_win_rate:.2f}%, "
                        f"win: {wins}, loss: {losses}, tie: {ties}, "
                        f"e: {self.players[1].e:.4f}, alpha: {self.players[1].alpha:.4f}, gamma: {self.players[1].gamma:.4f}")
                # Sauvegarde périodique
                if isinstance(self.players[1], AIPlayer):
                    self.players[1].save_q()
                    pass
                

            if isinstance(self.players[1], AIPlayer):
                if self.players[1].e != 1.0 and self.players[1].e != 0.0:
                    self.players[1].e = max(0.01, self.players[1].e * 0.9999999) 
        
        # Afficher les résultats finaux
        print(f"\nEntraînement terminé!")
        print(f"Victoires: {wins} ({(wins/episodes)*100:.2f}%)")
        print(f"Égalités: {ties} ({(ties/episodes)*100:.2f}%)")
        print(f"Défaites: {losses} ({(losses/episodes)*100:.2f}%)")
        
        # Sauvegarder la table Q finale
        if isinstance(self.players[1], AIPlayer):
            self.players[1].save_q()
        
        # Restaurer l'état du jeu
        self.gameOver = save_game_over
        self.reset_players()