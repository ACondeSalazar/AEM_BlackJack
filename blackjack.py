import random
import os 
import pickle
import numpy as np
import math  # Ajout de l'import math nécessaire pour le MCTS
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
    
class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        """
        state : tuple (score_ia, score_joueur, dealer_visible)
        parent : noeud parent
        action : action appliquée pour atteindre ce nœud (0 = stand, 1 = hit)
        """
        self.state = state  
        self.parent = parent
        self.action = action  
        self.children = {}  # dictionnaire action -> enfant
        self.visits = 0
        self.reward = 0.0

    def is_terminal(self):
        """
        Considère l'état comme terminal dans deux cas :
          - L'IA est bust (score > 21)
          - L'action stand (0) a été prise (on considère alors que l'IA a terminé son tour)
        """
        ai_score, _, _ = self.state
        # Si l'IA est bust, on est terminal.
        if ai_score > 21:
            return True
        # Si c'est un noeud qui provient d'une décision stand,
        # on considère alors que l'IA a terminé son tour.
        if self.action == 0:
            return True
        return False

    def fully_expanded(self):
        """Ici, l'expansion est complète si les deux actions (hit et stand) ont été explorées."""
        return len(self.children) == 2

    @staticmethod
    def uct_value(child, total_visits, exploration=1.41):
        """Calcule la valeur UCT (Upper Confidence Bound for Trees) pour un enfant."""
        if child.visits == 0:
            return float('inf')
        return (child.reward / child.visits) + exploration * math.sqrt(math.log(total_visits) / child.visits)

# Fonctions MCTS déplacées au niveau du module
def select(node):
    """
    Sélectionne récursivement un noeud à développer
    en choisissant à chaque étape le noeud enfant avec la plus grande valeur UCT.
    """
    current = node
    while not current.is_terminal():
        if not current.fully_expanded():
            return current
        else:
            total_visits = sum(child.visits for child in current.children.values())
            # Choix du fils avec la plus grande UCT
            current = max(current.children.values(), key=lambda n: MCTSNode.uct_value(n, total_visits))
    return current

def simulate_transition(state, action):
    """
    Retourne le nouvel état à partir d'un état et d'une action.
    - Si l'action est "hit" (1) : tire une carte et met à jour le score de l'IA.
    - Si l'action est "stand" (0) : l'état reste inchangé.
    
    L'état est un tuple : (score_ia, score_joueur, dealer_visible)
    """
    ai_score, player_score, dealer_visible = state
    if action == 1:  # hit
        card = draw_random_card()
        new_score = simulate_draw(ai_score, card)
        return (new_score, player_score, dealer_visible)
    elif action == 0:  # stand
        return state

def expand(node):
    """
    Développe le noeud en ajoutant un enfant pour une action non encore explorée.
    """
    possible_actions = [0, 1]  # 0: stand, 1: hit
    for action in possible_actions:
        if action not in node.children:
            new_state = simulate_transition(node.state, action)
            child = MCTSNode(new_state, parent=node, action=action)
            node.children[action] = child
            return child
    # Si les deux actions sont déjà explorées, retourne aléatoirement un enfant
    return random.choice(list(node.children.values()))

def rollout(state):
    """
    Simule (rollout) la partie à partir de l'état donné en suivant une politique simple :
    - Tant que le score de l'IA est inférieur à 17, l'IA continue de piocher.
    - Sinon, l'IA reste.
    À la fin, on simule le tour du dealer et on détermine la récompense :
    - L'IA est considérée gagnante (reward = 1) si son score est supérieur au score du joueur et du dealer.
    - Si l'IA dépasse 21, c'est une défaite (reward = -1).
    - Un égalité donne reward = 0.
    """
    current_state = state
    ai_score, player_score, dealer_visible = current_state
    # Continuer à piocher tant que le score est inférieur à 17 et que l'IA n'est pas bust
    while ai_score < 17:
        # On applique la politique "piocher" (hit)
        card = draw_random_card()
        ai_score = simulate_draw(ai_score, card)
        current_state = (ai_score, player_score, dealer_visible)
        # Si bust, on sort immédiatement
        if ai_score > 21:
            break
    # Si l'IA s'arrête (stand) ou est bust, on simule le dealer.
    if ai_score > 21:
        return -1
    dealer_score = simulate_dealer(dealer_visible)
    # Simple règle d'évaluation :
    # L'IA gagne si son score est supérieur à celui du joueur et du dealer.
    if ai_score > player_score and ai_score > dealer_score:
        return 1
    elif ai_score == player_score or ai_score == dealer_score:
        return 0
    else:
        return -1

def simulate_dealer(dealer_visible_card):
    """Simule le jeu du dealer selon les règles du blackjack"""
    # Si dealer_visible_card est un entier, c'est déjà le score
    if isinstance(dealer_visible_card, int):
        dealer_score = dealer_visible_card
    # Sinon, calculer le score en fonction de la carte
    else:
        dealer_score = card_value(dealer_visible_card)
    
    # Tirer des cartes jusqu'à atteindre au moins 17
    hidden = draw_random_card()
    dealer_score = simulate_draw(dealer_score, hidden)
    
    # Le dealer continue à tirer tant que son score est inférieur à 17.
    while dealer_score < 17:
        dealer_score = simulate_draw(dealer_score, draw_random_card())
        # Si on bust, on retourne directement
        if dealer_score > 21:
            break
    
    return dealer_score

def backpropagate(node, reward):
    """
    Remonte dans l'arbre en mettant à jour les statistiques (visites et récompense cumulée) de chaque noeud.
    """
    while node is not None:
        node.visits += 1
        node.reward += reward
        node = node.parent

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
    
    def __init__(self, e=0.00, gamma=0.8, alpha=0.1):
        super().__init__()
        self.name = "AI"
        self.e = e
        self.gamma = gamma
        self.alpha = alpha
        self.Q = defaultdict(lambda: np.zeros(2))  
        self.currentEpisode = [] 

        # Pré-initialiser les états critiques
        for score in range(11, 17):  # Scores faibles où hit est généralement meilleur
            for dealer_card in range(7, 11):  # Cartes fortes du dealer
                for player_card in range(1, 30):
                    self.Q[(score, player_card, dealer_card)][1] = 0.3  # Légère préférence pour hit

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
        self.currentEpisode.append((currentState, action, 0))

        return action

    def mcts(self, root_state, iterations=1000):
        """
        Effectue l'exploration MCTS à partir de l'état racine pendant un nombre donné d'itérations.
        Retourne l'action (0 = stand, 1 = hit) à jouer à partir du noeud racine.
        """
        root_node = MCTSNode(root_state)
        for _ in range(iterations):
            # Sélection : choisir un noeud à développer
            node = select(root_node)
            # Expansion : si le noeud n'est pas terminal, développer un enfant
            if not node.is_terminal():
                node = expand(node)
            # Simulation (rollout)
            reward = rollout(node.state)
            # Backpropagation : mettre à jour les statistiques le long du chemin
            backpropagate(node, reward)
            
        # Choix de l'action optimale : on prend l'action de l'enfant le plus visité
        if not root_node.children:
            return 0  # par défaut, si aucun enfant, on reste (stand)
        
        best_child = max(root_node.children.values(), key=lambda n: n.visits)
        return best_child.action

    def mcts_decision(self, player_score, dealer_score):
        """
        Calcule l'état pour le MCTS et lance MCTS pour déterminer l'action.
        Retourne 1 (piocher) ou 0 (stand).
        """
        # Format d'état compatible avec le reste du système
        ai_score = self.hand.score()
        
        # Si c'est un score inférieur à 11, toujours piocher
        if ai_score < 11:
            return 1
        
        # Si c'est un entier, c'est juste un score
        if isinstance(dealer_score, int):
            dealer_visible = dealer_score
        else:
            # Si c'est une main, on prend la première carte
            dealer_visible = dealer_score.cards[0] if dealer_score.cards else ('0', '')
        
        # État pour MCTS: (score_ia, score_joueur, carte_visible_dealer)
        state = (ai_score, player_score, dealer_visible)
        print(f"MCTS state: {state}")
        
        # Lancer MCTS pour déterminer l'action
        action = self.mcts(state, iterations=500)
        return action

    def play(self, deck, state):
        """
        Prend une décision basée sur MCTS ou Q-Learning
        """
        if isinstance(state, list) and len(state) >= 2:
            otherPlayerHand = state[0]
            dealerHand = state[1]
            
            currentState = self.createStateValues(otherPlayerHand, dealerHand)
            print(f"Current state: {currentState}")
            
            # Utiliser MCTS ou Q-learning pour prendre une décision
            use_mcts = True  # Mettre à True pour utiliser MCTS, False pour Q-learning
            
            if use_mcts:
                action = self.mcts_decision(otherPlayerHand.score(), dealerHand)
            else:
                action = self.genAction(currentState, self.e, self.Q)
                
            print(f"Action: {action}")
            self.currentEpisode.append((currentState, action, 0))
            
            # Exécuter l'action
            if action == 1:  # Hit
                self.draw(deck)
            else:  # Stand
                self.stand()
                
            return action
        else:
            print("Format d'état invalide")
            return 0

    # créer un etat en fonction du score de la main de l'ia de l'autre joueur et du dealer 
    def createStateValues(self, otherPlayerHand, dealerHand):
        return (self.hand.score(), otherPlayerHand.score(), dealerHand.score())

    def genAction(self, state, e, Q):
        if self.hand.score() < 11:
            return 1
        else :
            probHit = Q[state][1]
            probStick = Q[state][0]
            print("proba de hit :", probHit)
            print("proba de stand :", probStick)
            
            if probHit>probStick:
                probs = [e, 1-e] 
            elif probStick>probHit:
                probs = [1-e, e]  
            else:
                probs = [0.5, 0.5]
                
            action = np.random.choice(np.arange(2), p=probs)   
            return action


    def setQ(self):
        for t in range(len(self.currentEpisode)):
            # Récupérer les récompenses de t à la fin de l'épisode
            rewards = np.array([step[2] for step in self.currentEpisode[t:]])
            # Créer la liste des taux de discount pour chaque pas de temps, en commençant par gamma**0 = 1 pour la récompense immédiate
            discountRates = np.array([self.gamma ** i for i in range(len(rewards))])
            # Calculer le retour Gt
            Gt = np.sum(rewards * discountRates)
            # Mise à jour de la Q-value pour l'état et l'action à l'instant t
            state, action, _ = self.currentEpisode[t]
            self.Q[state][action] += self.alpha * (Gt - self.Q[state][action])
        return self.Q


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
                print(f"Taille de la table Q : {len(self.Q)}") 
                print(f"Table Q chargée depuis {filename}")
                return True
            else:
                print("Aucun fichier de table Q trouvé, utilisation d'une nouvelle table")
                return False
        except Exception as e:
            print(f"Erreur lors du chargement de la table Q: {e}")
            return False 

def draw_random_card():
    """Simule le tirage d'une carte aléatoire et renvoie sa valeur"""
    card_rank = random.choice(RANKS)
    card_suit = random.choice(SUITS)
    return (card_rank, card_suit)

def simulate_draw(current_score, card):
    """Calcule le nouveau score après avoir tiré une carte"""
    card_value = card[0]  # Prend le rang de la carte
    
    if card_value in 'JQK':
        value = 10
    elif card_value == 'A':
        # Logique pour l'as: vaut 11 si ne fait pas dépasser 21, sinon 1
        if current_score <= 10:
            value = 11
        else:
            value = 1
    else:
        value = int(card_value)
    
    return current_score + value

def simulate_dealer(dealer_visible_card):
    """Simule le jeu du dealer selon les règles du blackjack"""
    # Si dealer_visible_card est un entier, c'est déjà le score
    if isinstance(dealer_visible_card, int):
        dealer_score = dealer_visible_card
    # Sinon, calculer le score en fonction de la carte
    else:
        dealer_score = card_value(dealer_visible_card)
    
    # Tirer des cartes jusqu'à atteindre au moins 17
    hidden = draw_random_card()
    dealer_score = simulate_draw(dealer_score, hidden)
    
    # Le dealer continue à tirer tant que son score est inférieur à 17.
    while dealer_score < 17:
        dealer_score = simulate_draw(dealer_score, draw_random_card())
        # Si on bust, on retourne directement
        if dealer_score > 21:
            break
    
    return dealer_score

def card_value(card):
    """Retourne la valeur d'une carte"""
    # Si c'est un entier, c'est déjà la valeur
    if isinstance(card, int):
        return card
    
    # Sinon, c'est une carte (tuple)
    value = card[0]  # Premier élément du tuple (rank, suit)
    if value in 'JQK':
        return 10
    elif value == 'A':
        return 11  # Pour la simulation du dealer, on prend 11 par défaut
    else:
        return int(value)

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
        
        # Pour le joueur humain
        self.players[0].play(self.deck, [])
        
        # Créer l'état pour l'IA
        ai_state = [self.players[0].hand, self.dealer.hand]
        
        # Pour le joueur IA
        self.players[1].play(self.deck, ai_state)
        
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



    def train_ai(self, episodes=100000):
        """Entraîne l'IA en simulant de nombreuses parties automatiquement"""
        print(f"Démarrage de l'entraînement sur {episodes} parties...")
        wins = losses = ties = 0
        
        # Sauvegarder les états actuels
        save_game_over = self.gameOver
        
        # Au début de train_ai
        initial_epsilon = 0.3
        min_epsilon = 0.01

        for i in range(episodes):
            # Réinitialiser le jeu
            self.reset_players()
            game_finished = False
            self.players[1].e = max(min_epsilon, initial_epsilon * (1 - i/episodes))
            
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
            
            # Calculer la récompense
            reward = -1  # Par défaut (défaite)

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
                losses += 1
            elif reward == 1:
                wins += 1
            elif reward == 0:
                ties += 1

            
            # Propager la récompense à toutes les actions prises par l'IA
            for i in range(len(ai_player.currentEpisode)):
                state, action, _ = ai_player.currentEpisode[i]
                ai_player.currentEpisode[i] = (state, action, reward)

            # Mise à jour de la table Q
            self.players[1].setQ()
            
            # Afficher la progression régulièrement
            if (i + 1) % 1000 == 0:
                win_rate = (wins / (i + 1)) * 100
                print(f"Partie {i + 1}/{episodes} - Taux de victoire: {win_rate:.2f}%")
                # Sauvegarde périodique
                if isinstance(self.players[1], AIPlayer):
                    self.players[1].save_q()
        
        # Afficher les résultats finaux
        print(f"\nEntraînement terminé!")
        print(f"Victoires: {wins} ({(wins/episodes)*100:.2f}%)")
        print(f"Égalités: {ties} ({(ties/episodes)*100:.2f}%)")
        print(f"Défaites: {losses} ({(losses/episodes)*100:.2f}%)")
        print(f"Taille de la table Q : {len(self.players[1].Q)}") 
        
        # Sauvegarder la table Q finale
        if isinstance(self.players[1], AIPlayer):
            self.players[1].save_q()
        
        # Restaurer l'état du jeu
        self.gameOver = save_game_over
        self.reset_players()