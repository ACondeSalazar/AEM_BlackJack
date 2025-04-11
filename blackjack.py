import random
import os 
import pickle
import numpy as np
import math  # Ajout de l'import math nécessaire pour le MCTS
from constants import *
from collections import defaultdict, namedtuple
import matplotlib.pyplot as plt
import networkx as nx


def visualize_complete_mcts(root_state, max_depth=2):
    """
    Visualise un arbre MCTS complet avec toutes les possibilités de tirage.
    
    Args:
        root_state: L'état racine (score_ia, score_joueur, carte_visible_dealer)
        max_depth: Profondeur maximale à visualiser
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    import math
    
    # Créer un graphe dirigé
    G = nx.DiGraph()
    
    # Dictionnaires pour stocker les labels et couleurs
    node_labels = {}
    node_colors = []
    edge_labels = {}
    
    # Valeurs possibles des cartes
    card_values = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11]  # 2-10, J, Q, K, A
    card_names = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
    
    # Construire l'arbre complet de manière récursive
    def build_complete_tree(state, node_id, action=None, depth=0):
        ai_score, player_score, dealer_visible = state
        
        # Ajouter le nœud au graphe
        G.add_node(node_id)
        
        # Format du label: score_ia
        node_labels[node_id] = f"{ai_score}"
        
        # Couleur basée sur l'action
        if action == 0:  # Stand
            node_colors.append('lightcoral')
        elif action == 1:  # Hit
            node_colors.append('lightgreen')
        else:  # Racine
            node_colors.append('lightblue')
        
        # Si on atteint la profondeur maximale ou si l'IA bust, on s'arrête
        if depth >= max_depth or ai_score > 21:
            return
        
        # Action Stand (toujours possible)
        stand_id = f"{node_id}_stand"
        G.add_node(stand_id)
        G.add_edge(node_id, stand_id)
        edge_labels[(node_id, stand_id)] = "Stand"
        build_complete_tree(state, stand_id, action=0, depth=depth+1)
        
        # Action Hit avec toutes les cartes possibles
        for i, (value, name) in enumerate(zip(card_values, card_names)):
            new_score = ai_score + value
            # Si l'as nous fait dépasser 21, il vaut 1
            if name == 'A' and new_score > 21:
                new_score = ai_score + 1
                
            hit_id = f"{node_id}_hit_{name}"
            new_state = (new_score, player_score, dealer_visible)
            
            G.add_node(hit_id)
            G.add_edge(node_id, hit_id)
            edge_labels[(node_id, hit_id)] = f"Hit ({name})"
            
            build_complete_tree(new_state, hit_id, action=1, depth=depth+1)
    
    # Commencer avec l'état racine
    build_complete_tree(root_state, "root")
    
    # Calculer la mise en page hiérarchique
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
    except:
        pos = nx.spring_layout(G, seed=42)
    
    # Créer la figure
    plt.figure(figsize=(20, 12))
    
    # Dessiner le graphe
    nx.draw(G, pos, labels=node_labels, node_color=node_colors, 
            node_size=1000, font_size=8, arrows=True)
    
    # Dessiner les labels des arêtes
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)
    
    # Ajouter une légende
    plt.legend(handles=[
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', markersize=10, label='Racine'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', markersize=10, label='Hit'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral', markersize=10, label='Stand')
    ])
    
    plt.title("Arbre complet de possibilités pour BlackJack")
    plt.tight_layout()
    plt.savefig('mcts_complete_tree.png', dpi=300)
    plt.show()

""" 
def visualize_mcts(root_node, max_depth=3):

    Visualise l'arbre MCTS à partir du nœud racine.
    
    Args:
        root_node: Le nœud racine du MCTS
        max_depth: Profondeur maximale à visualiser (pour éviter les arbres trop grands)

    # Créer un graphe dirigé
    G = nx.DiGraph()
    
    # Dictionnaire pour stocker les labels des nœuds
    node_labels = {}
    node_colors = []
    
    # Construire le graphe de manière récursive avec limite de profondeur
    def build_graph(node, node_id, depth=0):
        if depth > max_depth:
            return
            
        # Format pour le label: score_ia | visites | récompense moyenne
        ai_score = node.state[0]
        node_labels[node_id] = f"{ai_score}\n{node.visits}\n{node.reward/node.visits if node.visits else 0:.2f}"
        
        # Couleur basée sur l'action (rouge pour stand, vert pour hit)
        if node.action == 0:
            node_colors.append('lightcoral')  # Stand
        elif node.action == 1:
            node_colors.append('lightgreen')  # Hit
        else:
            node_colors.append('lightblue')  # Racine
        
        # Ajouter le nœud au graphe
        G.add_node(node_id)
        
        # Pour chaque enfant, ajouter au graphe et créer une arête
        for action, child in node.children.items():
            child_id = f"{node_id}_{action}"
            G.add_node(child_id)
            G.add_edge(node_id, child_id, label="Hit" if action == 1 else "Stand")
            build_graph(child, child_id, depth + 1)
    
    # Commencer la construction du graphe depuis la racine
    build_graph(root_node, "root")
    
    # Position des nœuds (utiliser un layout hiérarchique)
    pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
    
    # Créer la figure
    plt.figure(figsize=(12, 8))
    
    # Dessiner le graphe
    nx.draw(G, pos, labels=node_labels, node_color=node_colors, 
            node_size=2000, font_size=8, arrows=True)
    
    # Dessiner les labels des arêtes
    edge_labels = {(u, v): d['label'] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    
    # Ajouter une légende
    plt.legend(handles=[
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', markersize=10, label='Racine'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', markersize=10, label='Hit'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral', markersize=10, label='Stand')
    ])
    
    plt.title("Arbre MCTS pour la prise de décision BlackJack")
    plt.tight_layout()
    plt.savefig('mcts_tree.png')
    plt.show()

 """

def visualize_mcts(root_node, max_depth=3):
    """
    Visualise l'arbre MCTS à partir du nœud racine.
    
    Args:
        root_node: Le nœud racine du MCTS
        max_depth: Profondeur maximale à visualiser (pour éviter les arbres trop grands)
    """
    # Créer un graphe dirigé
    G = nx.DiGraph()
    
    # Dictionnaires pour stocker les labels, couleurs et types de nœuds
    node_labels = {}
    node_types = {}  # Stocke le type de chaque nœud (0=stand, 1=hit, None=racine)
    edge_labels = {}
    
    # Construire le graphe de manière récursive avec limite de profondeur
    def build_graph(node, node_id, depth=0):
        if depth > max_depth:
            return
            
        # Format pour le label: score_ia | visites | récompense moyenne
        ai_score = node.state[0]
        node_labels[node_id] = f"{ai_score}\n{node.visits}\n{node.reward/node.visits if node.visits else 0:.2f}"
        
        # Stocker le type de nœud
        node_types[node_id] = node.action
        
        # Ajouter le nœud au graphe
        G.add_node(node_id)
        
        # Pour chaque enfant, ajouter au graphe et créer une arête
        for action, children in node.children.items():
            if isinstance(children, dict):  # Cas pour "hit" avec plusieurs cartes
                for sub_id, child in children.items():
                    child_id = f"{node_id}_{sub_id}"
                    G.add_node(child_id)
                    node_types[child_id] = child.action  # Stocker le type de l'enfant
                    label = f"Hit ({child.state[0] - node.state[0]})"
                    G.add_edge(node_id, child_id, label=label)
                    edge_labels[(node_id, child_id)] = label
                    build_graph(child, child_id, depth + 1)
            else:  # Cas pour "stand" (un seul enfant)
                child_id = f"{node_id}_{action}"
                G.add_node(child_id)
                node_types[child_id] = children.action  # Stocker le type de l'enfant
                label = "Stand" if action == 0 else "Hit" 
                G.add_edge(node_id, child_id, label=label)
                edge_labels[(node_id, child_id)] = label
                build_graph(children, child_id, depth + 1)
    
    # Commencer la construction du graphe depuis la racine
    build_graph(root_node, "root")
    
    # Maintenant que tous les nœuds sont ajoutés, créer un tableau de couleurs correspondant
    node_colors = []
    for node in G.nodes():
        if node_types.get(node) == 0:
            node_colors.append('lightcoral')  # Stand
        elif node_types.get(node) == 1:
            node_colors.append('lightgreen')  # Hit
        else:
            node_colors.append('lightblue')  # Racine
    
    # Position des nœuds (utiliser un layout hiérarchique)
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
    except:
        pos = nx.spring_layout(G, seed=42)
    
    # Créer la figure
    plt.figure(figsize=(16, 10))
    
    # Dessiner le graphe
    nx.draw(G, pos, labels=node_labels, node_color=node_colors, 
            node_size=2000, font_size=8, arrows=True)
    
    # Dessiner les labels des arêtes
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    
    # Ajouter une légende
    plt.legend(handles=[
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', markersize=10, label='Racine'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', markersize=10, label='Hit'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral', markersize=10, label='Stand')
    ])
    
    plt.title("Arbre MCTS pour la prise de décision BlackJack")
    plt.tight_layout()
    plt.savefig('mcts_tree.png')
    plt.show()

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
    def branch_value(child, total_visits, exploration=1.41):
        if child.visits == 0:
            return float('inf')
            
        # Réduire l'exploration pour les scores élevés
        ai_score = child.state[0]
        if ai_score >= 17:
            # Réduire le coefficient d'exploration pour les scores élevés
            local_exploration = exploration * max(0.5, (21 - ai_score) / 8)
        else:
            local_exploration = exploration
            
        return (child.reward / child.visits) + local_exploration * math.sqrt(math.log(total_visits) / child.visits)

# Fonctions MCTS déplacées au niveau du module
""" 
def select(node):

    Sélectionne récursivement un noeud à développer
    en choisissant à chaque étape le noeud enfant avec la plus grande valeur UCT.

    current = node
    while not current.is_terminal():
        if not current.fully_expanded():
            return current
        else:
            total_visits = sum(child.visits for child in current.children.values())
            # Choix du fils avec la plus grande UCT
            current = max(current.children.values(), key=lambda n: MCTSNode.branch_value(n, total_visits))
    return current 
    """

def select(node):
    current = node
    while not current.is_terminal():
        if not current.fully_expanded():
            return current
        else:
            total_visits = sum(child.visits for action_type in current.children.values() 
                              for child in (action_type.values() if isinstance(action_type, dict) else [action_type]))
            
            # Trouver l'enfant avec la plus grande valeur UCT
            best_child = None
            best_value = float('-inf')
            
            for action, children in current.children.items():
                if isinstance(children, dict):  # Pour les actions "hit" avec plusieurs résultats
                    for child in children.values():
                        uct_val = MCTSNode.branch_value(child, total_visits)
                        if uct_val > best_value:
                            best_value = uct_val
                            best_child = child
                else:  # Pour les actions simples comme "stand"
                    uct_val = MCTSNode.branch_value(children, total_visits)
                    if uct_val > best_value:
                        best_value = uct_val
                        best_child = children
            
            current = best_child
    return current

def simulate_transition(state, action):
    ai_score, player_score, dealer_visible = state
    if action == 1: 
        card = draw_random_card()
        new_score = simulate_draw(ai_score, card)
        return (new_score, player_score, dealer_visible)
    elif action == 0: 
        return state
""" 
def expand(node):
    possible_actions = [0, 1] 
    for action in possible_actions:
        if action not in node.children:
            new_state = simulate_transition(node.state, action)
            child = MCTSNode(new_state, parent=node, action=action)
            node.children[action] = child
            return child
        
    return random.choice(list(node.children.values())) """

def expand(node):
    possible_actions = [0, 1]
    for action in possible_actions:
        if action not in node.children:
            if action == 0:  # Stand - simple, juste un nœud
                new_state = node.state  # L'état reste le même
                child = MCTSNode(new_state, parent=node, action=0)
                node.children[0] = child
                return child
            
            elif action == 1:  # Hit - créer un nœud pour chaque carte possible
                ai_score, player_score, dealer_visible = node.state
                
                # Créer un dictionnaire pour stocker les nœuds enfants
                hit_children = {}
                
                # Distribution des cartes dans un jeu standard:
                # 4 cartes de chaque valeur 2-9
                # 16 cartes valant 10 (10, J, Q, K × 4 couleurs)
                # 4 As
                
                # Créer des nœuds pour chaque valeur numérique 2-9
                for card_value in range(2, 10):
                    new_score = ai_score + card_value
                    new_state = (new_score, player_score, dealer_visible)
                    card_node = MCTSNode(new_state, parent=node, action=1)
                    # Stocker le nœud avec une clé qui inclut la probabilité (4/52)
                    hit_children[f"1_{card_value}"] = {
                        "node": card_node,
                        "prob": 4/52  # 4 cartes de chaque valeur sur 52
                    }
                
                # Cartes valant 10 (10, J, Q, K)
                ten_score = ai_score + 10
                ten_state = (ten_score, player_score, dealer_visible)
                ten_node = MCTSNode(ten_state, parent=node, action=1)
                hit_children[f"1_10"] = {
                    "node": ten_node,
                    "prob": 16/52  # 16 cartes valant 10 sur 52
                }
                
                # As (vaut 11 si possible, sinon 1)
                ace_value = 11 if ai_score <= 10 else 1
                ace_score = ai_score + ace_value
                ace_state = (ace_score, player_score, dealer_visible)
                ace_node = MCTSNode(ace_state, parent=node, action=1)
                hit_children[f"1_A"] = {
                    "node": ace_node,
                    "prob": 4/52  # 4 As sur 52
                }
                
                # Stocker tous ces nœuds dans un dictionnaire
                node_dict = {}
                for key, data in hit_children.items():
                    node_dict[key] = data["node"]
                
                node.children[1] = node_dict
                
                # Sélectionner un nœud aléatoirement mais pondéré par probabilité
                weights = [data["prob"] for data in hit_children.values()]
                keys = list(hit_children.keys())
                chosen_key = random.choices(keys, weights=weights, k=1)[0]
                
                # Retourner le nœud choisi
                return hit_children[chosen_key]["node"]
    
    # Si les deux actions sont déjà explorées, choisir un enfant "hit" pondéré par probabilité
    if isinstance(node.children.get(1), dict):
        # Distribution des probabilités selon les cartes
        probs = []
        nodes = []
        for key, child in node.children[1].items():
            if key.startswith("1_10"):
                probs.append(16/52)  # 10, J, Q, K (16 cartes)
            elif key.startswith("1_A"):
                probs.append(4/52)   # As (4 cartes)
            else:
                probs.append(4/52)   # Autres valeurs (4 cartes chacune)
            nodes.append(child)
        
        return random.choices(nodes, weights=probs, k=1)[0]
    
    # Fallback
    return random.choice(list(node.children.values()))
""" 
def rollout(state, action):
    current_state = state
    ai_score, player_score, dealer_visible = current_state

    if action == 1:
        while ai_score < 17:
            card = draw_random_card()
            ai_score = simulate_draw(ai_score, card)
            current_state = (ai_score, player_score, dealer_visible)
            if ai_score > 21:
                break

    if ai_score > 21:
        return -1

    dealer_score = simulate_dealer(dealer_visible)

    return check_score_state(ai_score, player_score, dealer_score)

 """
def rollout(state, action):
    current_state = state
    ai_score, player_score, dealer_visible = current_state

    # Valeur entre 0 et 1 qui représente la probabilité de bust
    bust_risk = max(0, min(1, (ai_score - 11) / 10))
    
    if action == 1:  # Si l'action est Hit
        # Pour les scores élevés, calculer le risque de bust
        if ai_score >= 17:
            # Probabilité accrue de bust = résultat négatif plus fréquent
            if random.random() < bust_risk:
                return -1  # Simuler un bust
        
        # Stratégie de base améliorée
        while ai_score < 17:
            card = draw_random_card()
            ai_score = simulate_draw(ai_score, card)
            current_state = (ai_score, player_score, dealer_visible)
            if ai_score > 21:
                return -1  # Bust immédiat

    if ai_score > 21:
        return -1

    # Bonus pour des scores proches de 21 sans bust
    score_bonus = 0
    if 17 <= ai_score <= 21:
        score_bonus = (ai_score - 17) / 8  # Entre 0 et 0.5 pour scores 17-21
    
    dealer_score = simulate_dealer(dealer_visible)
    
    result = check_score_state(ai_score, player_score, dealer_score)
    
    # Amplifier légèrement les résultats positifs pour les scores élevés sans bust
    if result > 0 and score_bonus > 0:
        result += score_bonus * 0.2
        
    return result


def check_score_state(ai_score, human_score, dealer_score):

        ai_busted = False
        if human_score > 21:
            human_busted = True
        else:
            human_busted = False
        if dealer_score > 21:
            dealer_busted = True
        else:
            dealer_busted = False


        # Vérification prioritaire du score de l'IA
        if ai_score > 21:
            reward = -1
        else:

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

        return reward


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

        """     
        def play(self, otherPlayerHand, dealerHand):
    
            print(f"Cartes du joueur 0: {self.hand}")
            print(f"Score: {self.hand.score()}")
            print(f"Cartes du joueur 1: {otherPlayerHand}")
            print(f"Score: {otherPlayerHand.score()}")
            print(f"Cartes du Dealer: {dealerHand}")
            print(f"Score: {dealerHand.score()}") 
        currentState = self.createStateValues(otherPlayerHand, dealerHand)
        print(f"Current state: {currentState}")
        action = self.genAction(currentState, self.e, self.Q)
        print(f"Action: {action}")
        self.currentEpisode.append((currentState, action, 0))

        return action 
        """

    def mcts(self, root_state, iterations=1000, visualize=False):
        root_node = MCTSNode(root_state)
        for _ in range(iterations):
            node = select(root_node)
            if not node.is_terminal():
                node = expand(node)
            reward = rollout(node.state, node.action)
            backpropagate(node, reward)
            
        if visualize:
            try:
                visualize_mcts(root_node)
            except ImportError:
                print("Bibliothèques de visualisation non disponibles.")
        
        if not root_node.children:
            return 0  
        
        # Décision finale - tenir compte du score et du risque
        ai_score = root_state[0]
        
        # Règles spécifiques pour les scores élevés
        if ai_score >= 17:
            # Calculer le risque de bust
            bust_risk = (ai_score - 16) / 5  # 0.2 pour 17, 0.4 pour 18, etc.
            
            # Si risque élevé et un des enfants n'a pas bust, préférer stand
            if bust_risk > 0.6 and 0 in root_node.children:
                return 0
        
        # Pour les autres cas, évaluer les visites pondérées par récompenses moyennes
        best_action = None
        best_value = float('-inf')
        
        for action, child_or_dict in root_node.children.items():
            if isinstance(child_or_dict, dict):  # Si c'est un dictionnaire (action "hit")
                # Calculer la valeur moyenne pour tous les enfants de cette action
                total_visits = sum(child.visits for child in child_or_dict.values())
                total_reward = sum(child.reward for child in child_or_dict.values())
                
                # Score pondéré supplémentaire: réduire la valeur pour les scores risqués
                action_value = (total_reward / total_visits) if total_visits > 0 else 0
                
                if action_value > best_value:
                    best_value = action_value
                    best_action = action
            else:  # Si c'est un nœud unique (action "stand")
                action_value = (child_or_dict.reward / child_or_dict.visits) if child_or_dict.visits > 0 else 0
                if action_value > best_value:
                    best_value = action_value
                    best_action = action
        
        return best_action

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
        
        print(f"Dealer visible card: {dealer_visible}") 

        # État pour MCTS: (score_ia, score_joueur, carte_visible_dealer)
        state = (ai_score, player_score, dealer_visible)
        print(f"MCTS state: {state}")
        
        #visualize_complete_mcts(state, max_depth=2)

        # Lancer MCTS pour déterminer l'action
        action = self.mcts(state, iterations=100000, visualize=True)
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
            use_mcts = False  # Mettre à True pour utiliser MCTS, False pour Q-learning
            
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
                max_score = max(human_score, dealer_score)
                if ai_score > human_score and ai_score > dealer_score:
                    reward = 1
                elif ai_score == max_score:
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
                    max_score = max(human_score, dealer_score)
                    if ai_score > max_score:
                        reward = 1
                    elif ai_score == max_score:
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