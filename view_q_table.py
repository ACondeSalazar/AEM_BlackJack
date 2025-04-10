import pickle
import matplotlib.pyplot as plt
import numpy as np

# Charger le fichier pickle
with open('q_table.pkl', 'rb') as f:
    data = pickle.load(f)

# Extraire les états pertinents (par exemple, ceux où le score du joueur est entre 12 et 21)
filtered_states = {state: values for state, values in data.items() if isinstance(state, tuple) and 4 <= state[0] <= 21}  

# Organiser par score du joueur
player_scores = sorted(set(state[0] for state in filtered_states.keys()))

plt.figure(figsize=(12, 8))
for score in player_scores:
    states_with_score = {s: v for s, v in filtered_states.items() if s[0] == score}
    best_actions = ['Stand' if v[0] > v[1] else 'Hit' for v in states_with_score.values()]
    stand_count = best_actions.count('Stand')
    hit_count = best_actions.count('Hit')
    
    plt.bar(str(score), stand_count, label='Stand' if score == player_scores[0] else None, color='blue')
    plt.bar(str(score), hit_count, bottom=stand_count, label='Hit' if score == player_scores[0] else None, color='red')

plt.xlabel('Score du joueur')
plt.ylabel('Nombre d\'états')
plt.title('Répartition des meilleures actions par score du joueur')
plt.legend()
plt.tight_layout()
plt.savefig('q_table_visualization.png')
plt.show()