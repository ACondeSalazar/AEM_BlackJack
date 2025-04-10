import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm  # For colormap

# Load the pickle file
with open('q_table.pkl', 'rb') as f:
    data = pickle.load(f)

# Fix filtering based on correct state structure
filtered_states = {state: values for state, values in data.items() if isinstance(state, tuple) and state[0] > 4 and state[2] > 2}

print("Filtered states count:", len(filtered_states))  # Debugging
print(filtered_states)

# Extract player scores and dealer face-up cards
player_scores = sorted(set(state[0] for state in filtered_states.keys()))
dealer_cards = sorted(set(state[2] for state in filtered_states.keys()))  # Use state[2] for dealer

print("Unique Player Scores:", player_scores)
print("Unique Dealer Cards:", dealer_cards)

# Initialize action grid (Z-axis)
Z = np.full((len(player_scores), len(dealer_cards)), np.nan)  # Fill with NaN initially

# Fill grid with best actions
for i, p_score in enumerate(player_scores):
    for j, d_card in enumerate(dealer_cards):
        # Find matching state in filtered_states (ignoring state[1])
        matching_state = next((state for state in filtered_states.keys() if state[0] == p_score and state[2] == d_card), None)
        
        if matching_state:
            values = filtered_states[matching_state]
            best_action = 0 if values[0] > values[1] else 1  # 0 = Stand, 1 = Hit
            Z[i, j] = best_action


# Debugging Z
print("Z matrix shape:", Z.shape)
print("Sample Z values:", Z[:5, :5])

# Convert lists to meshgrid format
X, Y = np.meshgrid(dealer_cards, player_scores)  # X = Dealer, Y = Player

# Create 3D figure
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot surface with color mapping
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, edgecolor='k', alpha=0.8)

# Labels
ax.set_xlabel('Dealer Face-up Card')
ax.set_ylabel('Player Score')
ax.set_zlabel('Best Action (0 = Stand, 1 = Hit)')
ax.set_title('Best Action Surface Plot')

# Colorbar
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label="Action (0 = Stand, 1 = Hit)")

plt.show()
