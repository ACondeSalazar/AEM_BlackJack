import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# === Paramètres ===
pickle_file = 'q_table.pkl'
has_ace_filter = True  # True = avec un As, False = sans As

# === Charger le modèle ===
with open(pickle_file, 'rb') as f:
    q_table = pickle.load(f)

    # === Collecter les données ===
    data_confidence = {}
    data_choice = {}

    for state, action_values in q_table.items():
        ia_score = state[0]
        dealer_score = state[2]
        ia_has_ace = state[3]

        if ia_has_ace != has_ace_filter:
            continue

        stand_val = action_values[0]
        hit_val = action_values[1]
        confidence = confidence = abs(hit_val - stand_val) / (abs(hit_val) + abs(stand_val))


        preferred_action = 1 if hit_val > stand_val else 0
        key = (ia_score, dealer_score)

        data_confidence[key] = confidence
        data_choice[key] = preferred_action

    # === Grilles ===
    ia_scores = sorted(set(k[0] for k in data_confidence.keys()))
    dealer_scores = sorted(set(k[1] for k in data_confidence.keys()))

    X, Y = np.meshgrid(ia_scores, dealer_scores)
    Z = np.zeros_like(X, dtype=float)
    C = np.zeros_like(X, dtype=float)

    for i in range(Y.shape[0]):
        for j in range(X.shape[1]):
            key = (X[0][j], Y[i][0])
            Z[i, j] = data_confidence.get(key, np.nan)
            C[i, j] = data_choice.get(key, np.nan)

    # === Normaliser la confiance (Z) pour l’intensité ===
    Z_max = np.nanmax(Z)
    if Z_max > 0:  # Prevent division by zero
        Z_norm = Z / Z_max
    else:
        Z_norm = Z.copy()  # If all values are 0, don't normalize

    # Make sure all values are within 0-1 range
    Z_norm = np.clip(Z_norm, 0, 1)

    # === Générer des couleurs RGB personnalisées ===
    colors = np.zeros((X.shape[0], X.shape[1], 4))  # RGBA

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if np.isnan(Z[i, j]):
                colors[i, j] = [0, 0, 0, 0]  # Transparent
            else:
                confidence = (Z_norm[i, j] + 1) / 2  # Normalize confidence to range [0, 1]
                if np.isnan(C[i, j]):  # Handle NaN values explicitly
                    colors[i, j] = [0, 0, 0, 0]  # Transparent
                elif C[i, j] == 1:  # Hit → light orange
                    r = 1.0
                    g = min(1.0, max(0.0, 0.6 + (1 - confidence) * 0.4))
                    b = min(1.0, max(0.0, 0.4 + (1 - confidence) * 0.6))
                    colors[i, j] = [r, g, b, 1]  # Alpha=1
                else:  # Stand → teal
                    r = min(1.0, max(0.0, 0.4 + (1 - confidence) * 0.6))
                    g = min(1.0, max(0.0, 0.8 + (1 - confidence) * 0.2))
                    b = min(1.0, max(0.0, 0.8 + (1 - confidence) * 0.2))
                    colors[i, j] = [r, g, b, 1]  # Alpha=1

    # === Tracer la surface ===
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    mask = ~np.isnan(Z)
    X_masked = np.ma.array(X, mask=~mask)
    Y_masked = np.ma.array(Y, mask=~mask)
    Z_masked = np.ma.array(Z, mask=~mask)
    colors_masked = np.ma.array(colors, mask=np.repeat(~mask[:, :, np.newaxis], 4, axis=2))

    surf = ax.plot_surface(X_masked, Y_masked, Z_masked, facecolors=colors_masked, edgecolor='k', linewidth=0.1, antialiased=True, shade=True, rstride = 1, cstride = 1)

    # Légende manuelle
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', edgecolor='k', label='Hit (plus rouge = plus confiant)'),
        Patch(facecolor='blue', edgecolor='k', label='Stand (plus bleu = plus confiant)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    ax.set_xlabel('Score main IA')
    ax.set_ylabel('Carte visible Dealer')
    ax.set_zlabel('Confiance IA (|hit - stand|)')

    title_prefix = "avec un As" if has_ace_filter else "sans As"
    plt.title(f"Monte Carlo - Couleur = Choix IA, Intensité = Confiance ({title_prefix})")

    plt.tight_layout()
    plt.show()
