# AI Digits Classification Lab 🧠🎨

Ce projet est un tableau de bord interactif d'apprentissage automatique (Machine Learning) développé avec **Python**, **Scikit-learn** et **Streamlit**. Il permet d'explorer, de comprendre et de comparer visuellement les performances de plusieurs algorithmes de classification sur la célèbre base de données **MNIST** (70 000 images de chiffres manuscrits).

---

## ✨ Fonctionnalités Principales

### 🔬 Laboratoire d'Algorithmes (Visualisation 2D)
Découvrez comment les mathématiques séparent les chiffres dans un espace vectoriel. L'application compresse les images avec une **ACP (PCA)** et trace la **frontière de décision** interactive pour comparer :
- **Analyse Discriminante** : LDA (Linéaire) vs QDA (Quadratique)
- **Régression Logistique** (avec régularisation)
- **Random Forest** (Forêt Aléatoire, remplaçant l'arbre de décision simple pour contrer le sur-apprentissage)
- **KNN** (K-Plus Proches Voisins, pondéré par la distance)

### 🌌 Réduction de Dimensionnalité Avancée
- Projection et clustering non-linéaire avec **t-SNE** et **UMAP**.
- Visualisez comment ces algorithmes puissants regroupent les pixels similaires pour former des "galaxies" de chiffres sans aucune connaissance préalable.

### ✏️ Bac à Sable Interactif (Interactive Sandbox)
Testez les modèles en temps réel avec vos propres données !
1. **Dessinez un chiffre** directement sur l'écran.
2. **Téléversez une image** (photo papier de votre écriture).
3. Les algorithmes prédisent simultanément votre chiffre.
4. **Détection d'Anomalie :** Un modèle **One-Class SVM** évalue la "qualité" de votre tracé pour détecter si vous avez bien dessiné un chiffre (Chiffre Valide) ou si le dessin est aberrant (Anomalie).

---

## ⚙️ Traitement d'Image & Mathématiques
L'application ne se contente pas de redimensionner vos dessins. Elle réplique le **pipeline mathématique exact de MNIST** pour garantir la plus haute précision :
- **Center of Mass** : L'encre est parfaitement recentrée dans la matrice 28x28 selon son centre de gravité.
- **Anti-Aliasing (Filtre Gaussien)** : L'application applique un flou mathématique (`sigma=0.8`) pour recréer la distribution de l'intensité des pixels d'un vrai stylo, évitant ainsi que des pixels 100% bruts ne perturbent les modèles.

---

## 🚀 Installation & Exécution

Assurez-vous d'avoir Python installé (idéalement 3.10+), puis exécutez ces commandes :

```bash
# 1. Cloner ou télécharger le projet, puis aller dans le dossier
cd MNIST

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Lancer l'application locale
streamlit run app.py
```

### 📦 Dépendances Principales
- `streamlit` & `streamlit-drawable-canvas` (Interface Web)
- `scikit-learn` (Machine Learning)
- `numpy<2.5` & `scipy` (Calculs matriciels et traitement d'image)
- `umap-learn` & `matplotlib` (Réduction de dimension et graphiques)

---
*Projet réalisé dans le cadre de l'étude approfondie du pipeline de Machine Learning : Nettoyage de données, Sélection/Extraction de variables (Features), et Évaluation (Métriques/Généralisation).*