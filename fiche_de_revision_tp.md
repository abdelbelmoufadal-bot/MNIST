# Fiche de Révision - Soutenance de Machine Learning (Projet MNIST)

Ce document est conçu pour vous préparer aux questions de votre professeur. Il résume chaque TP, explique le workflow global et vous donne la réponse exacte à la question piège sur la généralisation.

---

## 💡 La question piège du prof : "Est-ce que le modèle est généralisé ?"

**Ce que le prof veut entendre :** Il veut savoir si votre modèle fait du "par-cœur" (Overfitting / Sur-apprentissage) ou s'il a vraiment compris la logique mathématique.

**Comment répondre :**
> *"Oui, le modèle généralise bien car nous avons respecté la séparation stricte des données (Train/Test Split). Nous l'avons entraîné sur un sous-ensemble (Train) et nous l'avons évalué sur des données (Test) qu'il n'avait jamais vues auparavant. Les métriques (précision, rappel) sont élevées sur le jeu de Test, ce qui prouve qu'il ne fait pas de sur-apprentissage et qu'il est capable de classifier de nouveaux chiffres manuscrits."*

*(Bonus : Vous pouvez ajouter que la régularisation, comme le paramètre C dans la Régression Logistique ou la limitation de profondeur `max_depth` dans les Arbres, aide spécifiquement à la généralisation).*

---

## 🛠️ Le Workflow Machine Learning (Les Étapes Clés)

Voici comment expliquer la logique de construction de votre projet à votre professeur :

### 3. Data Cleaning & Outlier Detection
**Objectif :** Nettoyer les données pour ne garder que les signaux pertinents.
**Dans ce projet :**
- L'image doit être recadrée et recentrée (Center of Mass) pour être alignée avec les données d'entraînement.
- **Outliers (Valeurs aberrantes) :** Si un utilisateur dessine une maison au lieu d'un chiffre, cela perturbe la prédiction. Nous utilisons **One-Class SVM** (voir TP correspondant) pour détecter mathématiquement si l'image est un chiffre valide ou un *outlier* (anomalie) en calculant sa distance par rapport à la distribution normale des chiffres MNIST.

### 4. Feature Selection (Sélection des Variables)
**Objectif :** Choisir les variables (pixels) les plus importantes et écarter celles qui sont redondantes ou inutiles.
**Concept :** On utilise souvent des matrices de corrélation. Si deux variables apportent exactement la même information (corrélation de 1.0), on peut en supprimer une. Dans le cas d'images, de nombreux pixels sur les bords sont toujours noirs (variance = 0), on n'a donc pas besoin de les donner aux algorithmes.

### 5. Feature Extraction (Extraction de Variables - ex: ACP/PCA)
**Objectif :** Créer de *nouvelles* variables plus puissantes en condensant l'information.
**Concept (TP PCA) :** L'Analyse en Composantes Principales (ACP) réduit la dimension de l'image (ex: de 784 pixels à 50 composantes). Contrairement à la *sélection*, l'ACP combine les pixels pour trouver les axes (composantes) qui contiennent le plus de "variance" (d'information).
**Pourquoi l'utiliser ?** Pour accélérer les calculs, retirer le bruit de fond, et permettre l'utilisation d'algorithmes capricieux comme LDA/QDA qui plantent face à des pixels colinéaires.

### 7. Les Métriques (Évaluation)
**Objectif :** Quantifier mathématiquement la performance du modèle.
- **Matrice de confusion :** Montre exactement où le modèle se trompe (ex: combien de "8" ont été confondus avec des "3").
- **Accuracy (Précision globale) :** Le pourcentage de prédictions justes. Attention, trompeur si les données sont déséquilibrées !
- **Precision / Recall / F1-Score :** Indispensables pour évaluer la robustesse classe par classe. Le F1-score est la moyenne harmonique de la précision et du rappel.

---

## 📚 Résumé des TP (Les Algorithmes)

Voici l'antisèche pour chaque fichier Jupyter (`.ipynb`) étudié en TP.

### 1. PCA (Analyse en Composantes Principales)
- **Concept :** Transformation mathématique linéaire qui projette les données dans un espace de dimension inférieure en maximisant la variance.
- **Objectif :** Compression de données et réduction de bruit.
- **Quand l'utiliser ?** Avant d'entraîner des modèles lourds, ou pour pouvoir visualiser des données complexes sur un graphique en 2D ou 3D.

### 2. Logistic Regression (Régression Logistique)
- **Concept :** Modèle linéaire probabiliste. Il trace un hyperplan (une ligne) pour séparer les classes, en utilisant la fonction Sigmoïde pour sortir des probabilités (entre 0 et 1).
- **Objectif :** Classification binaire ou multi-classes (One-vs-Rest).
- **Quand l'utiliser ?** C'est le modèle de référence de base ("Baseline"). Très rapide, très interprétable, parfait quand les classes sont linéairement séparables.

### 3. KNN (K-Nearest Neighbors / K-Plus Proches Voisins)
- **Concept :** Algorithme paresseux (Lazy learning). Il ne calcule pas d'équation. Pour prédire, il cherche les $K$ images d'entraînement les plus proches géométriquement et prend la classe majoritaire.
- **Objectif :** Classification basée sur la similarité.
- **Quand l'utiliser ?** Sur des données non-linéaires complexes. Idéal quand on a beaucoup de données d'entraînement. *Désavantage* : lent lors de la prédiction et prend beaucoup de mémoire.

### 4. Decision Tree (Arbre de Décision)
- **Concept :** Suite de règles "Si / Sinon" basées sur les pixels. Il découpe l'espace des données de manière orthogonale jusqu'à avoir des feuilles pures.
- **Objectif :** Créer un modèle extrêmement explicable (on peut dessiner l'arbre).
- **Quand l'utiliser ?** Quand l'explicabilité est primordiale. *Attention* : Il est très sujet à l'overfitting (il fait du par-cœur). Il est souvent remplacé par le **Random Forest** (Forêt Aléatoire) pour corriger ce défaut.

### 5. LDA & QDA (Analyse Discriminante Linéaire et Quadratique)
- **Concept :** Modèles génératifs. Ils modélisent la distribution (moyenne et covariance) des pixels pour chaque chiffre en utilisant le théorème de Bayes.
  - **LDA :** Suppose que toutes les classes ont la même forme (covariance identique) $\rightarrow$ Frontière droite.
  - **QDA :** Permet à chaque classe d'avoir sa propre forme $\rightarrow$ Frontière courbée.
- **Objectif :** Trouver les axes qui maximisent la séparation entre les classes tout en minimisant la dispersion au sein d'une même classe.
- **Quand l'utiliser ?** Très puissant après une PCA. Le LDA est excellent quand l'espace des données est petit.

### 6. One-Class SVM (Machine à Vecteurs de Support)
- **Concept :** Variante du SVM classique qui n'apprend qu'à partir d'une seule classe (ici, "les vrais chiffres"). Il enveloppe cette classe dans une sphère de densité complexe en utilisant l'astuce du noyau (Kernel Trick - RBF).
- **Objectif :** Détection d'anomalies (Outlier detection) ou de nouveautés.
- **Quand l'utiliser ?** Quand on veut vérifier qu'une nouvelle donnée appartient bien au domaine étudié avant de la classer (par exemple : rejeter un dessin de visage dans un classifieur de chiffres).
