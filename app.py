import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from streamlit_drawable_canvas import st_canvas
import scipy.ndimage

# --- Configuration de la page ---
st.set_page_config(page_title="AI Digits Classification Lab", layout="wide")

# --- Injection de styles CSS personnalisés (Aesthetics Upgrade) ---
st.markdown(
    """
<style>
/* Import Google Font Outfit & JetBrains Mono */
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&family=JetBrains+Mono&display=swap');

/* Variables de design & styles globaux */
.stApp {
background: radial-gradient(circle at 50% 50%, #0F172A 0%, #090D16 100%) !important;
color: #F1F5F9 !important;
font-family: 'Outfit', sans-serif !important;
}

/* Barre latérale (Sidebar) */
section[data-testid="stSidebar"] {
background: linear-gradient(180deg, #090D16 0%, #05070B 100%) !important;
border-right: 1px solid rgba(255, 255, 255, 0.06);
}

/* En-tête de navigation */
section[data-testid="stSidebar"] .stRadio > label {
color: #00F2FE !important;
font-weight: 800;
font-size: 14px;
text-transform: uppercase;
letter-spacing: 0.15em;
margin-bottom: 18px;
padding-left: 8px;
border-left: 3px solid #9B51E0;
}

/* Conteneurs d'options de menu individuelles (Cartes de Navigation) */
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label {
background: rgba(255, 255, 255, 0.02) !important;
border: 1px solid rgba(255, 255, 255, 0.04) !important;
border-radius: 10px !important;
padding: 10px 14px !important;
margin-bottom: 8px !important;
transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
cursor: pointer !important;
}

/* Couleur du texte par défaut (Amélioration de contraste radicale) */
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label p {
color: #94A3B8 !important;
font-size: 14px !important;
font-weight: 500 !important;
transition: color 0.25s ease !important;
}

/* Effet au survol (Hover) */
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label:hover {
background: rgba(0, 242, 254, 0.06) !important;
border-color: rgba(0, 242, 254, 0.3) !important;
transform: translateX(3px);
}

section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label:hover p {
color: #F1F5F9 !important;
}

/* Style de l'option sélectionnée (Active State) */
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label:has(input[type="radio"]:checked) {
background: linear-gradient(90deg, rgba(0, 242, 254, 0.12) 0%, rgba(79, 172, 254, 0.12) 100%) !important;
border-color: rgba(0, 242, 254, 0.6) !important;
box-shadow: 0 0 15px rgba(0, 242, 254, 0.15) !important;
}

section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label:has(input[type="radio"]:checked) p {
color: #00F2FE !important;
font-weight: 700 !important;
}

/* Ajustement de l'alignement des éléments internes */
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label > div:first-child {
display: flex;
align-items: center;
}

/* Cartes Glassmorphism */
div.glass-card {
background: rgba(30, 41, 59, 0.4);
border: 1px solid rgba(255, 255, 255, 0.06);
border-radius: 16px;
padding: 24px;
margin-bottom: 20px;
backdrop-filter: blur(16px);
-webkit-backdrop-filter: blur(16px);
box-shadow: 0 10px 40px 0 rgba(0, 0, 0, 0.3);
transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

div.glass-card:hover {
transform: translateY(-2px);
box-shadow: 0 12px 40px 0 rgba(0, 242, 254, 0.15);
border-color: rgba(0, 242, 254, 0.3);
}

/* Titres avec dégradé moderne */
.gradient-text {
background: linear-gradient(90deg, #00F2FE 0%, #4FACFE 50%, #9B51E0 100%);
-webkit-background-clip: text;
-webkit-text-fill-color: transparent;
font-weight: 800;
font-size: 42px;
margin-bottom: 10px;
}

.gradient-subtext {
color: #94A3B8;
font-size: 16px;
font-weight: 400;
margin-bottom: 30px;
}

/* Sections Headers */
h2, h3 {
color: #F8FAFC !important;
font-weight: 700 !important;
}

/* Blocs de code & rapports textuels */
pre, code {
font-family: 'JetBrains Mono', monospace !important;
background-color: #04060B !important;
color: #38BDF8 !important;
border: 1px solid rgba(56, 189, 248, 0.15) !important;
border-radius: 8px !important;
font-size: 13px !important;
}

/* Boîtes de formules mathématiques */
div.math-box {
background: rgba(15, 23, 42, 0.5);
border-left: 4px solid #9B51E0;
border-radius: 6px;
padding: 16px;
margin: 16px 0;
box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}

/* Métriques personnalisées */
.metric-title {
color: #94A3B8;
font-size: 12px;
font-weight: 700;
text-transform: uppercase;
letter-spacing: 0.1em;
margin-bottom: 4px;
}

.metric-value {
font-size: 32px;
font-weight: 800;
background: linear-gradient(90deg, #00F2FE 0%, #4FACFE 100%);
-webkit-background-clip: text;
-webkit-text-fill-color: transparent;
}

.metric-value.pink {
background: linear-gradient(90deg, #FF007B 0%, #FF7B00 100%);
-webkit-background-clip: text;
-webkit-text-fill-color: transparent;
}

.metric-value.purple {
background: linear-gradient(90deg, #9B51E0 0%, #E040FB 100%);
-webkit-background-clip: text;
-webkit-text-fill-color: transparent;
}

/* Tableaux comparatifs */
table.styled-table {
width: 100%;
border-collapse: collapse;
margin: 20px 0;
font-size: 15px;
border-radius: 8px;
overflow: hidden;
background: rgba(30, 41, 59, 0.2);
}

table.styled-table th {
background-color: rgba(30, 41, 59, 0.8);
color: #00F2FE;
text-align: left;
font-weight: 700;
padding: 14px 16px;
border-bottom: 2px solid rgba(255, 255, 255, 0.05);
}

table.styled-table td {
padding: 14px 16px;
border-bottom: 1px solid rgba(255, 255, 255, 0.05);
color: #E2E8F0;
}

table.styled-table tr:hover {
background-color: rgba(255, 255, 255, 0.03);
}

/* --- Correctifs de lisibilité : widgets natifs Streamlit --- */
/* Sans thème sombre configuré, ces widgets gardent un arrière-plan clair
   par défaut alors que le texte est forcé en clair par .stApp : on fixe
   explicitement fond ET couleur de texte pour éviter tout texte illisible. */

/* Boîtes d'alerte : st.info / st.warning / st.error / st.success */
div[data-testid="stAlert"], .stAlert {
background: rgba(0, 242, 254, 0.06) !important;
border: 1px solid rgba(0, 242, 254, 0.25) !important;
border-radius: 10px !important;
}
div[data-testid="stAlert"] p,
div[data-testid="stAlert"] span,
div[data-testid="stAlert"] div,
.stAlert p, .stAlert span {
color: #E2E8F0 !important;
}

/* Expanders : "Aperçu Mathématique" */
div[data-testid="stExpander"] {
background: rgba(30, 41, 59, 0.35) !important;
border: 1px solid rgba(255, 255, 255, 0.08) !important;
border-radius: 10px !important;
}
div[data-testid="stExpander"] summary,
div[data-testid="stExpander"] summary p,
div[data-testid="stExpander"] p,
div[data-testid="stExpander"] span,
div[data-testid="stExpander"] label {
color: #F1F5F9 !important;
}

/* Libellés de widgets (slider, selectbox, radio, file_uploader...) */
div[data-testid="stWidgetLabel"] p,
div[data-testid="stSlider"] label,
div[data-testid="stSlider"] div,
label[data-testid="stWidgetLabel"] {
color: #E2E8F0 !important;
}

/* Boutons */
.stButton button, button[kind] {
background: rgba(0, 242, 254, 0.08) !important;
border: 1px solid rgba(0, 242, 254, 0.4) !important;
color: #00F2FE !important;
}
.stButton button:hover, button[kind]:hover {
background: rgba(0, 242, 254, 0.18) !important;
color: #FFFFFF !important;
}

/* Zone de dépôt de fichier (file_uploader) */
div[data-testid="stFileUploaderDropzone"] {
background: rgba(30, 41, 59, 0.35) !important;
border: 1px dashed rgba(0, 242, 254, 0.3) !important;
}
div[data-testid="stFileUploaderDropzone"] * {
color: #CBD5E1 !important;
}
</style>
""",
    unsafe_allow_html=True
)

# --- Configuration du thème Matplotlib (Dark Neon Theme) ---
def configure_plot_theme():
    plt.style.use('dark_background')
    plt.rcParams.update({
        'figure.facecolor': 'none',
        'axes.facecolor': 'none',
        'axes.edgecolor': '#334155',
        'grid.color': '#1E293B',
        'text.color': '#E2E8F0',
        'axes.labelcolor': '#94A3B8',
        'xtick.color': '#94A3B8',
        'ytick.color': '#94A3B8',
        'font.family': 'sans-serif'
    })

# --- Helper: Prétraitement MNIST (Center of Mass & Bounding Box) ---
def preprocess_to_mnist_format(gray_image, thicken_factor=0):
    if thicken_factor > 0:
        gray_image = scipy.ndimage.maximum_filter(gray_image, size=thicken_factor)
        
    if gray_image.max() < 10:
        return np.zeros((28, 28), dtype=np.float32)
        
    # Bounding box
    rows = np.any(gray_image > 10, axis=1)
    cols = np.any(gray_image > 10, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    cropped = gray_image[rmin:rmax+1, cmin:cmax+1]

    # MNIST normalise la dimension maximale à 20 pixels
    h, w = cropped.shape
    scale = 20.0 / max(h, w)
    new_h, new_w = max(1, int(np.round(h * scale))), max(1, int(np.round(w * scale)))
    
    img_cropped = Image.fromarray(cropped)
    # L'utilisation de BILINEAR donne un rendu un peu plus net (plus proche de MNIST) que LANCZOS
    img_resized = img_cropped.resize((new_w, new_h), Image.BILINEAR)
    resized = np.array(img_resized)

    # Placer dans une image 28x28 vide
    final_img = np.zeros((28, 28), dtype=np.float32)
    start_y = (28 - new_h) // 2
    start_x = (28 - new_w) // 2
    final_img[start_y:start_y+new_h, start_x:start_x+new_w] = resized

    # Centrer selon le centre de gravité (Center of Mass)
    cy, cx = scipy.ndimage.center_of_mass(final_img)
    if not np.isnan(cy) and not np.isnan(cx):
        shift_y = 13.5 - cy
        shift_x = 13.5 - cx
        final_img = scipy.ndimage.shift(final_img, (shift_y, shift_x), order=1, cval=0)

    return np.clip(final_img, 0, 255)

# --- Chargement des données en Cache ---
@st.cache_resource
def load_data():
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X = mnist.data / 255.0
    y = mnist.target.astype(int)
    return X, y

with st.spinner('Chargement du dataset MNIST...'):
    X, y = load_data()

# --- Modèles & Transformations Caches ---
@st.cache_resource
def get_trained_pca(X, n_components):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    return pca, X_pca

@st.cache_resource
def get_binary_data(X, y, digit_a=0, digit_b=9):
    mask = (y == digit_a) | (y == digit_b)
    X_bin = X[mask]
    y_bin = y[mask]
    
    y_bin_mapped = np.where(y_bin == digit_a, 0, 1)
    
    pca_bin = PCA(n_components=2)
    X_bin_pca = pca_bin.fit_transform(X_bin)
    return X_bin, y_bin_mapped, pca_bin, X_bin_pca

@st.cache_resource
def train_model(model_name, X_train, y_train, k_neighbors=5):
    if model_name == "LDA":
        model = LinearDiscriminantAnalysis()
    elif model_name == "QDA":
        model = QuadraticDiscriminantAnalysis()
    elif model_name == "Logistic Regression":
        model = LogisticRegression(random_state=42)
    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier(random_state=42)
    elif model_name == "KNN":
        model = KNeighborsClassifier(n_neighbors=k_neighbors)
    elif model_name == "One-Class SVM":
        model = OneClassSVM(kernel="rbf", gamma="auto", nu=0.05)
    else:
        raise ValueError(f"Modèle inconnu: {model_name}")
    
    model.fit(X_train, y_train)
    return model

@st.cache_resource
def get_multiclass_models(X, y):
    # Séparation rapide sur les indices pour ÉVITER LE CRASH MÉMOIRE (OOM) sur Streamlit Cloud
    # 14,000 images sont largement suffisantes pour ces modèles simples
    indices = np.arange(len(y))
    train_idx, _ = train_test_split(indices, train_size=0.2, random_state=42, stratify=y)
    
    X_train = X[train_idx]
    y_train = y[train_idx]
    
    # 1. Modèles sur pixels bruts (plus robustes aux variations humaines)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    
    lr = LogisticRegression(max_iter=300, random_state=42)
    lr.fit(X_train, y_train)
    
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    
    # 2. Modèles mathématiques (requièrent une PCA car 784 variables colinéaires font planter QDA)
    pca = PCA(n_components=50)
    X_train_pca = pca.fit_transform(X_train)
    
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train_pca, y_train)
    
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(X_train_pca, y_train)
    
    return pca, {"KNN": knn, "Logistic Regression": lr, "Decision Tree": dt}, {"LDA": lda, "QDA": qda}

# --- Navigation latérale ---
menu = st.sidebar.radio(
    "Navigation", 
    [
        "🏠 Dashboard Home", 
        "1. PCA Compression & Noise", 
        "2. QDA / LDA", 
        "3. Logistic Regression",
        "4. Decision Tree", 
        "5. KNN", 
        "6. One-Class SVM",
        "✏️ Interactive Sandbox"
    ]
)

# --- Configuration des chiffres à classer ---
st.sidebar.markdown("---")
st.sidebar.subheader("🔧 Classification Config")
digit_a = st.sidebar.selectbox("Chiffre A (Classe 0)", list(range(10)), index=0)
digit_b = st.sidebar.selectbox("Chiffre B (Classe 1)", list(range(10)), index=9)

if digit_a == digit_b:
    st.sidebar.error("Erreur : Veuillez choisir deux chiffres différents.")
    st.stop()

# --- SECTION 0 : DASHBOARD HOME ---
if menu == "🏠 Dashboard Home":
    st.markdown('<h1 class="gradient-text">AI Digits Classification Lab</h1>', unsafe_allow_html=True)
    st.markdown('<p class="gradient-subtext">Tableau de bord interactif pour l\'exploration d\'algorithmes de Machine Learning appliqués à la reconnaissance d\'images (Dataset MNIST).</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(
            f"""
<div class="glass-card">
<h3>À propos du Lab</h3>
<p>Ce laboratoire vous permet d'explorer de manière interactive différentes méthodes mathématiques et modèles de classification. Le jeu de données de référence est <b>MNIST</b>, constitué de 70 000 images manuscrites de dimensions 28x28 (784 pixels).</p>
<p>Pour les besoins de visualisation 2D des frontières de décision, les algorithmes de classification binaire séparent les chiffres <b>{digit_a}</b> et <b>{digit_b}</b> en projetant les données sur les deux premières composantes principales (PCA).</p>
</div>
""", 
            unsafe_allow_html=True
        )
        
        # Comparaison des modèles en temps réel
        st.subheader(f"Matrice de Comparaison des Algorithmes (Classification {digit_a} vs {digit_b})")
        
        with st.spinner(f"Génération du comparatif des performances ({digit_a} vs {digit_b})..."):
            X_bin, y_bin, pca_bin, X_bin_pca = get_binary_data(X, y, digit_a, digit_b)
            X_train, X_test, y_train, y_test = train_test_split(X_bin_pca, y_bin, test_size=0.2, random_state=42, stratify=y_bin)
            
            # Entraînement et calcul des scores
            scores = {}
            for name in ["LDA", "QDA", "Logistic Regression", "Decision Tree", "KNN"]:
                model = train_model(name, X_train, y_train)
                y_pred = model.predict(X_test)
                scores[name] = accuracy_score(y_test, y_pred)
        
        table_html = f"""
<table class="styled-table">
<thead>
<tr>
<th>Algorithme</th>
<th>Type</th>
<th>Complexité mathématique</th>
<th>Précision (Accuracy)</th>
</tr>
</thead>
<tbody>
"""
        for m_name, acc in scores.items():
            table_html += f"""
<tr>
<td><b>{m_name}</b></td>
<td>Binaire ({digit_a} vs {digit_b})</td>
<td>{"Quadratique" if "QDA" in m_name else "Linéaire" if "LDA" in m_name or "Logistic" in m_name else "Non-paramétrique"}</td>
<td><span style="color: #00F2FE; font-weight: bold;">{acc*100:.3f}%</span></td>
</tr>
"""
        table_html += "</tbody></table>"
        st.markdown(table_html, unsafe_allow_html=True)
        
    with col2:
        st.markdown(
            f"""
<div class="glass-card" style="text-align: center;">
<p class="metric-title">Dataset MNIST</p>
<p class="metric-value">70 000</p>
<p style="color: #94A3B8; margin: 0; font-size: 14px;">Images labellisées</p>
</div>
<div class="glass-card" style="text-align: center;">
<p class="metric-title">Dimensions Originales</p>
<p class="metric-value pink">28 x 28</p>
<p style="color: #94A3B8; margin: 0; font-size: 14px;">784 pixels par image</p>
</div>
<div class="glass-card" style="text-align: center;">
<p class="metric-title">Meilleure Précision 2D</p>
<p class="metric-value purple">{max(scores.values())*100:.2f}%</p>
<p style="color: #94A3B8; margin: 0; font-size: 14px;">{max(scores, key=scores.get)}</p>
</div>
""",
            unsafe_allow_html=True
        )

# --- SECTION 1 : PCA ---
elif menu == "1. PCA Compression & Noise":
    st.markdown('<h1 class="gradient-text">Analyse en Composantes Principales (PCA)</h1>', unsafe_allow_html=True)
    st.markdown('<p class="gradient-subtext">Compressez et reconstruisez les images en projetant les pixels sur les vecteurs propres de variance maximale.</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Paramètres de la PCA")
        n_components = st.slider("Composantes principales (PCs)", 1, 784, 50)
        st.markdown('</div>', unsafe_allow_html=True)
        
        with st.spinner('Calcul de la PCA...'):
            pca, X_pca = get_trained_pca(X, n_components)
            
        st.markdown(
            f"""
<div class="glass-card">
<div style="margin-bottom: 16px;">
<p class="metric-title">Taux de Compression</p>
<p class="metric-value">{100 * n_components / X.shape[1]:.2f}%</p>
</div>
<div style="margin-bottom: 16px;">
<p class="metric-title">Dimension Réduite</p>
<p class="metric-value pink">{n_components} PCs</p>
</div>
<div>
<p class="metric-title">Dimension Originale</p>
<p class="metric-value purple">{X.shape[1]} Pixels</p>
</div>
</div>
""",
            unsafe_allow_html=True
        )
        
    with col2:
        configure_plot_theme()
        st.subheader("Visualisation de la Reconstruction d'Image")
        
        image = X[0]
        X_reconstructed = pca.inverse_transform(X_pca)
        reconstructed_image = X_reconstructed[0]
        
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(image.reshape(28, 28), cmap="gray")
        axes[0].set_title("Image Originale", fontsize=10, pad=10)
        axes[0].axis("off")
        
        axes[1].imshow(reconstructed_image.reshape(28, 28), cmap="gray")
        axes[1].set_title(f"Reconstruite ({n_components} PCs)", fontsize=10, pad=10)
        axes[1].axis("off")
        st.pyplot(fig)
        
        st.subheader("Réduction du Bruit Blanc avec PCA")
        original = X[0].reshape(28, 28)
        noise = np.random.normal(0, 0.30, (28, 28))
        noisy = np.clip(original + noise, 0, 1)
        noisy_vector = noisy.reshape(1, -1)
        
        compressed = pca.transform(noisy_vector)
        denoised = pca.inverse_transform(compressed).reshape(28, 28)
        
        fig_noise, axes_noise = plt.subplots(1, 3, figsize=(12, 4))
        axes_noise[0].imshow(original, cmap='gray')
        axes_noise[0].set_title("Originale", fontsize=10, pad=10)
        axes_noise[0].axis("off")
        axes_noise[1].imshow(noisy, cmap='gray')
        axes_noise[1].set_title("Avec Bruit", fontsize=10, pad=10)
        axes_noise[1].axis("off")
        axes_noise[2].imshow(denoised, cmap='gray')
        axes_noise[2].set_title("Après PCA (Dénoyée)", fontsize=10, pad=10)
        axes_noise[2].axis("off")
        st.pyplot(fig_noise)
        
        with st.expander("Aperçu Mathématique : Formule PCA"):
            st.markdown(
                """
<div class="math-box">
La PCA cherche à projeter les données $X$ dans un espace de dimension inférieure de façon à minimiser l'erreur de reconstruction :
"""
                , unsafe_allow_html=True
            )
            st.latex(r"\text{Erreur} = \|X - X_{\text{reconstructed}}\|_F^2")
            st.markdown(
                """
Les composantes principales sont obtenues via la décomposition en valeurs propres de la matrice de covariance.
</div>
""", unsafe_allow_html=True
            )

# --- SECTION 2 : QDA / LDA ---
elif menu == "2. QDA / LDA":
    st.markdown('<h1 class="gradient-text">Analyse Discriminante (QDA & LDA)</h1>', unsafe_allow_html=True)
    st.markdown(f'<p class="gradient-subtext">Classifiez les chiffres {digit_a} et {digit_b} en modélisant les distributions de probabilités des classes.</p>', unsafe_allow_html=True)
    
    X_bin, y_bin, pca_bin, X_bin_pca = get_binary_data(X, y, digit_a, digit_b)
    X_train, X_test, y_train, y_test = train_test_split(X_bin_pca, y_bin, test_size=0.2, random_state=42, stratify=y_bin)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Choix du Modèle")
        model_type = st.radio("Sélectionner l'algorithme", ["LDA", "QDA"])
        
        if model_type == "LDA":
            st.info("Fisher Criterion: LDA cherche à séparer les classes à l'aide d'une frontière de décision linéaire (hypothèse de matrices de covariance identiques).")
            model = train_model("LDA", X_train, y_train)
        else:
            st.info("QDA permet une frontière quadratique en estimant des matrices de covariance distinctes pour chaque classe.")
            model = train_model("QDA", X_train, y_train)
        st.markdown('</div>', unsafe_allow_html=True)
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        st.markdown(
            f"""
<div class="glass-card">
<p class="metric-title">Précision (Accuracy)</p>
<p class="metric-value">{acc*100:.3f}%</p>
</div>
""",
            unsafe_allow_html=True
        )
        
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Confusion Matrix")
        st.text(confusion_matrix(y_test, y_pred))
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        configure_plot_theme()
        st.subheader(f"Frontière de décision - {model_type}")
        
        h = 0.05
        x_min, x_max = X_bin_pca[:, 0].min() - 1, X_bin_pca[:, 0].max() + 1
        y_min, y_max = X_bin_pca[:, 1].min() - 1, X_bin_pca[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.contourf(xx, yy, Z, alpha=0.15, cmap="cool")
        
        # Plot class 0 (Teal)
        ax.scatter(X_bin_pca[y_bin == 0, 0], X_bin_pca[y_bin == 0, 1], s=8, label=f"Chiffre {digit_a}", color="#00F2FE", alpha=0.7)
        # Plot class 1 (Pink)
        ax.scatter(X_bin_pca[y_bin == 1, 0], X_bin_pca[y_bin == 1, 1], s=8, label=f"Chiffre {digit_b}", color="#FF007B", alpha=0.7)
        ax.set_xlabel("Première Composante Principale (PC1)", color="#94A3B8")
        ax.set_ylabel("Deuxième Composante Principale (PC2)", color="#94A3B8")
        ax.legend(facecolor="none", edgecolor="none")
        st.pyplot(fig)
        
        st.subheader("Rapport de Classification complet")
        st.text(classification_report(y_test, y_pred))
        
        with st.expander("Aperçu Mathématique : LDA vs QDA"):
            st.markdown(
                """
<div class="math-box">
La fonction discriminante quadratique de QDA s'écrit :
"""
                , unsafe_allow_html=True
            )
            st.latex(r"\delta_k(x) = -\frac{1}{2} \ln |\Sigma_k| - \frac{1}{2} (x - \mu_k)^T \Sigma_k^{-1} (x - \mu_k) + \ln \pi_k")
            st.markdown(
                r"""
Pour la LDA, on fait l'hypothèse que tous les groupes ont la même matrice de covariance ($\Sigma_k = \Sigma$), ce qui linéarise le terme quadratique.
</div>
""", unsafe_allow_html=True
            )

# --- SECTION 3 : LOGISTIC REGRESSION ---
elif menu == "3. Logistic Regression":
    st.markdown('<h1 class="gradient-text">Régression Logistique</h1>', unsafe_allow_html=True)
    st.markdown(f'<p class="gradient-subtext">Classifiez les chiffres {digit_a} et {digit_b} en utilisant une fonction sigmoïde pour estimer la probabilité d\'appartenance aux classes.</p>', unsafe_allow_html=True)
    
    X_bin, y_bin, pca_bin, X_bin_pca = get_binary_data(X, y, digit_a, digit_b)
    X_train, X_test, y_train, y_test = train_test_split(X_bin_pca, y_bin, test_size=0.2, random_state=42, stratify=y_bin)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Régression Logistique 2D")
        st.info(r"La régression logistique cherche la frontière linéaire optimale qui sépare la probabilité $P(Y=1|X) > 0.5$ de $P(Y=1|X) \leq 0.5$.")
        model = train_model("Logistic Regression", X_train, y_train)
        st.markdown('</div>', unsafe_allow_html=True)
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        st.markdown(
            f"""
<div class="glass-card">
<p class="metric-title">Précision (Accuracy)</p>
<p class="metric-value">{acc*100:.3f}%</p>
</div>
""",
            unsafe_allow_html=True
        )
        
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Confusion Matrix")
        st.text(confusion_matrix(y_test, y_pred))
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        configure_plot_theme()
        st.subheader("Frontière de décision - Régression Logistique")
        
        h = 0.05
        x_min, x_max = X_bin_pca[:, 0].min() - 1, X_bin_pca[:, 0].max() + 1
        y_min, y_max = X_bin_pca[:, 1].min() - 1, X_bin_pca[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.contourf(xx, yy, Z, alpha=0.15, cmap="cool")
        
        ax.scatter(X_bin_pca[y_bin == 0, 0], X_bin_pca[y_bin == 0, 1], s=8, label=f"Chiffre {digit_a}", color="#00F2FE", alpha=0.7)
        ax.scatter(X_bin_pca[y_bin == 1, 0], X_bin_pca[y_bin == 1, 1], s=8, label=f"Chiffre {digit_b}", color="#FF007B", alpha=0.7)
        ax.set_xlabel("PC1", color="#94A3B8")
        ax.set_ylabel("PC2", color="#94A3B8")
        ax.legend(facecolor="none", edgecolor="none")
        st.pyplot(fig)
        
        st.subheader("Rapport de Classification complet")
        st.text(classification_report(y_test, y_pred))
        
        with st.expander("Aperçu Mathématique : Régression Logistique"):
            st.markdown(
                """
<div class="math-box">
La probabilité d'appartenance à la classe 1 est calculée via la fonction sigmoïde :
"""
                , unsafe_allow_html=True
            )
            st.latex(r"P(Y=1|X) = \sigma(w^T X + b) = \frac{1}{1 + e^{-(w^T X + b)}}")
            st.markdown(
                """
Le modèle est optimisé en minimisant la fonction de coût d'entropie croisée binaire (Log Loss).
</div>
""", unsafe_allow_html=True
            )

# --- SECTION 4 : DECISION TREE ---
elif menu == "4. Decision Tree":
    st.markdown('<h1 class="gradient-text">Arbre de Décision (Decision Tree)</h1>', unsafe_allow_html=True)
    st.markdown('<p class="gradient-subtext">Classifiez les images en utilisant un arbre de décision non-paramétrique qui segmente l\'espace en régions rectangulaires.</p>', unsafe_allow_html=True)
    
    X_bin, y_bin, pca_bin, X_bin_pca = get_binary_data(X, y, digit_a, digit_b)
    X_train, X_test, y_train, y_test = train_test_split(X_bin_pca, y_bin, test_size=0.2, random_state=42, stratify=y_bin)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Paramètres de l'Arbre")
        st.info("Un arbre de décision coupe l'espace de manière orthogonale aux axes des variables d'entrée.")
        model = train_model("Decision Tree", X_train, y_train)
        st.markdown('</div>', unsafe_allow_html=True)
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        st.markdown(
            f"""
<div class="glass-card">
<p class="metric-title">Précision (Accuracy)</p>
<p class="metric-value">{acc*100:.3f}%</p>
</div>
""",
            unsafe_allow_html=True
        )
        
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Confusion Matrix")
        st.text(confusion_matrix(y_test, y_pred))
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        configure_plot_theme()
        st.subheader("Frontière de décision - Decision Tree")
        
        h = 0.05
        x_min, x_max = X_bin_pca[:, 0].min() - 1, X_bin_pca[:, 0].max() + 1
        y_min, y_max = X_bin_pca[:, 1].min() - 1, X_bin_pca[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.contourf(xx, yy, Z, alpha=0.15, cmap="cool")
        
        ax.scatter(X_bin_pca[y_bin == 0, 0], X_bin_pca[y_bin == 0, 1], s=8, label=f"Chiffre {digit_a}", color="#00F2FE", alpha=0.7)
        ax.scatter(X_bin_pca[y_bin == 1, 0], X_bin_pca[y_bin == 1, 1], s=8, label=f"Chiffre {digit_b}", color="#FF007B", alpha=0.7)
        ax.set_xlabel("PC1", color="#94A3B8")
        ax.set_ylabel("PC2", color="#94A3B8")
        ax.legend(facecolor="none", edgecolor="none")
        st.pyplot(fig)
        
        st.subheader("Rapport de Classification complet")
        st.text(classification_report(y_test, y_pred))
        
        with st.expander("Aperçu Mathématique : Impureté de Gini"):
            st.markdown(
                """
<div class="math-box">
Les coupures d'un arbre de décision sont choisies pour maximiser la baisse d'impureté de Gini dans les sous-nœuds. L'impureté de Gini $I_G(p)$ d'un nœud est définie par :
"""
                , unsafe_allow_html=True
            )
            st.latex(r"I_G(p) = 1 - \sum_{i=1}^{C} p_i^2")
            st.markdown(
                """
où $p_i$ est la fraction d'échantillons de classe $i$ présents dans le nœud.
</div>
""", unsafe_allow_html=True
            )

# --- SECTION 5 : KNN ---
elif menu == "5. KNN":
    st.markdown('<h1 class="gradient-text">K-Nearest Neighbors (KNN)</h1>', unsafe_allow_html=True)
    st.markdown('<p class="gradient-subtext">Classifiez les images en vous basant sur la classe majoritaire de ses plus proches voisins dans l\'espace de représentation.</p>', unsafe_allow_html=True)
    
    X_bin, y_bin, pca_bin, X_bin_pca = get_binary_data(X, y, digit_a, digit_b)
    X_train, X_test, y_train, y_test = train_test_split(X_bin_pca, y_bin, test_size=0.2, random_state=42, stratify=y_bin)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Paramètres de KNN")
        k = st.slider("Nombre de voisins (K)", 1, 15, 5)
        model = train_model("KNN", X_train, y_train, k_neighbors=k)
        st.markdown('</div>', unsafe_allow_html=True)
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        st.markdown(
            f"""
<div class="glass-card">
<p class="metric-title">Précision (Accuracy)</p>
<p class="metric-value">{acc*100:.3f}%</p>
</div>
""",
            unsafe_allow_html=True
        )
        
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Confusion Matrix")
        st.text(confusion_matrix(y_test, y_pred))
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        configure_plot_theme()
        st.subheader(f"Frontière de décision - KNN (K={k})")
        
        h = 0.1
        x_min, x_max = X_bin_pca[:, 0].min() - 1, X_bin_pca[:, 0].max() + 1
        y_min, y_max = X_bin_pca[:, 1].min() - 1, X_bin_pca[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.contourf(xx, yy, Z, alpha=0.15, cmap="cool")
        
        ax.scatter(X_bin_pca[y_bin == 0, 0], X_bin_pca[y_bin == 0, 1], s=8, label=f"Chiffre {digit_a}", color="#00F2FE", alpha=0.7)
        ax.scatter(X_bin_pca[y_bin == 1, 0], X_bin_pca[y_bin == 1, 1], s=8, label=f"Chiffre {digit_b}", color="#FF007B", alpha=0.7)
        ax.set_xlabel("PC1", color="#94A3B8")
        ax.set_ylabel("PC2", color="#94A3B8")
        ax.legend(facecolor="none", edgecolor="none")
        st.pyplot(fig)
        
        st.subheader("Rapport de Classification complet")
        st.text(classification_report(y_test, y_pred))
        
        with st.expander("Aperçu Mathématique : Distance Euclidienne"):
            st.markdown(
                """
<div class="math-box">
La distance euclidienne entre deux points de représentation $x$ et $y$ en dimension $D$ (ici $D=2$) s'écrit :
"""
                , unsafe_allow_html=True
            )
            st.latex(r"d(x, y) = \sqrt{\sum_{i=1}^{D} (x_i - y_i)^2}")
            st.markdown(
                """
Le modèle affecte à chaque point inconnu la classe majoritaire parmi ses $K$ voisins les plus proches.
</div>
""", unsafe_allow_html=True
            )

# --- SECTION 6 : ONE-CLASS SVM ---
elif menu == "6. One-Class SVM":
    st.markdown('<h1 class="gradient-text">Détection d\'Anomalies (One-Class SVM)</h1>', unsafe_allow_html=True)
    st.markdown('<p class="gradient-subtext">Modélisez les contours de densité d\'une seule classe (le chiffre 0) pour détecter les outliers.</p>', unsafe_allow_html=True)
    
    pca_full, X_pca_full = get_trained_pca(X, n_components=2)
    X_train_svm = X_pca_full[y == digit_a]
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Configuration du SVM")
        st.info(f"Le One-Class SVM est entraîné exclusivement sur les exemples du chiffre {digit_a}. Tout autre chiffre est censé être détecté comme une anomalie.")
        
        with st.spinner(f"Entraînement de One-Class SVM ({digit_a})..."):
            svm = train_model("One-Class SVM", X_train_svm, y[y == digit_a])
            pred = svm.predict(X_pca_full)
            
        unique, counts = np.unique(pred, return_counts=True)
        counts_dict = dict(zip(unique, counts))
        normal_cnt = counts_dict.get(1, 0)
        anomaly_cnt = counts_dict.get(-1, 0)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown(
            f"""
<div class="glass-card">
<div style="margin-bottom: 16px;">
<p class="metric-title">{digit_a} Détectés (Normaux)</p>
<p class="metric-value">{normal_cnt}</p>
</div>
<div>
<p class="metric-title">Anomalies Détectées</p>
<p class="metric-value pink">{anomaly_cnt}</p>
</div>
</div>
""",
            unsafe_allow_html=True
        )
        
    with col2:
        configure_plot_theme()
        st.subheader("Détection d'anomalies sur l'espace latent PCA")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        # Plot normal/recognized zeros (Teal)
        ax.scatter(X_pca_full[pred == 1, 0], X_pca_full[pred == 1, 1], s=8, label=f"Reconnu comme {digit_a} (Normal)", color="#00F2FE", alpha=0.5)
        # Plot anomaly/others (Red/Pink)
        ax.scatter(X_pca_full[pred == -1, 0], X_pca_full[pred == -1, 1], s=8, label=f"Anomalie (Pas un {digit_a})", color="#FF007B", alpha=0.3)
        ax.set_xlabel("PC1", color="#94A3B8")
        ax.set_ylabel("PC2", color="#94A3B8")
        ax.legend(facecolor="none", edgecolor="none")
        st.pyplot(fig)
        
        with st.expander("Aperçu Mathématique : Noyau RBF"):
            st.markdown(
                """
<div class="math-box">
Le One-Class SVM projette les données dans un espace de Hilbert de dimension infinie via une fonction noyau RBF (Radial Basis Function) :
"""
                , unsafe_allow_html=True
            )
            st.latex(r"K(x, y) = \exp(-\gamma \|x - y\|^2)")
            st.markdown(
                """
Il cherche ensuite l'hyperplan optimal séparant l'origine de la région de forte densité des points normaux.
</div>
""", unsafe_allow_html=True
            )
elif menu == "✏️ Interactive Sandbox":
    st.markdown('<h1 class="gradient-text">Bac à sable interactif</h1>', unsafe_allow_html=True)
    st.markdown('<p class="gradient-subtext">Dessinez, téléversez ou sélectionnez un chiffre pour le soumettre à l\'évaluation en temps réel de tous nos modèles.</p>', unsafe_allow_html=True)
    
    # 1. Sélection du canal d'entrée
    mode_selection = st.radio("Méthode de saisie", ["Dessiner un chiffre", "Téléverser une image", "Prendre un chiffre MNIST aléatoire"])
    
    pixels = None
    
    if mode_selection == "Dessiner un chiffre":
        st.write("Dessinez un chiffre (de 0 à 9) à l'aide de votre souris ou écran tactile dans le cadre noir ci-dessous.")

        col_canvas, col_preview = st.columns([1, 1])
        with col_canvas:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            canvas_result = st_canvas(
                fill_color="rgba(255, 255, 255, 1)",
                stroke_width=30,
                stroke_color="#FFFFFF",
                background_color="#000000",
                height=280,
                width=280,
                drawing_mode="freedraw",
                key="digit_canvas",
            )
            st.markdown('</div>', unsafe_allow_html=True)

        # ---- Prétraitement exact façon MNIST (Center of Mass) ----
        pixels = None
        if canvas_result.image_data is not None:
            rgba = canvas_result.image_data.astype(np.uint8)
            gray = np.dot(rgba[:, :, :3], [0.299, 0.587, 0.114]).astype(np.uint8)

            drawn_image_28 = preprocess_to_mnist_format(gray)
            if drawn_image_28.max() > 0:
                pixels = (drawn_image_28 / 255.0).flatten().tolist()

                with col_preview:
                    st.markdown('<div class="glass-card" style="text-align:center">', unsafe_allow_html=True)
                    st.markdown("**Aperçu 28×28 transmis aux modèles**")
                    configure_plot_theme()
                    fig_prev, ax_prev = plt.subplots(figsize=(2, 2))
                    ax_prev.imshow(drawn_image_28, cmap='gray', vmin=0, vmax=255)
                    ax_prev.axis('off')
                    st.pyplot(fig_prev)
                    st.markdown('</div>', unsafe_allow_html=True)
        
    elif mode_selection == "Téléverser une image":
        col_upload, col_upload_preview = st.columns([1, 1])
        with col_upload:
            uploaded_file = st.file_uploader("Choisissez une image de chiffre (Format carré recommandé)", type=["png", "jpg", "jpeg"])
            
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('L')
            # Réduire la résolution pour éviter des lenteurs et garder un trait épais (ex: photo 4K)
            image.thumbnail((280, 280), Image.LANCZOS)
            img_array = np.array(image) / 255.0
            
            # Détection du fond pour savoir s'il faut inverser
            border_mean = (img_array[0, :].mean() + img_array[-1, :].mean() + img_array[:, 0].mean() + img_array[:, -1].mean()) / 4.0
            
            # --- Nettoyage intelligent du bruit (Adaptive Thresholding) ---
            # Un filtre gaussien estime l'éclairage/ombre locale de la feuille de papier
            bg = scipy.ndimage.gaussian_filter(img_array, sigma=15)
            
            if border_mean > 0.5:
                # Fond clair (ex: photo papier), l'encre est plus sombre que le fond
                signal = bg - img_array
            else:
                # Fond sombre (ex: screenshot), l'encre est plus claire
                signal = img_array - bg
                
            # Éliminer le bruit léger
            signal[signal < 0.08] = 0
            
            if signal.max() > 0:
                signal = signal / signal.max()
                
            gray_scaled = (signal * 255).astype(np.uint8)
            
            # Application du prétraitement exact MNIST avec épaississement sur la haute résolution
            processed_28 = preprocess_to_mnist_format(gray_scaled, thicken_factor=8)
            
            if processed_28.max() > 0:
                pixels = (processed_28 / 255.0).flatten().tolist()
                
                with col_upload_preview:
                    st.markdown('<div class="glass-card" style="text-align:center">', unsafe_allow_html=True)
                    st.markdown("**Aperçu 28×28 transmis aux modèles**")
                    configure_plot_theme()
                    fig_up, ax_up = plt.subplots(figsize=(2, 2))
                    ax_up.imshow(processed_28, cmap='gray', vmin=0, vmax=255)
                    ax_up.axis('off')
                    st.pyplot(fig_up)
                    st.markdown('</div>', unsafe_allow_html=True)
            
    elif mode_selection == "Prendre un chiffre MNIST aléatoire":
        if st.button("Charger un chiffre MNIST au hasard"):
            rand_idx = np.random.randint(0, len(X))
            st.session_state.rand_img = X[rand_idx]
            st.session_state.rand_label = y[rand_idx]
            
        if 'rand_img' in st.session_state:
            pixels = st.session_state.rand_img.tolist()
            st.write(f"**Chiffre MNIST original chargé :** {st.session_state.rand_label}")
            
            # Affichage de l'image
            configure_plot_theme()
            fig_rand, ax_rand = plt.subplots(figsize=(2, 2))
            ax_rand.imshow(st.session_state.rand_img.reshape(28, 28), cmap="gray")
            ax_rand.axis("off")
            st.pyplot(fig_rand)
 
    # 2. Exécution des prédictions
    if pixels is not None:
        x_input = np.array(pixels).reshape(1, -1)
        
        with st.spinner("Prédiction multi-classes en cours (0 à 9)..."):
            pca_multi, models_raw, models_pca = get_multiclass_models(X, y)
            x_input_multi_pca = pca_multi.transform(x_input)
            
            preds = {}
            # Modèles sur pixels bruts
            for name, model in models_raw.items():
                preds[name] = model.predict(x_input)[0]
            # Modèles sur PCA
            for name, model in models_pca.items():
                preds[name] = model.predict(x_input_multi_pca)[0]
                
        # Affichage des prédictions sous forme de cartes d'informations
        st.subheader("Prédictions Multi-classes (0 à 9)")
        
        col_res1, col_res2, col_res3 = st.columns(3)
        
        with col_res1:
            st.markdown(
                f"""
<div class="glass-card" style="text-align: center;">
<p class="metric-title">LDA</p>
<p class="metric-value">{"Chiffre " + str(preds['LDA'])}</p>
</div>
<div class="glass-card" style="text-align: center;">
<p class="metric-title">QDA</p>
<p class="metric-value pink">{"Chiffre " + str(preds['QDA'])}</p>
</div>
""",
                unsafe_allow_html=True
            )
            
        with col_res2:
            st.markdown(
                f"""
<div class="glass-card" style="text-align: center;">
<p class="metric-title">Logistic Regression</p>
<p class="metric-value purple">{"Chiffre " + str(preds['Logistic Regression'])}</p>
</div>
<div class="glass-card" style="text-align: center;">
<p class="metric-title">Decision Tree</p>
<p class="metric-value">{"Chiffre " + str(preds['Decision Tree'])}</p>
</div>
""",
                unsafe_allow_html=True
            )
            
        with col_res3:
            st.markdown(
                f"""
<div class="glass-card" style="text-align: center;">
<p class="metric-title">KNN (K=5)</p>
<p class="metric-value pink">{"Chiffre " + str(preds['KNN'])}</p>
</div>
""",
                unsafe_allow_html=True
            )
    else:
        st.info("En attente d'une entrée utilisateur. Veuillez dessiner un chiffre ou importer un fichier.")