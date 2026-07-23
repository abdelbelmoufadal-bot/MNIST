import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap

@st.cache_resource
def get_manifold_embeddings(X_bin_pca, y_bin):
    if len(X_bin_pca) > 2500:
        np.random.seed(42)
        idx = np.random.choice(len(X_bin_pca), 2500, replace=False)
        X_sub = X_bin_pca[idx]
        y_sub = y_bin[idx]
    else:
        X_sub = X_bin_pca
        y_sub = y_bin
        
    tsne = TSNE(n_components=2, random_state=42)
    tsne_emb = tsne.fit_transform(X_sub)
    
    umapper = umap.UMAP(n_components=2, random_state=42)
    umap_emb = umapper.fit_transform(X_sub)
    
    return y_sub, tsne_emb, umap_emb

def render_manifold_page(X_bin_pca, y_bin, digit_a, digit_b, configure_plot_theme):
    st.markdown('<h1 class="gradient-text">Réduction de Dimensionnalité Avancée</h1>', unsafe_allow_html=True)
    st.markdown(f'<p class="gradient-subtext">Comparez t-SNE et UMAP pour la visualisation 2D des chiffres {digit_a} et {digit_b}. Ces algorithmes non-linéaires capturent la structure complexe des données (contrairement à la PCA linéaire).</p>', unsafe_allow_html=True)
    
    with st.spinner("Calcul des projections UMAP et t-SNE en cours (cela peut prendre quelques secondes)..."):
        y_sub, tsne_emb, umap_emb = get_manifold_embeddings(X_bin_pca, y_bin)
        
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("t-SNE (t-Distributed Stochastic Neighbor Embedding)")
        st.info("Privilégie la structure locale : garde les points similaires très proches, mais perd un peu la structure globale.")
        configure_plot_theme()
        fig_tsne, ax_tsne = plt.subplots(figsize=(6, 5))
        
        scatter1 = ax_tsne.scatter(tsne_emb[y_sub == 0, 0], tsne_emb[y_sub == 0, 1], color='#38BDF8', alpha=0.6, label=f'Chiffre {digit_a}')
        scatter2 = ax_tsne.scatter(tsne_emb[y_sub == 1, 0], tsne_emb[y_sub == 1, 1], color='#F472B6', alpha=0.6, label=f'Chiffre {digit_b}')
        
        ax_tsne.set_xlabel("Dimension 1", color="#94A3B8")
        ax_tsne.set_ylabel("Dimension 2", color="#94A3B8")
        ax_tsne.legend(facecolor="none", edgecolor="none")
        st.pyplot(fig_tsne)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("UMAP (Uniform Manifold Approximation and Projection)")
        st.info("Trouve un compromis exceptionnel entre la préservation des relations locales et la structure globale du jeu de données.")
        configure_plot_theme()
        fig_umap, ax_umap = plt.subplots(figsize=(6, 5))
        
        scatter3 = ax_umap.scatter(umap_emb[y_sub == 0, 0], umap_emb[y_sub == 0, 1], color='#38BDF8', alpha=0.6, label=f'Chiffre {digit_a}')
        scatter4 = ax_umap.scatter(umap_emb[y_sub == 1, 0], umap_emb[y_sub == 1, 1], color='#F472B6', alpha=0.6, label=f'Chiffre {digit_b}')
        
        ax_umap.set_xlabel("Dimension 1", color="#94A3B8")
        ax_umap.set_ylabel("Dimension 2", color="#94A3B8")
        ax_umap.legend(facecolor="none", edgecolor="none")
        st.pyplot(fig_umap)
        st.markdown('</div>', unsafe_allow_html=True)
