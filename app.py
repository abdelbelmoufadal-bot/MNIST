import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from PIL import Image

from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import OneClassSVM


st.set_page_config(
    page_title="AI Digits Classification Lab",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
<style>
.big-title {
    font-size:42px;
    font-weight:800;
    color:#38BDF8;
}
.subtitle {
    font-size:18px;
    color:#CBD5E1;
}
.box {
    padding:18px;
    border-radius:15px;
    background-color:#111827;
    border:1px solid #334155;
}
</style>
""", unsafe_allow_html=True)


st.markdown('<div class="big-title">🤖 AI Digits Classification Lab</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Master IA Mini-Lab : PCA, LDA, QDA, Logistic Regression, Decision Tree, KNN, One-Class SVM</div>',
    unsafe_allow_html=True
)

st.markdown("---")


@st.cache_data
def load_data():
    digits = load_digits()
    X = digits.data / 16.0
    y = digits.target
    mask = (y == 0) | (y == 1)
    return X[mask], y[mask]


X, y = load_data()

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_pca,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "LDA": LinearDiscriminantAnalysis(),
    "QDA": QuadraticDiscriminantAnalysis(),
    "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=4),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}


st.sidebar.header("⚙️ Control Panel")

selected_model_name = st.sidebar.selectbox(
    "Choose a model",
    list(models.keys())
)

index = st.sidebar.slider(
    "Choose an image",
    0,
    len(X) - 1,
    0
)

show_boundary = st.sidebar.checkbox("Show Decision Boundary", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("📌 Project Info")
st.sidebar.write("Dataset: sklearn Digits")
st.sidebar.write(f"Samples: {len(X)}")
st.sidebar.write(f"Features: {X.shape[1]}")
st.sidebar.write("Classes: 0 and 1")
st.sidebar.write("PCA Components: 2")


model = models[selected_model_name]

t0 = time.time()
model.fit(X_train, y_train)
training_time = time.time() - t0

t1 = time.time()
y_pred_test = model.predict(X_test)
prediction_time = time.time() - t1

accuracy = accuracy_score(y_test, y_pred_test)

sample = X[index].reshape(1, -1)
sample_pca = pca.transform(sample)
prediction = model.predict(sample_pca)[0]

try:
    confidence = np.max(model.predict_proba(sample_pca))
except:
    confidence = None


tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🔍 Prediction",
    "📊 PCA & Boundary",
    "📈 Evaluation",
    "🏆 Model Comparison",
    "🧠 Math Behind",
    "🧪 Upload / One-Class SVM"
])


with tab1:
    st.subheader("Prediction Panel")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.image(X[index].reshape(8, 8), width=230, caption="Selected Image")

    with c2:
        st.metric("True Label", int(y[index]))
        st.metric("Predicted Label", int(prediction))
        if confidence is not None:
            st.metric("Confidence", f"{confidence * 100:.2f}%")
        else:
            st.metric("Confidence", "Not available")

    with c3:
        st.metric("Model", selected_model_name)
        st.metric("Accuracy", f"{accuracy * 100:.2f}%")
        st.metric("Training Time", f"{training_time:.4f} sec")

        if int(y[index]) == int(prediction):
            st.success("✅ Correct prediction")
        else:
            st.error("❌ Wrong prediction")

    st.markdown("---")
    st.subheader("🎲 Random Test")

    if st.button("Test Random Image"):
        random_index = np.random.randint(0, len(X))
        random_sample = X[random_index].reshape(1, -1)
        random_pca = pca.transform(random_sample)
        random_prediction = model.predict(random_pca)[0]

        r1, r2 = st.columns(2)

        with r1:
            st.image(X[random_index].reshape(8, 8), width=230, caption="Random Image")

        with r2:
            st.metric("True Label", int(y[random_index]))
            st.metric("Predicted Label", int(random_prediction))

            if int(y[random_index]) == int(random_prediction):
                st.success("✅ Correct prediction")
            else:
                st.error("❌ Wrong prediction")


with tab2:
    st.subheader("PCA Projection and Decision Boundary")

    fig, ax = plt.subplots(figsize=(9, 6))

    if show_boundary:
        h = 0.05
        x_min, x_max = X_pca[:, 0].min() - 0.5, X_pca[:, 0].max() + 0.5
        y_min, y_max = X_pca[:, 1].min() - 0.5, X_pca[:, 1].max() + 0.5

        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, h),
            np.arange(y_min, y_max, h)
        )

        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        ax.contourf(xx, yy, Z, alpha=0.25)

    ax.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], s=20, label="Digit 0")
    ax.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], s=20, label="Digit 1")
    ax.scatter(sample_pca[:, 0], sample_pca[:, 1], s=220, marker="X", label="Selected Image")

    ax.set_title(f"PCA Projection - {selected_model_name}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()

    st.pyplot(fig)

    st.info("PCA transforms the original image features into two principal components for visualization.")


with tab3:
    st.subheader("Model Evaluation")

    e1, e2 = st.columns(2)

    with e1:
        st.metric("Accuracy", f"{accuracy * 100:.2f}%")
        st.metric("Training Time", f"{training_time:.4f} sec")
        st.metric("Prediction Time", f"{prediction_time:.4f} sec")

        cm = confusion_matrix(y_test, y_pred_test)

        st.write("Confusion Matrix")
        st.dataframe(
            pd.DataFrame(
                cm,
                index=["True 0", "True 1"],
                columns=["Pred 0", "Pred 1"]
            )
        )

    with e2:
        st.write("Confusion Matrix Heatmap")

        fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
        im = ax_cm.imshow(cm)

        ax_cm.set_xticks([0, 1])
        ax_cm.set_yticks([0, 1])
        ax_cm.set_xticklabels(["Pred 0", "Pred 1"])
        ax_cm.set_yticklabels(["True 0", "True 1"])

        for i in range(2):
            for j in range(2):
                ax_cm.text(j, i, cm[i, j], ha="center", va="center")

        ax_cm.set_title("Confusion Matrix")
        fig_cm.colorbar(im)
        st.pyplot(fig_cm)

    st.write("Classification Report")
    report = classification_report(y_test, y_pred_test, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())


with tab4:
    st.subheader("Comparison Between All Models")

    results = []

    for name, clf in models.items():
        start = time.time()
        clf.fit(X_train, y_train)
        train_time = time.time() - start

        pred = clf.predict(X_test)
        acc = accuracy_score(y_test, pred)

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Training Time": train_time
        })

    results_df = pd.DataFrame(results)
    st.dataframe(results_df)

    fig_bar, ax_bar = plt.subplots(figsize=(8, 5))
    ax_bar.bar(results_df["Model"], results_df["Accuracy"])
    ax_bar.set_ylim(0, 1.05)
    ax_bar.set_ylabel("Accuracy")
    ax_bar.set_title("Model Accuracy Comparison")
    plt.xticks(rotation=25)

    st.pyplot(fig_bar)


with tab5:
    st.subheader("Mathematical Explanation")

    st.markdown("""
### 1. PCA — Principal Component Analysis

PCA reduces the dimension of the data.

\[
X_{new} = XW
\]

The first principal component keeps the maximum variance.

---

### 2. Logistic Regression

It estimates the probability of belonging to class 0 or 1.

\[
P(y=1|x)=\\frac{1}{1+e^{-w^Tx}}
\]

---

### 3. LDA — Linear Discriminant Analysis

LDA maximizes the separation between classes and minimizes the dispersion inside each class.

\[
J(w)=\\frac{w^T S_B w}{w^T S_W w}
\]

This is related to the Fisher criterion.

---

### 4. QDA — Quadratic Discriminant Analysis

QDA allows each class to have its own covariance matrix, which creates a curved boundary.

---

### 5. Decision Tree

A tree classifies data using simple decision rules.

Example:

If PC1 < threshold → class 0  
Else → class 1

---

### 6. KNN

KNN classifies a new point according to the majority class among its nearest neighbors.

\[
d(x_i,x_j)=\sqrt{\sum (x_i-x_j)^2}
\]

---

### 7. One-Class SVM

It learns only one class, for example digit 0, then detects whether a new image is similar or not.
""")


with tab6:
    st.subheader("Upload Image and One-Class SVM")

    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("L")
        img = img.resize((8, 8))
        img_array = np.array(img)

        img_array = 16 - (img_array / 255.0 * 16)
        img_array = np.clip(img_array, 0, 16)
        img_vector = img_array.reshape(1, -1) / 16.0

        img_pca = pca.transform(img_vector)
        uploaded_prediction = model.predict(img_pca)[0]

        u1, u2 = st.columns(2)

        with u1:
            st.image(img_array, width=230, caption="Uploaded Image")

        with u2:
            st.metric("Predicted Label", int(uploaded_prediction))
            st.write("The uploaded image is resized to 8x8 and projected using PCA.")

    st.markdown("---")
    st.subheader("One-Class SVM: 0 or Not 0")

    ocsvm = OneClassSVM(kernel="rbf", gamma="auto", nu=0.05)
    X_zero = X_pca[y == 0]
    ocsvm.fit(X_zero)

    oc_pred = ocsvm.predict(sample_pca)

    if oc_pred[0] == 1:
        st.success("One-Class SVM result: Recognized as digit 0")
    else:
        st.error("One-Class SVM result: Not 0")

    fig_svm, ax_svm = plt.subplots(figsize=(8, 6))

    oc_all = ocsvm.predict(X_pca)

    ax_svm.scatter(X_pca[oc_all == 1, 0], X_pca[oc_all == 1, 1], s=20, label="Recognized as 0")
    ax_svm.scatter(X_pca[oc_all == -1, 0], X_pca[oc_all == -1, 1], s=20, label="Not 0")
    ax_svm.scatter(sample_pca[:, 0], sample_pca[:, 1], s=220, marker="X", label="Selected Image")

    ax_svm.set_title("One-Class SVM Visualization")
    ax_svm.set_xlabel("PC1")
    ax_svm.set_ylabel("PC2")
    ax_svm.legend()

    st.pyplot(fig_svm)


st.markdown("---")
st.caption("Developed by AYYOUB REGUIGUE | Master IA | Python • Scikit-learn • Streamlit")