# ===================== IMPORT =====================
import streamlit as st
import pandas as pd
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

# ===================== UTILITIES =====================
def is_valid_data(filename):
    return filename.lower().endswith((".csv", ".json", ".xls", ".xlsx"))

def load_data(uploaded_file):
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    if ext == ".csv":
        return pd.read_csv(uploaded_file)
    elif ext == ".json":
        return pd.read_json(uploaded_file)
    elif ext in (".xls", ".xlsx"):
        return pd.read_excel(uploaded_file)
    else:
        raise ValueError("Format file tidak didukung")

def rename_column(df, col_name):
    return df.rename(columns={col_name: "komentar"})

# ===================== PREPROCESSING =====================
def preprocess_text(df):
    def clean(text):
        text = str(text).lower()
        text = re.sub(r"http\S+|[^a-z\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    df["komentar"] = df["komentar"].astype(str)
    df["clean"] = df["komentar"].apply(clean)
    return df

# ===================== TOPIC MODELING =====================
def topic_modeling(df, n_topics=2):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df["clean"])

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42
    )
    lda.fit(X)

    words = vectorizer.get_feature_names_out()
    topics = []
    for topic in lda.components_:
        topics.append(", ".join(words[i] for i in topic.argsort()[-5:]))

    df["Topik_Utama"] = topics[0]
    return df, topics

# ===================== VISUALISASI TOPIC =====================
def plot_top_words(model, feature_names, n_top_words=10):
    fig, ax = plt.subplots(figsize=(8, 4))
    top_features = model.components_[0].argsort()[-n_top_words:]
    words = [feature_names[i] for i in top_features]
    weights = model.components_[0][top_features]

    ax.barh(words, weights)
    ax.set_title("Kata Dominan pada Topik")
    plt.tight_layout()
    return fig

# ===================== CLUSTERING =====================
def elbow_plot(data):
    wcss = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, 11), wcss, marker="o")
    ax.set_title("Elbow Method")
    ax.set_xlabel("Jumlah Cluster")
    ax.set_ylabel("WCSS")
    return fig

def best_kmeans(data):
    best_k, best_score = 2, -1
    for k in range(2, 11):
        labels = KMeans(k, random_state=42).fit_predict(data)
        score = silhouette_score(data, labels)
        if score > best_score:
            best_k, best_score = k, score
    return best_k

def cluster_and_visualize(df):
    best_k = best_kmeans(df)
    kmeans = KMeans(n_clusters=best_k, random_state=42)
    df["Cluster"] = kmeans.fit_predict(df)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(x=df["Cluster"], ax=ax)
    ax.set_title("Distribusi Cluster")
    return df, fig, best_k

# ===================== STREAMLIT APP =====================
st.sidebar.title("Modeling")
menu = st.sidebar.selectbox(
    "Pilih Menu",
    ("Topic Modeling - Komentar", "Clustering - Data Ordinal")
)

# ===================== TOPIC MODELING =====================
if menu == "Topic Modeling - Komentar":
    st.title("🟢 Topic Modeling - Komentar")

    uploaded_file = st.file_uploader(
        "Upload File",
        type=["csv", "json", "xls", "xlsx"]
    )

    if uploaded_file:
        df = load_data(uploaded_file)
        st.subheader("Data Awal")
        st.dataframe(df)

        col = st.text_input("Nama kolom komentar")

        if col:
            df = rename_column(df, col)
            df = preprocess_text(df)

            st.subheader("Hasil Preprocessing")
            st.dataframe(df[["komentar", "clean"]])

            if st.button("Latih Topic Model"):
                df, topics = topic_modeling(df)
                st.subheader("Hasil Topic Modeling")
                st.dataframe(df[["komentar", "Topik_Utama"]])

                df.to_csv("topic_model.csv", index=False)

# ===================== CLUSTERING =====================
elif menu == "Clustering - Data Ordinal":
    st.title("🔵 Clustering - Data Ordinal")

    uploaded_file = st.file_uploader(
        "Upload Data Likert",
        type=["csv", "xls", "xlsx"]
    )

    if uploaded_file:
        df = load_data(uploaded_file)
        st.subheader("Data Likert")
        st.dataframe(df)

        if st.button("Latih Clustering"):
            st.subheader("Visualisasi Elbow")
            fig_elbow = elbow_plot(df)
            st.pyplot(fig_elbow)

            df_clustered, fig_cluster, k = cluster_and_visualize(df)
            st.success(f"Jumlah cluster terbaik: {k}")

            st.subheader("Distribusi Cluster")
            st.pyplot(fig_cluster)

            st.subheader("Data dengan Cluster")
            st.dataframe(df_clustered)

            df_clustered.to_csv("data_cluster.csv", index=False)
