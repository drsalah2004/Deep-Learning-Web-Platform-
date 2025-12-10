import os
import warnings
import logging

# ============================================
# CONFIGURATION DES WARNINGS - DOIT ÊTRE EN PREMIER
# ============================================
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Supprimer les warnings scikit-learn
import sklearn
sklearn.set_config(assume_finite=True)

# Imports principaux
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# TensorFlow en mode silencieux
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow.keras.models import load_model

from sklearn.preprocessing import MinMaxScaler

# ===================== CONFIG PAGE =====================
st.set_page_config(
    page_title="Prévision de Consommation Électrique",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================== STYLE =====================
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Titre principal
st.markdown('<h1 class="main-header">⚡ Prévision de Consommation Électrique</h1>', unsafe_allow_html=True)

# ===================== SIDEBAR =====================
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Choisir une page:",
    ["🏠 Accueil", "📈 Prédictions sur Fichier", "🔍 Exploration des Données", "📊 Comparaison des Modèles"]
)

# ===================== CHARGEMENT DES MODÈLES =====================
@st.cache_resource(show_spinner=False)
def load_models():
    models = {}
    model_files = {
        "Médiane": "median_model.pkl",
        "RNN": "rnn_model.h5",
        "LSTM Stacked": "lstm_stacked_model.h5",
        "BiLSTM": "bilstm_model.h5",
        "MLP": "mlp_model.h5",
        "CNN": "cnn_model.h5",
        "KMeans": "kmeans_model.pkl",
        "DBSCAN": "dbscan_model.pkl"
    }
    
    for name, file in model_files.items():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if file.endswith('.pkl'):
                    models[name] = joblib.load(file)
                elif file.endswith('.h5'):
                    models[name] = load_model(file, compile=False)
            st.sidebar.success(f"✓ {name}")
        except FileNotFoundError:
            st.sidebar.warning(f"⚠ {name} non trouvé")
            models[name] = None
        except Exception:
            st.sidebar.error(f"✗ {name} erreur")
            models[name] = None
    
    return models

@st.cache_resource(show_spinner=False)
def load_scaler():
    try:
        return joblib.load("scaler.pkl")
    except:
        st.warning("⚠ Scaler introuvable, création d'un nouveau")
        return MinMaxScaler()

# ===================== FONCTIONS UTILES =====================
def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i + n_steps])
        y.append(data[i + n_steps])
    return np.array(X), np.array(y)

def clean_data(df):
    df = df.replace(['?', '', ' '], np.nan)
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except:
            pass
    return df

def handle_missing_values(data, strategy='mean'):
    if data.isnull().sum().sum() > 0:
        if strategy == 'mean':
            return data.fillna(data.mean())
        elif strategy == 'median':
            return data.fillna(data.median())
        elif strategy == 'forward':
            return data.fillna(method='ffill').fillna(method='bfill')
        elif strategy == 'drop':
            return data.dropna()
    return data

# ===================== PAGE ACCUEIL =====================
if page == "🏠 Accueil":
    st.header("Bienvenue dans l'application de prévision")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("### 📊 Modèles Disponibles\n- Médiane\n- RNN\n- LSTM Stacked\n- BiLSTM\n- MLP\n- CNN\n- KMeans / DBSCAN")
    with col2:
        st.success("### 🎯 Objectif\nPrédire la consommation électrique globale")
    with col3:
        st.warning("### 📁 Données\nSérie temporelle multi-features")

    st.divider()

    models = load_models()

    st.subheader("📦 État des Modèles")
    cols = st.columns(5)
    model_names = list(models.keys())

    for idx, name in enumerate(model_names):
        with cols[idx % 5]:
            if models[name] is not None:
                st.success(f"✅ {name}")
            else:
                st.error(f"❌ {name}")

# ===================== PAGE PRÉDICTION SUR FICHIER =====================
elif page == "📈 Prédictions sur Fichier":
    st.header("Prédictions avec les Modèles")

    models = load_models()
    scaler = load_scaler()

    uploaded_file = st.file_uploader("📂 Charger un fichier CSV", type=['csv'])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        df = clean_data(df)

        missing_before = df.isnull().sum().sum()

        st.success(f"Fichier chargé ({df.shape[0]} lignes, {df.shape[1]} colonnes)")

        if missing_before > 0:
            st.warning(f"{missing_before} valeurs manquantes nettoyées")

        target_col = st.selectbox("Sélectionner la colonne cible:", df.columns)

        missing_strategy = st.selectbox(
            "Stratégie valeurs manquantes:",
            ["mean", "median", "forward", "drop"]
        )

        col1, col2 = st.columns(2)
        with col1:
            n_steps = st.slider("Nombre de pas de temps:", 5, 50, 10)
        with col2:
            test_size = st.slider("Taille du test (%):", 10, 40, 20)

        model_choice = st.selectbox(
            "Choisir un modèle:",
            ["Médiane", "SARIMA", "RNN", "LSTM Stacked", "BiLSTM", "MLP", "CNN"]
        )

        if st.button("🚀 Lancer la Prédiction"):

            data_series = df[target_col]

            data_series = handle_missing_values(data_series, strategy=missing_strategy)

            data = data_series.values.reshape(-1, 1)

            data_scaled = scaler.fit_transform(data)

            train_size = int(len(data_scaled) * (1 - test_size/100))
            train, test = data_scaled[:train_size], data_scaled[train_size:]

            model = models.get(model_choice)

            if model is None:
                st.error("Modèle indisponible")
                st.stop()

            try:
                if model_choice == "Médiane":
                    y_pred = np.full(len(test), model)
                    y_test = test
                elif model_choice == "SARIMA":
                    forecast = model.forecast(len(test))
                    y_pred = forecast.values.reshape(-1, 1)
                    y_test = scaler.inverse_transform(test)
                else:
                    X_test, y_test = create_sequences(test, n_steps)
                    X_test_seq = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
                    y_pred = model.predict(X_test_seq, verbose=0)

                y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
                y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1))

                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
                mae = mean_absolute_error(y_test_inv, y_pred_inv)
                r2 = r2_score(y_test_inv, y_pred_inv)

                st.subheader("📊 Métriques de Performance")
                col1, col2, col3 = st.columns(3)
                col1.metric("RMSE", f"{rmse:.4f}")
                col2.metric("MAE", f"{mae:.4f}")
                col3.metric("R²", f"{r2:.4f}")

                st.subheader("📈 Comparaison Réel vs Prédiction")

                fig, ax = plt.subplots(figsize=(12, 6))
                n_display = min(200, len(y_test_inv))
                ax.plot(y_test_inv[:n_display], label="Réel", color='black')
                ax.plot(y_pred_inv[:n_display], '--', label="Prédiction", color='red')
                ax.legend()
                ax.grid(alpha=0.3)

                st.pyplot(fig)

            except Exception as e:
                st.error(f"Erreur: {str(e)}")

# ===================== PAGE EXPLORATION =====================
elif page == "🔍 Exploration des Données":
    st.header("Exploration des Données")

    uploaded_file = st.file_uploader("📂 Charger un fichier CSV", type=['csv'])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df = clean_data(df)
        df = handle_missing_values(df, strategy='mean')

        st.subheader("📊 Aperçu")
        st.dataframe(df.head(10))

        col1, col2, col3 = st.columns(3)
        col1.metric("Lignes", df.shape[0])
        col2.metric("Colonnes", df.shape[1])
        col3.metric("Valeurs manquantes", df.isnull().sum().sum())

        st.subheader("📈 Statistiques")
        st.dataframe(df.describe())

        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

        st.subheader("📊 Distribution")
        selected_col = st.selectbox("Colonne:", numeric_cols)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(df[selected_col], bins=50)
        ax.set_title(f"Distribution de {selected_col}")
        st.pyplot(fig)

# ===================== PAGE COMPARAISON =====================
elif page == "📊 Comparaison des Modèles":
    st.header("Comparaison des Modèles")

    if st.checkbox("Afficher exemple"):
        models_rmse = {
            "Médiane": 150.23,
            "RNN": 87.32,
            "LSTM Stacked": 78.90,
            "BiLSTM": 76.45,
            "MLP": 85.67,
            "CNN": 81.23
        }

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(models_rmse.keys(), models_rmse.values())
        ax.set_ylabel("RMSE")
        st.pyplot(fig)

# ===================== FOOTER =====================
st.divider()
st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>📊 Application de Prévision de Consommation Électrique | Développée avec Streamlit</p>
    </div>
""", unsafe_allow_html=True)
