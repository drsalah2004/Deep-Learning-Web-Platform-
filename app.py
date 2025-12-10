import os
import warnings
import logging

# ============================================
# CONFIGURATION DES WARNINGS - DOIT ÊTRE EN PREMIER
# ============================================
# Supprimer TOUS les warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Supprimer logs TensorFlow (0=all, 1=info, 2=warning, 3=error)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Supprimer les warnings scikit-learn
import sklearn
sklearn.set_config(assume_finite=True)

# Maintenant importer le reste
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Importer TensorFlow en mode silencieux
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow.keras.models import load_model

from sklearn.preprocessing import MinMaxScaler
import json

# Configuration de la page
st.set_page_config(
    page_title="Prévision de Consommation Électrique",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
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

# Sidebar pour la navigation
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Choisir une page:",
    ["🏠 Accueil", "📈 Prédictions sur Fichier", "🎯 Prédiction Personnalisée", "🔍 Exploration des Données", "📊 Comparaison des Modèles"]
)

# Fonction pour charger les modèles sans afficher warnings
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
        except Exception as e:
            st.sidebar.error(f"✗ {name} erreur")
            models[name] = None
    
    return models

# Fonction pour charger le scaler sans warnings
@st.cache_resource(show_spinner=False)
def load_scaler():
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return joblib.load("scaler.pkl")
    except:
        st.warning("⚠ Scaler non trouvé, création d'un nouveau")
        return MinMaxScaler()

# Fonction pour créer des séquences
def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i + n_steps])
        y.append(data[i + n_steps])
    return np.array(X), np.array(y)

# Fonction pour nettoyer les données
def clean_data(df):
    """
    Nettoie le DataFrame en remplaçant les valeurs invalides
    et en convertissant en numérique
    """
    # Remplacer les valeurs problématiques
    df = df.replace('?', np.nan)
    df = df.replace('', np.nan)
    df = df.replace(' ', np.nan)
    
    # Convertir toutes les colonnes en numérique si possible
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except:
            pass
    
    return df

# Fonction pour gérer les valeurs manquantes
def handle_missing_values(data, strategy='mean'):
    """
    Gère les valeurs manquantes dans les données
    strategy: 'mean', 'median', 'forward', 'drop'
    """
    if isinstance(data, pd.DataFrame):
        missing_count = data.isnull().sum().sum()
    elif isinstance(data, pd.Series):
        missing_count = data.isnull().sum()
    else:
        return data
    
    if missing_count > 0:
        if strategy == 'mean':
            data = data.fillna(data.mean())
        elif strategy == 'median':
            data = data.fillna(data.median())
        elif strategy == 'forward':
            data = data.fillna(method='ffill').fillna(method='bfill')
        elif strategy == 'drop':
            data = data.dropna()
    
    return data

# ==================== PAGE ACCUEIL ====================
if page == "🏠 Accueil":
    st.header("Bienvenue dans l'application de prévision")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("### 📊 Modèles Disponibles\n- Médiane (Baseline)\n- RNN\n- LSTM Stacked\n- BiLSTM\n- MLP\n- CNN\n- KMeans & DBSCAN")
    
    with col2:
        st.success("### 🎯 Objectif\nPrédire la consommation électrique globale en utilisant différents algorithmes de ML et Deep Learning")
    
    with col3:
        st.warning("### 📁 Données\nSérie temporelle de consommation électrique avec plusieurs features")
    
    st.divider()
    
    # Charger les modèles
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

# ==================== PAGE PRÉDICTIONS SUR FICHIER ====================
elif page == "📈 Prédictions sur Fichier":
    st.header("Prédictions avec les Modèles")
    
    # Charger les modèles et scaler
    models = load_models()
    scaler = load_scaler()
    
    # Upload de fichier
    uploaded_file = st.file_uploader("📂 Charger un fichier CSV", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        # Nettoyer les données
        df = clean_data(df)
        
        # Afficher les infos sur les valeurs manquantes
        missing_before = df.isnull().sum().sum()
        
        st.success(f"✅ Fichier chargé: {df.shape[0]} lignes, {df.shape[1]} colonnes")
        
        if missing_before > 0:
            st.warning(f"⚠️ {missing_before} valeurs manquantes détectées et nettoyées")
        
        # Sélection de la colonne cible
        target_col = st.selectbox("Sélectionner la colonne cible:", df.columns)
        
        # Option de gestion des valeurs manquantes
        missing_strategy = st.selectbox(
            "Stratégie pour les valeurs manquantes:",
            ["mean", "median", "forward", "drop"],
            help="mean: moyenne, median: médiane, forward: propagation, drop: suppression"
        )
        
        # Paramètres
        col1, col2 = st.columns(2)
        
        with col1:
            n_steps = st.slider("Nombre de pas de temps (n_steps):", 5, 50, 10)
        
        with col2:
            test_size = st.slider("Taille du test (%):", 10, 40, 20)
        
        # Sélection du modèle
        model_choice = st.selectbox(
            "Choisir un modèle:",
            ["Médiane", "SARIMA", "RNN", "LSTM Stacked", "BiLSTM", "MLP", "CNN"]
        )
        
        if st.button("🚀 Lancer la Prédiction"):
            with st.spinner(f"Prédiction en cours avec {model_choice}..."):
                
                # Préparation des données avec nettoyage complet
                data_series = df[target_col].copy()
                
                # Gérer les valeurs manquantes
                initial_missing = data_series.isnull().sum()
                if initial_missing > 0:
                    st.info(f"ℹ️ {initial_missing} valeurs manquantes traitées avec stratégie: {missing_strategy}")
                    data_series = handle_missing_values(data_series, strategy=missing_strategy)
                
                # Vérifier qu'il reste des données
                if len(data_series) == 0:
                    st.error("❌ Pas assez de données après nettoyage")
                    st.stop()
                
                # Convertir en array numpy
                data = data_series.values.reshape(-1, 1)
                
                # Vérifier les valeurs infinies
                if np.isinf(data).any():
                    st.warning("⚠️ Valeurs infinies détectées et remplacées")
                    data = np.nan_to_num(data, nan=np.nanmean(data), posinf=np.nanmax(data[~np.isinf(data)]), neginf=np.nanmin(data[~np.isinf(data)]))
                
                # Normalisation
                try:
                    data_scaled = scaler.fit_transform(data)
                except Exception as e:
                    st.error(f"❌ Erreur lors de la normalisation: {str(e)}")
                    st.write("Aperçu des données:", data[:10])
                    st.stop()
                
                # Split train/test
                train_size = int(len(data_scaled) * (1 - test_size/100))
                train, test = data_scaled[:train_size], data_scaled[train_size:]
                
                model = models[model_choice]
                
                if model is not None:
                    try:
                        # Prédiction selon le type de modèle
                        if model_choice == "Médiane":
                            y_pred = np.full(len(test), model)
                            y_test = test
                        
                        elif model_choice == "SARIMA":
                            forecast = model.forecast(len(test))
                            y_pred = forecast.values.reshape(-1, 1)
                            y_test = scaler.inverse_transform(test)
                            y_pred = y_pred
                        
                        elif model_choice in ["RNN", "LSTM Stacked", "BiLSTM", "CNN"]:
                            X_test, y_test = create_sequences(test, n_steps)
                            X_test_seq = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                y_pred = model.predict(X_test_seq, verbose=0)
                        
                        elif model_choice == "MLP":
                            X_test, y_test = create_sequences(test, n_steps)
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                y_pred = model.predict(X_test, verbose=0)
                        
                        # Inverse scaling
                        if model_choice != "SARIMA":
                            y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
                            y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1))
                        else:
                            y_test_inv = y_test
                            y_pred_inv = y_pred
                        
                        # Calcul RMSE
                        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                        rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
                        mae = mean_absolute_error(y_test_inv, y_pred_inv)
                        r2 = r2_score(y_test_inv, y_pred_inv)
                        
                        # Affichage des métriques
                        st.subheader("📊 Métriques de Performance")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("RMSE", f"{rmse:.4f}")
                        with col2:
                            st.metric("MAE", f"{mae:.4f}")
                        with col3:
                            st.metric("R² Score", f"{r2:.4f}")
                        
                        # Graphique
                        st.subheader("📈 Comparaison Valeurs Réelles vs Prédictions")
                        
                        fig, ax = plt.subplots(figsize=(12, 6))
                        
                        n_display = min(200, len(y_test_inv))
                        
                        ax.plot(y_test_inv[:n_display], label='Valeurs réelles', color='black', linewidth=2)
                        ax.plot(y_pred_inv[:n_display], label=f'Prédictions {model_choice}', 
                               color='red', linestyle='--', linewidth=2)
                        
                        ax.set_title(f'Modèle {model_choice} - RMSE: {rmse:.4f}', fontsize=14, fontweight='bold')
                        ax.set_xlabel('Temps (échantillons)', fontsize=12)
                        ax.set_ylabel('Valeur', fontsize=12)
                        ax.legend(fontsize=11)
                        ax.grid(alpha=0.3)
                        
                        st.pyplot(fig)
                        
                        # Tableau des premières prédictions
                        st.subheader("📋 Premières Prédictions")
                        results_df = pd.DataFrame({
                            'Valeurs Réelles': y_test_inv[:20].flatten(),
                            'Prédictions': y_pred_inv[:20].flatten(),
                            'Erreur': np.abs(y_test_inv[:20].flatten() - y_pred_inv[:20].flatten())
                        })
                        st.dataframe(results_df, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Erreur lors de la prédiction: {str(e)}")
                else:
                    st.error(f"Le modèle {model_choice} n'est pas disponible")

# ==================== PAGE PRÉDICTION PERSONNALISÉE ====================
elif page == "🎯 Prédiction Personnalisée":
    st.header("🎯 Prédiction Personnalisée")
    st.info("💡 Entrez vos propres valeurs pour obtenir une prédiction")
    
    # Charger les modèles et scaler
    models = load_models()
    scaler = load_scaler()
    
    # Charger le fichier pour connaître les features disponibles
    uploaded_file = st.file_uploader("📂 Charger votre fichier CSV (pour référence des colonnes)", type=['csv'], key="custom_pred")
    
    if uploaded_file is not None:
        df_ref = pd.read_csv(uploaded_file)
        
        # Nettoyer les données avec la fonction
        df_ref = clean_data(df_ref)
        
        # Gérer les valeurs manquantes
        df_ref = handle_missing_values(df_ref, strategy='mean')
        
        st.success(f"✅ Fichier chargé: {df_ref.shape[1]} colonnes disponibles")
        
        # Afficher les colonnes disponibles
        st.subheader("📋 Colonnes Disponibles")
        st.write(df_ref.columns.tolist())
        
        # Sélection des features
        st.subheader("🔧 Configuration de la Prédiction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Sélection du modèle
            model_choice = st.selectbox(
                "Choisir un modèle:",
                ["Médiane", "RNN", "LSTM Stacked", "BiLSTM", "MLP", "CNN"],
                key="model_custom"
            )
        
        with col2:
            n_steps = st.slider("Nombre de pas de temps:", 5, 50, 10, key="steps_custom")
        
        # Sélection des features à utiliser
        st.subheader("📊 Sélection des Features")
        
        numeric_cols = df_ref.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        selected_features = st.multiselect(
            "Choisir les features pour la prédiction:",
            numeric_cols,
            default=numeric_cols[:min(3, len(numeric_cols))]
        )
        
        if selected_features:
            st.subheader("✏️ Entrez vos Valeurs")
            st.write(f"Vous devez entrer **{n_steps}** valeurs pour chaque feature sélectionnée")
            
            # Créer des inputs pour chaque feature
            input_data = {}
            
            tabs = st.tabs(selected_features)
            
            for idx, feature in enumerate(selected_features):
                with tabs[idx]:
                    st.write(f"### {feature}")
                    
                    # Option: copier des valeurs du fichier
                    if st.checkbox(f"Utiliser des valeurs du fichier", key=f"use_file_{feature}"):
                        row_start = st.number_input(
                            f"Ligne de départ pour {feature}:", 
                            min_value=0, 
                            max_value=len(df_ref)-n_steps, 
                            value=0,
                            key=f"row_{feature}"
                        )
                        input_data[feature] = df_ref[feature].iloc[row_start:row_start+n_steps].values.tolist()
                        st.write(f"Valeurs sélectionnées: {input_data[feature]}")
                    else:
                        # Entrée manuelle
                        st.write("Entrez les valeurs (séparées par des virgules):")
                        values_str = st.text_input(
                            f"Valeurs pour {feature}:",
                            value=", ".join([str(round(df_ref[feature].mean(), 2))] * n_steps),
                            key=f"input_{feature}"
                        )
                        
                        try:
                            input_data[feature] = [float(x.strip()) for x in values_str.split(',')]
                            
                            if len(input_data[feature]) != n_steps:
                                st.error(f"⚠️ Vous devez entrer exactement {n_steps} valeurs!")
                        except:
                            st.error("⚠️ Format incorrect! Utilisez des nombres séparés par des virgules")
            
            # Vérifier que toutes les features ont le bon nombre de valeurs
            all_valid = all(
                feature in input_data and len(input_data[feature]) == n_steps 
                for feature in selected_features
            )
            
            if all_valid:
                st.success(f"✅ Toutes les valeurs sont correctes ({n_steps} valeurs par feature)")
                
                # Bouton de prédiction
                if st.button("🚀 Faire la Prédiction", type="primary"):
                    with st.spinner("Prédiction en cours..."):
                        try:
                            model = models[model_choice]
                            
                            if model is not None:
                                # Préparer les données
                                if len(selected_features) == 1:
                                    # Une seule feature
                                    input_array = np.array(input_data[selected_features[0]]).reshape(-1, 1)
                                else:
                                    # Plusieurs features
                                    input_array = np.array([input_data[f] for f in selected_features]).T
                                
                                # Normaliser
                                input_scaled = scaler.fit_transform(input_array)
                                
                                # Faire la prédiction selon le modèle
                                if model_choice == "Médiane":
                                    prediction_scaled = model
                                
                                elif model_choice in ["RNN", "LSTM Stacked", "BiLSTM", "CNN"]:
                                    # Utiliser seulement la première colonne (target)
                                    input_seq = input_scaled[:, 0].reshape(1, n_steps, 1)
                                    with warnings.catch_warnings():
                                        warnings.simplefilter("ignore")
                                        prediction_scaled = model.predict(input_seq, verbose=0)[0][0]
                                
                                elif model_choice == "MLP":
                                    # Utiliser seulement la première colonne (target)
                                    input_flat = input_scaled[:, 0].reshape(1, -1)
                                    with warnings.catch_warnings():
                                        warnings.simplefilter("ignore")
                                        prediction_scaled = model.predict(input_flat, verbose=0)[0][0]
                                
                                # Dénormaliser
                                prediction = scaler.inverse_transform([[prediction_scaled]])[0][0]
                                
                                # Afficher le résultat
                                st.subheader("🎉 Résultat de la Prédiction")
                                
                                col1, col2, col3 = st.columns([1, 2, 1])
                                
                                with col2:
                                    st.markdown(f"""
                                    <div style='text-align: center; padding: 2rem; background-color: #e3f2fd; border-radius: 1rem; border: 3px solid #1976d2;'>
                                        <h2 style='color: #1565c0; margin-bottom: 1rem;'>Prédiction</h2>
                                        <h1 style='color: #0d47a1; font-size: 3rem;'>{prediction:.4f}</h1>
                                        <p style='color: #424242; font-size: 1.2rem;'>Modèle: {model_choice}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                # Visualisation
                                st.subheader("📊 Visualisation")
                                
                                fig, ax = plt.subplots(figsize=(12, 6))
                                
                                # Tracer les valeurs d'entrée
                                x_input = list(range(n_steps))
                                y_input = input_data[selected_features[0]]
                                
                                ax.plot(x_input, y_input, 'o-', label='Valeurs d\'entrée', 
                                       color='blue', linewidth=2, markersize=8)
                                
                                # Tracer la prédiction
                                ax.plot([n_steps], [prediction], 'r*', 
                                       label='Prédiction', markersize=20)
                                
                                ax.axvline(x=n_steps-0.5, color='gray', linestyle='--', alpha=0.5)
                                
                                ax.set_xlabel('Pas de temps', fontsize=12)
                                ax.set_ylabel(selected_features[0], fontsize=12)
                                ax.set_title(f'Prédiction avec {model_choice}', fontsize=14, fontweight='bold')
                                ax.legend(fontsize=11)
                                ax.grid(alpha=0.3)
                                
                                st.pyplot(fig)
                                
                                # Détails
                                with st.expander("📋 Détails de la Prédiction"):
                                    st.write("**Valeurs d'entrée:**")
                                    for feature in selected_features:
                                        st.write(f"- {feature}: {input_data[feature]}")
                                    
                                    st.write(f"\n**Modèle utilisé:** {model_choice}")
                                    st.write(f"**Nombre de pas de temps:** {n_steps}")
                                    st.write(f"**Valeur prédite:** {prediction:.4f}")
                                
                            else:
                                st.error(f"Le modèle {model_choice} n'est pas disponible")
                        
                        except Exception as e:
                            st.error(f"Erreur lors de la prédiction: {str(e)}")
                            st.write("Détails de l'erreur:", e)
            else:
                st.warning("⚠️ Veuillez corriger les valeurs avant de continuer")

# ==================== PAGE EXPLORATION ====================
elif page == "🔍 Exploration des Données":
    st.header("Exploration des Données")
    
    uploaded_file = st.file_uploader("📂 Charger un fichier CSV", type=['csv'], key="explore")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        # Nettoyer les données
        df = clean_data(df)
        df = handle_missing_values(df, strategy='mean')
        
        st.subheader("📊 Aperçu des Données")
        st.dataframe(df.head(10), use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Nombre de lignes", df.shape[0])
        with col2:
            st.metric("Nombre de colonnes", df.shape[1])
        with col3:
            st.metric("Valeurs manquantes", df.isnull().sum().sum())
        
        st.subheader("📈 Statistiques Descriptives")
        st.dataframe(df.describe(), use_container_width=True)
        
        # Histogrammes
        st.subheader("📊 Distribution des Variables Numériques")
        
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        selected_col = st.selectbox("Sélectionner une colonne:", numeric_cols)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(df[selected_col].dropna(), bins=50, color='steelblue', edgecolor='black')
        ax.set_xlabel(selected_col, fontsize=12)
        ax.set_ylabel('Fréquence', fontsize=12)
        ax.set_title(f'Distribution de {selected_col}', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        
        st.pyplot(fig)
        
        # Série temporelle
        if st.checkbox("Afficher la série temporelle"):
            st.subheader("📈 Visualisation Temporelle")
            time_col = st.selectbox("Colonne temporelle:", df.columns)
            value_col = st.selectbox("Colonne de valeur:", numeric_cols)
            
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(df[time_col], df[value_col], color='darkblue', linewidth=1)
            ax.set_xlabel(time_col, fontsize=12)
            ax.set_ylabel(value_col, fontsize=12)
            ax.set_title(f'Série Temporelle: {value_col}', fontsize=14, fontweight='bold')
            ax.grid(alpha=0.3)
            
            st.pyplot(fig)

# ==================== PAGE COMPARAISON ====================
elif page == "📊 Comparaison des Modèles":
    st.header("Comparaison des Performances des Modèles")
    
    # Données fictives pour la démonstration
    st.info("💡 Chargez vos résultats ou lancez les prédictions pour voir la comparaison")
    
    # Exemple de comparaison
    if st.checkbox("Afficher un exemple de comparaison"):
        models_rmse = {
            "Médiane": 150.23,
            
            "RNN": 87.32,
            "LSTM Stacked": 78.90,
            "BiLSTM": 76.45,
            "MLP": 85.67,
            "CNN": 81.23
        }
        
        # Graphique en barres
        fig, ax = plt.subplots(figsize=(12, 6))
        
        models = list(models_rmse.keys())
        rmse_values = list(models_rmse.values())
        colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
        
        bars = ax.bar(models, rmse_values, color=colors, edgecolor='black', linewidth=1.5)
        
        # Ajouter les valeurs sur les barres
        for bar, value in zip(bars, rmse_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.2f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_xlabel('Modèles', fontsize=13, fontweight='bold')
        ax.set_ylabel('RMSE', fontsize=13, fontweight='bold')
        ax.set_title('Comparaison des RMSE par Modèle', fontsize=15, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        st.pyplot(fig)
        
        # Tableau de comparaison
        st.subheader("📋 Tableau Récapitulatif")
        comparison_df = pd.DataFrame({
            'Modèle': models,
            'RMSE': rmse_values,
            'Rang': range(1, len(models) + 1)
        }).sort_values('RMSE')
        
        st.dataframe(comparison_df, use_container_width=True)
        
        # Meilleur modèle
        best_model = comparison_df.iloc[0]['Modèle']
        best_rmse = comparison_df.iloc[0]['RMSE']
        
        st.success(f"🏆 Meilleur Modèle: **{best_model}** avec RMSE = **{best_rmse:.4f}**")

# Footer
st.divider()
st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>📊 Application de Prévision de Consommation Électrique | Développé avec Streamlit</p>
    </div>
""", unsafe_allow_html=True)