⚡ Prévision de la Consommation Électrique

Projet réalisé par : Salah Eddine Khamraoui

📌 Description du Projet

Cette application web permet de prédire la consommation électrique en utilisant plusieurs modèles de Machine Learning et Deep Learning (RNN, LSTM, BiLSTM, CNN, MLP…).
Développée avec Streamlit, elle offre une interface intuitive pour :

Charger des fichiers CSV

Explorer et nettoyer les données

Générer des prédictions

Comparer les performances de différents modèles

Faire des prédictions personnalisées en entrant vos propres valeurs

L’interface est optimisée pour supprimer les warnings TensorFlow et scikit-learn afin de garantir une utilisation fluide.

🚀 Fonctionnalités Principales
✔️ Multi-modèles intégrés

Médiane (baseline)

RNN

LSTM Stacked

BiLSTM

MLP

CNN

K-Means, DBSCAN

SARIMA (si disponible)

✔️ Pages interactives

🏠 Accueil

📈 Prédictions sur fichier CSV

🎯 Prédiction personnalisée

🔍 Exploration des données

📊 Comparaison des modèles

✔️ Prétraitement automatique des données

Nettoyage des valeurs aberrantes

Gestion des valeurs manquantes

Normalisation MinMax

Création automatique de séquences temporelles

✔️ Visualisations intégrées

Graphiques réels vs prédictions

Histogrammes

Série temporelle

Tableaux récapitulatifs

🛠️ Technologies Utilisées

Python

Streamlit

TensorFlow / Keras

scikit-learn

Pandas / NumPy

Matplotlib

Joblib

📂 Organisation du Projet
📁 Electric-Consumption-Prediction
     │── app.py                  # Code principal Streamlit
     │── median_model.pkl        # Modèle baseline
     │── rnn_model.h5            # Modèle RNN
     │── lstm_stacked_model.h5   # Modèle LSTM empilé
     │── bilstm_model.h5         # Modèle BiLSTM
     │── mlp_model.h5            # Modèle MLP
     │── cnn_model.h5            # Modèle CNN
     │── kmeans_model.pkl        # Modèle KMeans
     │── dbscan_model.pkl        # Modèle DBSCAN
     │── scaler.pkl              # Scaler MinMax
     │── requirements.txt        # Dépendances Python
     │── README.md               # Documentation

▶️ Comment exécuter l’application ?
1️⃣ Installer les dépendances
pip install -r requirements.txt

2️⃣ Lancer l'application Streamlit
streamlit run app.py

3️⃣ Ouvrir dans le navigateur

Streamlit s’ouvrira automatiquement à l’adresse :

http://localhost:8501

🧪 Données

Le projet fonctionne avec n’importe quel fichier CSV contenant des séries temporelles de consommation électrique.
L’utilisateur peut sélectionner la colonne cible dans l’application.

🧠 Modèles Deep Learning

Les modèles suivants ont été optimisés pour des tâches de prévision univariée :

RNN simple

LSTM empilé

BiLSTM

CNN 1D

MLP

Ils utilisent des séquences de longueur variable (paramètre n_steps).

📊 Comparaison des Modèles

L’application calcule et affiche :

RMSE

MAE

R² Score

Graphiques interactifs

Classement automatique des modèles

👤 Auteur

Salah Eddine Khamraoui
📧 salaheddine.khamraoui@etu.uae.ac.ma
💼 Salah Eddine Khamraoui

📜 Licence

Ce projet est publié sous licence libre (à préciser : MIT, GPL, etc.)
