# Prédiction du prix des maisons à Boston

Ce projet utilise un modèle de régression linéaire multivariée pour prédire les prix des maisons à Boston.  
Il inclut un exemple de Notebook Jupyter ainsi qu’un script Python pour effectuer des prédictions.

## Structure du projet

project/
├─ data/ # Jeu de données
│ └─ boston.csv
├─ models/ # Modèles entraînés
│ ├─ model.pkl
│ └─ feature_names.pkl
├─ notebooks/ # Notebook Jupyter
│ └─ Multivariable_Regression_and_Valuation_Model(start).ipynb
├─ app.py # Exemple d'application Web (optionnel)
├─ predict.py # Script de prédiction
├─ requirements.txt # Dépendances du projet
└─ README.md # Documentation


## Installation et utilisation

1. Installer les dépendances :
   ```bash
   pip install -r requirements.txt

2. Lancer la prédiction :
python predict.py

3. Exemple de sortie :
Prix prédit : 25000.00 USD

