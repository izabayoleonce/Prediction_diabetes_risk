# 🏥 MediPredict — Prédiction du risque de diabète de type 2

## Description

**MediPredict** est une application web de sensibilisation au risque de diabète de type 2,
développée pour la mutuelle fictive **SantéCo**. Elle permet à un utilisateur de renseigner
des indicateurs de santé anonymes et d'obtenir une estimation de son niveau de risque,
accompagnée d'explications claires et de recommandations.

> ⚠️ **Cet outil est un outil de sensibilisation. Il ne constitue pas un avis médical.
> En cas de doute, consultez un professionnel de santé.**

## Fonctionnalités

| Page | Description |
|------|-------------|
| 🏠 Accueil | Présentation du service, mention légale, consentement |
| 📋 Mon profil de risque | Formulaire de saisie + jauge de risque visuelle |
| 🔍 Comprendre ma prédiction | Explications SHAP, comparaison au dataset, recommandations |
| 📊 Explorer les données | Visualisations, performance du modèle, transparence |

## Installation

```bash
# Cloner le dépôt
git clone https://github.com/votre-username/medipredict.git
cd medipredict

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Installer les dépendances
pip install -r requirements.txt

# Lancer l'application
streamlit run app.py
```

## Structure du projet

```
medipredict/
├── app.py                  # Point d'entrée Streamlit
├── model/
│   ├── medipredict_model.pkl  # Arbre de Décision entraîné
│   ├── scaler.pkl             # StandardScaler
│   └── feature_names.pkl      # Noms des features
├── src/
│   ├── __init__.py
│   ├── predict.py          # Logique de prédiction + validation
│   ├── explain.py          # Génération SHAP + langage naturel
│   └── visualize.py        # Fonctions de visualisation
├── data/
│   └── diabetes.csv        # Dataset public Pima Indians
├── requirements.txt
├── .gitignore
├── .env.example
└── README.md
```

## Modèle

- **Algorithme** : Arbre de Décision (max_depth=5, class_weight=balanced)
- **Dataset** : Pima Indians Diabetes Dataset (768 obs., 8 features)
- **Performances** : Accuracy 88.3%, Rappel 92.6%, F1-score 84.8%, AUC-ROC 0.925

## Conformité

- **RGPD** : Aucune donnée stockée, consentement explicite, traitement en mémoire
- **Accessibilité WCAG 2.1 AA** : Contraste 4.5:1, police 16px min, navigation clavier, libellés texte sur la jauge
- **Éthique** : Mention légale obligatoire, pas de diagnostic, explications SHAP, biais documentés

## Technologies

- Python 3.10+
- Streamlit
- scikit-learn
- SHAP
- matplotlib / seaborn

## Licence

Projet pédagogique — Usage éducatif uniquement.
