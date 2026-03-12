"""
Module de prédiction — MediPredict
Charge le modèle et le scaler, valide les entrées, et retourne une prédiction.
Aucune donnée utilisateur n'est stockée (traitement en mémoire uniquement).
"""

import joblib
import numpy as np
import pandas as pd
import os

# Chemins des fichiers modèle
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "medipredict_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "model", "feature_names.pkl")


def charger_modele():
    """Charge le modèle, le scaler et les noms de features."""
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    return model, scaler, feature_names


# Plages biologiquement plausibles pour la validation côté serveur
PLAGES_VALIDES = {
    "Pregnancies":              {"min": 0,    "max": 20,   "label": "Nombre de grossesses"},
    "Glucose":                  {"min": 40,   "max": 250,  "label": "Taux de glucose (mg/dL)"},
    "BloodPressure":            {"min": 20,   "max": 140,  "label": "Pression artérielle (mm Hg)"},
    "SkinThickness":            {"min": 5,    "max": 100,  "label": "Épaisseur du pli cutané (mm)"},
    "Insulin":                  {"min": 10,   "max": 900,  "label": "Taux d'insuline (µU/mL)"},
    "BMI":                      {"min": 10.0, "max": 70.0, "label": "Indice de masse corporelle (IMC)"},
    "DiabetesPedigreeFunction": {"min": 0.05, "max": 2.5,  "label": "Antécédents familiaux de diabète"},
    "Age":                      {"min": 18,   "max": 100,  "label": "Âge (années)"},
}

# Tooltips d'aide pour chaque variable
TOOLTIPS = {
    "Pregnancies":              "Nombre total de grossesses. Sélectionnez 'Non applicable' si cette question ne vous concerne pas.",
    "Glucose":                  "Taux de glucose dans le sang mesuré lors d'un test de tolérance au glucose (valeur normale : 70-110 mg/dL).",
    "BloodPressure":            "Pression artérielle diastolique mesurée en mm Hg (valeur normale : 60-80 mm Hg).",
    "SkinThickness":            "Épaisseur du pli cutané au niveau du triceps en mm (valeur typique : 10-50 mm).",
    "Insulin":                  "Taux d'insuline sérique à jeun en µU/mL (valeur normale : 16-166 µU/mL).",
    "BMI":                      "Indice de masse corporelle = poids(kg) / taille(m)². Normal : 18.5-24.9, surpoids : 25-29.9, obésité : ≥30.",
    "DiabetesPedigreeFunction": "Score estimant la prédisposition génétique au diabète basé sur les antécédents familiaux (0 à 2.5).",
    "Age":                      "Votre âge en années.",
}


def valider_entrees(donnees: dict) -> list:
    """
    Valide les entrées utilisateur côté serveur.
    Retourne une liste d'erreurs (vide si tout est valide).
    """
    erreurs = []
    for feature, config in PLAGES_VALIDES.items():
        if feature not in donnees:
            erreurs.append(f"La variable '{config['label']}' est manquante.")
            continue
        val = donnees[feature]
        if val < config["min"] or val > config["max"]:
            erreurs.append(
                f"{config['label']} : la valeur {val} est hors de la plage "
                f"plausible ({config['min']} — {config['max']})."
            )
    return erreurs


def predire(donnees: dict, model, scaler, feature_names):
    """
    Effectue une prédiction à partir des données utilisateur.
    Retourne (probabilité, niveau_risque, couleur).
    Aucune donnée n'est stockée.
    """
    # Construire le DataFrame dans l'ordre attendu par le modèle
    df_input = pd.DataFrame([donnees], columns=feature_names)

    # Normaliser avec le scaler entraîné
    X_scaled = scaler.transform(df_input)

    # Prédiction
    proba = model.predict_proba(X_scaled)[0][1]

    # Déterminer le niveau de risque
    if proba >= 0.6:
        niveau = "Risque élevé"
        couleur = "#e74c3c"
        emoji = "🔴"
    elif proba >= 0.4:
        niveau = "Risque modéré"
        couleur = "#f39c12"
        emoji = "🟠"
    else:
        niveau = "Risque faible"
        couleur = "#2ecc71"
        emoji = "🟢"

    return proba, niveau, couleur, emoji
