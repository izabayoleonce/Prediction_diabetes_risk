"""
Module d'explicabilité — MediPredict
Génère les explications SHAP et les traduit en langage naturel.
"""

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Noms français des variables
NOMS_FRANCAIS = {
    "Pregnancies":              "Nombre de grossesses",
    "Glucose":                  "Taux de glucose",
    "BloodPressure":            "Pression artérielle",
    "SkinThickness":            "Épaisseur du pli cutané",
    "Insulin":                  "Taux d'insuline",
    "BMI":                      "Indice de masse corporelle (IMC)",
    "DiabetesPedigreeFunction": "Antécédents familiaux de diabète",
    "Age":                      "Âge",
}

# Recommandations par variable (facteurs modifiables)
RECOMMANDATIONS = {
    "Glucose": "Un taux de glucose élevé est un facteur de risque modifiable. "
               "Des ajustements alimentaires (réduction des sucres rapides, "
               "alimentation riche en fibres) et une activité physique régulière "
               "peuvent aider à le réduire.",
    "BMI":     "Un indice de masse corporelle élevé augmente le risque de diabète. "
               "Une alimentation équilibrée et une activité physique de 30 minutes "
               "par jour peuvent contribuer à atteindre un poids santé.",
    "BloodPressure": "Une pression artérielle élevée est souvent associée au risque "
                     "de diabète. Réduire le sel, gérer le stress et pratiquer une "
                     "activité physique régulière sont des pistes bénéfiques.",
    "Insulin": "Un taux d'insuline anormal peut indiquer une résistance à l'insuline. "
               "Consultez un professionnel de santé pour un bilan approfondi.",
    "Age":     "L'âge est un facteur non modifiable, mais un suivi médical régulier "
               "devient d'autant plus important avec les années.",
    "Pregnancies": "Les grossesses multiples peuvent augmenter le risque de diabète "
                   "gestationnel. Un suivi médical régulier est recommandé.",
    "SkinThickness": "Cette mesure reflète la composition corporelle. Maintenir une "
                     "activité physique régulière contribue à un meilleur équilibre.",
    "DiabetesPedigreeFunction": "Les antécédents familiaux sont un facteur non modifiable, "
                                "mais connaître son historique familial permet d'adopter "
                                "une prévention proactive.",
}


def calculer_shap(model, X_scaled, feature_names):
    """
    Calcule les valeurs SHAP pour une observation.
    Gère le format 3D retourné par l'Arbre de Décision.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)

    # Gestion du format : 3D (n, features, classes) ou liste
    if isinstance(shap_values, list):
        shap_class1 = shap_values[1]
        base_val = explainer.expected_value[1]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_class1 = shap_values[:, :, 1]
        base_val = explainer.expected_value[1]
    else:
        shap_class1 = shap_values
        base_val = explainer.expected_value

    return shap_class1[0], base_val, explainer


def generer_explication_naturelle(shap_values_patient, feature_names, donnees, proba):
    """
    Génère une explication en langage naturel à partir des valeurs SHAP.
    """
    # Trier par impact absolu décroissant
    indices_tries = np.argsort(np.abs(shap_values_patient))[::-1]

    # Top 3 des facteurs
    top3 = indices_tries[:3]
    facteurs_positifs = []
    facteurs_negatifs = []

    for idx in top3:
        feat = feature_names[idx]
        nom_fr = NOMS_FRANCAIS.get(feat, feat)
        val = donnees[feat]
        shap_val = shap_values_patient[idx]

        if shap_val > 0:
            facteurs_positifs.append((nom_fr, val, shap_val))
        else:
            facteurs_negatifs.append((nom_fr, val, shap_val))

    # Construction du texte
    texte = ""
    if proba >= 0.6:
        texte = "Votre profil présente un **risque élevé** de diabète de type 2. "
        if facteurs_positifs:
            f = facteurs_positifs[0]
            texte += f"Le facteur principal est votre **{f[0].lower()}** ({f[1]:.1f}), "
            texte += "qui est supérieur(e) à la moyenne. "
            if len(facteurs_positifs) > 1:
                f2 = facteurs_positifs[1]
                texte += f"Votre **{f2[0].lower()}** ({f2[1]:.1f}) contribue également à ce risque. "
        texte += "**Nous vous recommandons de consulter un professionnel de santé.**"
    elif proba >= 0.4:
        texte = "Votre profil présente un **risque modéré**. "
        if facteurs_positifs:
            f = facteurs_positifs[0]
            texte += f"Votre **{f[0].lower()}** ({f[1]:.1f}) est le facteur "
            texte += "qui influence le plus cette estimation. "
        texte += "Un suivi régulier est conseillé."
    else:
        texte = "Votre profil présente un **risque faible** de diabète de type 2. "
        if facteurs_negatifs:
            f = facteurs_negatifs[0]
            texte += f"Votre **{f[0].lower()}** contribue positivement à ce résultat favorable. "
        texte += "Continuez à maintenir un mode de vie sain."

    return texte


def generer_recommandations(shap_values_patient, feature_names):
    """
    Génère des recommandations basées sur les variables les plus influentes.
    """
    indices_tries = np.argsort(np.abs(shap_values_patient))[::-1]
    recommandations = []

    for idx in indices_tries[:3]:
        feat = feature_names[idx]
        if feat in RECOMMANDATIONS:
            recommandations.append({
                "variable": NOMS_FRANCAIS.get(feat, feat),
                "impact": "augmente" if shap_values_patient[idx] > 0 else "diminue",
                "texte": RECOMMANDATIONS[feat],
            })

    return recommandations


def generer_waterfall_plot(shap_values_patient, base_value, feature_values, feature_names):
    """
    Génère un waterfall plot SHAP pour un patient.
    Retourne la figure matplotlib.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.sca(ax)

    explanation = shap.Explanation(
        values=shap_values_patient,
        base_values=base_value,
        data=feature_values,
        feature_names=[NOMS_FRANCAIS.get(f, f) for f in feature_names],
    )
    shap.waterfall_plot(explanation, show=False)
    ax.set_title("Contribution de chaque variable à votre résultat",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig
