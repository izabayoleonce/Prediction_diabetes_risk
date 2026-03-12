import plotly.express as px

def plot_distribution(df, column):
    """Affiche la distribution d'une variable du dataset[cite: 89]."""
    fig = px.histogram(df, x=column, color="Outcome", 
                       title=f"Distribution de {column}",
                       barmode="overlay")
    return fig
"""
Module de visualisation — MediPredict
Fonctions de visualisation pour l'exploration des données et la performance du modèle.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

# Noms français
NOMS_FRANCAIS = {
    "Pregnancies": "Grossesses",
    "Glucose": "Glucose",
    "BloodPressure": "Pression artérielle",
    "SkinThickness": "Épaisseur peau",
    "Insulin": "Insuline",
    "BMI": "IMC",
    "DiabetesPedigreeFunction": "Antécédents fam.",
    "Age": "Âge",
    "Outcome": "Résultat",
}


def plot_distribution_classes(df):
    """Graphique de distribution de la variable cible."""
    fig, ax = plt.subplots(figsize=(8, 4))
    counts = df["Outcome"].value_counts()
    labels = ["Non diabétique", "Diabétique"]
    colors = ["#2ecc71", "#e74c3c"]
    bars = ax.bar(labels, counts.values, color=colors, edgecolor="black", alpha=0.85)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 8, str(val),
                ha="center", fontweight="bold", fontsize=13)
    ax.set_ylabel("Nombre de patients", fontsize=12)
    ax.set_title("Distribution de la variable cible", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_distributions_features(df):
    """Histogrammes des 8 variables, colorés par classe."""
    features = [c for c in df.columns if c != "Outcome"]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()

    for i, col in enumerate(features):
        for outcome, color, label in [(0, "#2ecc71", "Non diabétique"), (1, "#e74c3c", "Diabétique")]:
            axes[i].hist(df[df["Outcome"] == outcome][col], bins=25, alpha=0.6,
                         color=color, label=label, edgecolor="black", linewidth=0.5)
        axes[i].set_title(NOMS_FRANCAIS.get(col, col), fontsize=11, fontweight="bold")
        axes[i].legend(fontsize=7)
    plt.suptitle("Distribution des variables par classe", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


def plot_correlation_matrix(df):
    """Matrice de corrélation."""
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df.rename(columns=NOMS_FRANCAIS).corr().round(2)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, cmap="RdBu_r", center=0, mask=mask,
                square=True, linewidths=1, fmt=".2f", ax=ax, vmin=-1, vmax=1)
    ax.set_title("Matrice de corrélation", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_confusion_matrix(y_true, y_pred):
    """Matrice de confusion."""
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Non-diabétique", "Diabétique"],
                yticklabels=["Non-diabétique", "Diabétique"],
                annot_kws={"size": 16})
    ax.set_ylabel("Réalité", fontsize=12)
    ax.set_xlabel("Prédiction", fontsize=12)
    ax.set_title("Matrice de confusion (Arbre de Décision)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_roc_curve(y_true, y_proba):
    """Courbe ROC."""
    fig, ax = plt.subplots(figsize=(7, 6))
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    ax.plot(fpr, tpr, linewidth=2.5, color="#3498db",
            label=f"Arbre de Décision (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Aléatoire (AUC = 0.5)")
    ax.set_xlabel("Taux de faux positifs (FPR)", fontsize=12)
    ax.set_ylabel("Taux de vrais positifs (TPR)", fontsize=12)
    ax.set_title("Courbe ROC", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="lower right")
    plt.tight_layout()
    return fig


def plot_comparaison_profil(donnees_user, df, feature_names):
    """
    Compare le profil utilisateur à la distribution du dataset.
    Retourne une figure avec 8 histogrammes + la position de l'utilisateur.
    """
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()

    for i, feat in enumerate(feature_names):
        ax = axes[i]
        nom_fr = NOMS_FRANCAIS.get(feat, feat)
        ax.hist(df[feat], bins=25, color="#95a5a6", alpha=0.7, edgecolor="black",
                linewidth=0.5, label="Population du dataset")

        user_val = donnees_user[feat]
        ax.axvline(x=user_val, color="#e74c3c", linewidth=2.5, linestyle="--",
                   label=f"Votre valeur : {user_val:.1f}")
        ax.set_title(nom_fr, fontsize=11, fontweight="bold")
        ax.legend(fontsize=7, loc="upper right")

    plt.suptitle("Votre profil comparé à la distribution du dataset",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig
